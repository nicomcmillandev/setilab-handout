#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <assert.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include "filter.h"
#include "signal.h"
#include "timing.h"

#define MAXWIDTH 40
#define THRESHOLD 2.0
#define ALIENS_LOW  50000.0
#define ALIENS_HIGH 150000.0

typedef struct {
    int id;
    int t_count;
    int p_count;
    int bands;
    signal* sig;
    int order;
    double bw;
    double* p_out;
} t_data;

void usage() {
  printf("usage: p_band_scan text|bin|mmap signal_file Fs filter_order num_bands num_threads num_processors\n");
}

double avg_power(double* data, int num) {
  double ss = 0;
  for (int i = 0; i < num; i++) {
    ss += data[i] * data[i];
  }
  return ss / num;
}

double max_of(double* data, int num) {
  double m = data[0];
  for (int i = 1; i < num; i++) {
    if (data[i] > m) {
      m = data[i];
    }
  }
  return m;
}

double avg_of(double* data, int num) {
  double s = 0;
  for (int i = 0; i < num; i++) {
    s += data[i];
  }
  return s / num;
}

void remove_dc(double* data, int num) {
  double dc = avg_of(data,num);
  printf("Removing DC component of %lf\n",dc);
  for (int i = 0; i < num; i++) {
    data[i] -= dc;
  }
}

void* worker(void* arg) {
    t_data* data = (t_data*)arg;

    // round robin affinity
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(data->id % data->p_count, &cpuset);
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) < 0) {
        perror("Can't setaffinity");
        exit(-1);
    }

    // interleave the bands
    for (int b = data->id; b < data->bands; b += data->t_count) {
        double* coeffs = (double*)malloc((data->order + 1) * sizeof(double));
        
        generate_band_pass(data->sig->Fs,
                           b * data->bw + 0.0001,
                           (b + 1) * data->bw - 0.0001,
                           data->order,
                           coeffs);
                           
        hamming_window(data->order, coeffs);

        convolve_and_compute_power(data->sig->num_samples,
                                   data->sig->data,
                                   data->order,
                                   coeffs,
                                   &(data->p_out[b]));
                                   
        free(coeffs);
    }
    return NULL;
}

int analyze_signal(signal* sig, int filter_order, int num_bands, int t_count, int p_count, double* lb, double* ub) {
  double Fc        = (sig->Fs) / 2;
  double bandwidth = Fc / num_bands;

  remove_dc(sig->data,sig->num_samples);

  double signal_power = avg_power(sig->data,sig->num_samples);
  printf("signal average power:     %lf\n", signal_power);

  resources rstart;
  get_resources(&rstart,THIS_PROCESS);
  double start = get_seconds();
  unsigned long long tstart = get_cycle_count();

  double* band_power = (double*)malloc(num_bands * sizeof(double));
  pthread_t* threads = (pthread_t*)malloc(t_count * sizeof(pthread_t));
  t_data* args = (t_data*)malloc(t_count * sizeof(t_data));

  for (int i = 0; i < t_count; i++) {
      args[i].id = i;
      args[i].t_count = t_count;
      args[i].p_count = p_count;
      args[i].bands = num_bands;
      args[i].sig = sig;
      args[i].order = filter_order;
      args[i].bw = bandwidth;
      args[i].p_out = band_power;

      if (pthread_create(&threads[i], NULL, worker, (void*)&args[i]) != 0) {
          perror("thread creation failed");
          exit(-1);
      }
  }

  for (int i = 0; i < t_count; i++) {
      pthread_join(threads[i], NULL);
  }

  unsigned long long tend = get_cycle_count();
  double end = get_seconds();

  resources rend;
  get_resources(&rend,THIS_PROCESS);

  resources rdiff;
  get_resources_diff(&rstart, &rend, &rdiff);

  double max_band_power = max_of(band_power,num_bands);
  double avg_band_power = avg_of(band_power,num_bands);
  int wow = 0;
  *lb = -1;
  *ub = -1;

  for (int band = 0; band < num_bands; band++) {
    double band_low  = band * bandwidth + 0.0001;
    double band_high = (band + 1) * bandwidth - 0.0001;

    printf("%5d %20lf to %20lf Hz: %20lf ",
           band, band_low, band_high, band_power[band]);

    for (int i = 0; i < MAXWIDTH * (band_power[band] / max_band_power); i++) {
      printf("*");
    }

    if ((band_low >= ALIENS_LOW && band_low <= ALIENS_HIGH) ||
        (band_high >= ALIENS_LOW && band_high <= ALIENS_HIGH)) {
      if (band_power[band] > THRESHOLD * avg_band_power) {
        printf("(WOW)");
        wow = 1;
        if (*lb < 0) {
          *lb = band * bandwidth + 0.0001;
        }
        *ub = (band + 1) * bandwidth - 0.0001;
      } else {
        printf("(meh)");
      }
    } else {
      printf("(meh)");
    }

    printf("\n");
  }

  printf("Resource usages:\n\
User time        %lf seconds\n\
System time      %lf seconds\n\
Page faults      %ld\n\
Page swaps       %ld\n\
Blocks of I/O    %ld\n\
Signals caught   %ld\n\
Context switches %ld\n",
         rdiff.usertime,
         rdiff.systime,
         rdiff.pagefaults,
         rdiff.pageswaps,
         rdiff.ioblocks,
         rdiff.sigs,
         rdiff.contextswitches);

  printf("Analysis took %llu cycles (%lf seconds) by cycle count, timing overhead=%llu cycles\n"
         "Note that cycle count only makes sense if the thread stayed on one core\n",
         tend - tstart, cycles_to_seconds(tend - tstart), timing_overhead());
  printf("Analysis took %lf seconds by basic timing\n", end - start);

  free(band_power);
  free(threads);
  free(args);

  return wow;
}

int main(int argc, char* argv[]) {

  if (argc != 8) {
    usage();
    return -1;
  }

  char sig_type    = toupper(argv[1][0]);
  char* sig_file   = argv[2];
  double Fs        = atof(argv[3]);
  int filter_order = atoi(argv[4]);
  int num_bands    = atoi(argv[5]);
  int num_threads  = atoi(argv[6]);
  int num_procs    = atoi(argv[7]);

  assert(Fs > 0.0);
  assert(filter_order > 0 && !(filter_order & 0x1));
  assert(num_bands > 0);

  printf("type:     %s\n\
file:     %s\n\
Fs:       %lf Hz\n\
order:    %d\n\
bands:    %d\n",
         sig_type == 'T' ? "Text" : (sig_type == 'B' ? "Binary" : (sig_type == 'M' ? "Mapped Binary" : "UNKNOWN TYPE")),
         sig_file,
         Fs,
         filter_order,
         num_bands);

  printf("Load or map file\n");

  signal* sig;
  switch (sig_type) {
    case 'T':
      sig = load_text_format_signal(sig_file);
      break;
    case 'B':
      sig = load_binary_format_signal(sig_file);
      break;
    case 'M':
      sig = map_binary_format_signal(sig_file);
      break;
    default:
      printf("Unknown signal type\n");
      return -1;
  }

  if (!sig) {
    printf("Unable to load or map file\n");
    return -1;
  }

  sig->Fs = Fs;

  double start = 0;
  double end   = 0;
  
  if (analyze_signal(sig, filter_order, num_bands, num_threads, num_procs, &start, &end)) {
    printf("POSSIBLE ALIENS %lf-%lf HZ (CENTER %lf HZ)\n", start, end, (end + start) / 2.0);
  } else {
    printf("no aliens\n");
  }

  free_signal(sig);
  return 0;
}