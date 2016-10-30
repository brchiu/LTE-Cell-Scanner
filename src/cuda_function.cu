#include <itpp/base/array.h>
#include <itpp/base/converters.h>
#include <itpp/base/itassert.h>
#include <itpp/base/mat.h>
#include <itpp/base/matfunc.h>
#include <itpp/base/random.h>
#include <itpp/base/vec.h>
#include <itpp/base/math/elem_math.h>
#include <boost/math/special_functions/gamma.hpp>
#include <sys/time.h>
#include <cmath>
#include <list>
#include <iomanip>
#include <algorithm>
#include <vector>

#include "macros.h"
#include "common.h"
#include "lte_lib.h"
#include "constants.h"
#include "dsp.h"
#include "itpp_ext.h"

#include <math_constants.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <helper_cuda.h>

using namespace std;
using namespace itpp;

__constant__ cufftComplex pss_td[3][256];

#define THREAD_DIM_X   (256)

extern "C" void copy_pss_to_device()
{
    int i, t, len;
    cufftComplex pss[3][256];
  
    for (t = 0; t < 3; t++) {
        len = ROM_TABLES.pss_td[t].length();
        for (i = 0; i < len; i++) {
            pss[t][i].x = ROM_TABLES.pss_td[t][i].real();
            pss[t][i].y = ROM_TABLES.pss_td[t][i].imag();
        }
        for (; i < 256; i++) {
            pss[t][i].x = 0.0f;
            pss[t][i].y = 0.0f;
        }
    }
    checkCudaErrors(cudaMemcpyToSymbol(pss_td, &pss, sizeof(pss)));
}

#define COMPLEX_MUL_REAL(a, b)  ((a).x * (b).x - (a).y * (b).y)
#define COMPLEX_MUL_IMAG(a, b)  ((a).x * (b).y + (a).y * (b).x)

__global__ void xc_correlate_kernel(cufftComplex *d_capbuf, float *d_xc_sqr, 
                                    float *d_xc_incoherent_single, float *d_xc_incoherent,
                                    uint8 t, unsigned int n_cap, uint8 ds_comb_arm, 
                                  double f, double fs)
{
    __shared__ cufftComplex s_fshift_pss[THREAD_DIM_X], s_capbuf[THREAD_DIM_X + 137];

    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    double k = CUDART_PI * f * 2 / fs;
    double shift = k * tid;
    double x1 = cos(shift), y1 = sin(shift);
    double x2 = pss_td[t][tid].x, y2 = pss_td[t][tid].y;
    unsigned int max_m = (n_cap - 100 - 136) / 9600;
    unsigned int i, m;
 
    s_fshift_pss[tid].x = x1*x2 - y1*y2;
    s_fshift_pss[tid].y = -x1*y2 - x2*y1;

    s_capbuf[tid] = d_capbuf[THREAD_DIM_X * bid + tid];

    if (tid < 137) {
        if (THREAD_DIM_X * bid + THREAD_DIM_X + tid < n_cap) {
            s_capbuf[THREAD_DIM_X + tid] = d_capbuf[THREAD_DIM_X * bid + THREAD_DIM_X + tid];
        } else {
            s_capbuf[THREAD_DIM_X + tid] = d_capbuf[tid];
        }
    }
  
    __syncthreads();

    float real, imag;

    real = COMPLEX_MUL_REAL(s_fshift_pss[0], s_capbuf[tid]);
    imag = COMPLEX_MUL_IMAG(s_fshift_pss[0], s_capbuf[tid]);
    for (i = 1; i < 137; i++) {
        real += COMPLEX_MUL_REAL(s_fshift_pss[i], s_capbuf[tid + i]);
        imag += COMPLEX_MUL_IMAG(s_fshift_pss[i], s_capbuf[tid + i]);
    }
    d_xc_sqr[THREAD_DIM_X * bid + tid] = (real * real + imag * imag) / (137.0*137.0);

    __syncthreads();

    if (tid < 16) {
        unsigned int index = 16 * bid + tid;
        float xc_incoherent_single_val = d_xc_sqr[index];
        for (m = 1; m < max_m; m++) {
            unsigned int span = m * 0.005 * fs;
            xc_incoherent_single_val += d_xc_sqr[index + span];
        }
        float xc_incoherent_value = d_xc_incoherent_single[index] = xc_incoherent_single_val / max_m;

        __syncthreads();

        for (i = 1; i <= ds_comb_arm; i++) {
            if (index + i < 9600) {
                xc_incoherent_value += d_xc_incoherent_single[index + i];
            } else {
                xc_incoherent_value += d_xc_incoherent_single[index + i - 9600];
            }
            if (index > i) {
                xc_incoherent_value += d_xc_incoherent_single[index - i];
            } else {
                xc_incoherent_value += d_xc_incoherent_single[index - i + 9600];
            }
        }
        d_xc_incoherent[index] = xc_incoherent_value / (ds_comb_arm * 2 + 1);
    }

    __syncthreads();
}

void xc_correlate_step(const cvec & capbuf, vec &xc_sqr, vf3d & xc_incoherent_single, vf3d & xc_incoherent, uint8 t, uint16 foi, double f, double fs, uint8 ds_comb_arm)
{
    cufftComplex *h_capbuf;
    float *h_xc_sqr, *h_xc_incoherent_single, *h_xc_incoherent;

    unsigned int n_cap = capbuf.length();
    cufftComplex *d_capbuf = (cufftComplex *)NULL;
    float *d_xc_sqr = (float *)NULL;
    float *d_xc_incoherent_single = (float *)NULL;
    float *d_xc_incoherent = (float *)NULL;
 
    h_capbuf = (cufftComplex *)malloc(n_cap * sizeof(cufftComplex));
    h_xc_sqr = (float *)malloc(n_cap * sizeof(float));
    h_xc_incoherent_single = (float *)malloc(9600 * sizeof(float));
    h_xc_incoherent = (float *)malloc(9600 * sizeof(float));
 
    checkCudaErrors(cudaMalloc(&d_capbuf, n_cap * sizeof(cufftComplex)));
    checkCudaErrors(cudaMalloc(&d_xc_sqr, n_cap * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_xc_incoherent_single, 9600 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_xc_incoherent, 9600 * sizeof(float)));

    for (unsigned int i = 0; i < n_cap; i++) {
        h_capbuf[i].x = capbuf[i].real();
        h_capbuf[i].y = capbuf[i].imag();
    }
    checkCudaErrors(cudaMemcpy(d_capbuf, h_capbuf, n_cap * sizeof(cufftComplex), cudaMemcpyHostToDevice));

    xc_correlate_kernel<<<600, 256>>>(d_capbuf, d_xc_sqr, d_xc_incoherent_single, d_xc_incoherent, t, n_cap, ds_comb_arm, f, fs);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(h_xc_sqr, d_xc_sqr, n_cap * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_xc_incoherent_single, d_xc_incoherent_single, 9600 * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_xc_incoherent, d_xc_incoherent, 9600 * sizeof(float), cudaMemcpyDeviceToHost));

    for (unsigned int i = 0; i < n_cap - 136; i++) {
        xc_sqr[i] = h_xc_sqr[i];
    }

    for (unsigned int i = 0; i < 9600; i++) {
        xc_incoherent_single[t][i][foi] = h_xc_incoherent_single[i];
    }
    for (unsigned int i = 0; i < 9600; i++) {
        xc_incoherent[t][i][foi] = h_xc_incoherent[i];
    }

    free(h_capbuf);
    free(h_xc_sqr);
    free(h_xc_incoherent_single);
    free(h_xc_incoherent);
    checkCudaErrors(cudaFree(d_capbuf));
    checkCudaErrors(cudaFree(d_xc_sqr));
    checkCudaErrors(cudaFree(d_xc_incoherent_single));
    checkCudaErrors(cudaFree(d_xc_incoherent));
}


void xcorr_pss2(
  // Inputs
  const cvec & capbuf,
  const vec & f_search_set,
  const uint8 & ds_comb_arm,
  const double & fc_requested,
  const double & fc_programmed,
  const double & fs_programmed,
  // Outputs
  mat & xc_incoherent_collapsed_pow,
  imat & xc_incoherent_collapsed_frq,
  // Following used only for debugging...
  vf3d & xc_incoherent_single,
  vf3d & xc_incoherent,
  vec & sp_incoherent,
  vcf3d & xc,
  vec & sp,
  uint16 & n_comb_xc,
  uint16 & n_comb_sp
) 
{
  // Perform correlations
  const uint32 n_cap=length(capbuf);
  const uint16 n_f=length(f_search_set);

  // Set aside space for the vector and initialize with NAN's.
#ifndef NDEBUG
  xc = vector < vector < vector < complex < float > > > > (3,vector< vector < complex < float > > >(n_cap-136, vector < complex < float > > (n_f,NAN)));
  vcf3d xc2 = vector < vector < vector < complex < float > > > > (3,vector< vector < complex < float > > >(n_cap-136, vector < complex < float > > (n_f,NAN)));
  xc_incoherent_single = vector < vector < vector < float > > > (3,vector< vector < float > >(9600, vector < float > (n_f,NAN)));
  xc_incoherent = vector < vector < vector < float > > > (3,vector< vector < float > >(9600, vector < float > (n_f,NAN)));
#else
  xc = vector < vector < vector < complex < float > > > > (3,vector< vector < complex < float > > >(n_cap-136, vector < complex < float > > (n_f)));
  vcf3d xc2 = vector < vector < vector < complex < float > > > > (3,vector< vector < complex < float > > >(n_cap-136, vector < complex < float > > (n_f)));
  xc_incoherent_single = vector < vector < vector < float > > > (3,vector< vector < float > >(9600, vector < float > (n_f)));
  xc_incoherent = vector < vector < vector < float > > > (3,vector< vector < float > >(9600, vector < float > (n_f)));
#endif
  vec xc_sqr = vec(n_cap-136);

  // Local variables declared outside of the loop.
  double f_off;
  cvec temp;
  complex <double> acc;
  uint16 foi;
  uint8 t;
  uint32 k;
  uint8 m;

  struct timeval tv1, tv2;
  gettimeofday(&tv1, NULL);

  // Loop and perform correlations.
  // Incoherently combine correlations
  n_comb_xc = floor_i((xc[0].size()-100)/9600);
  const uint8 ds_com_arm_weight = (2*ds_comb_arm+1);
  printf("n_cap=%d\n", n_cap);

  for (foi=0;foi<n_f;foi++) {
    f_off = f_search_set(foi);
    const double k_factor = (fc_requested-f_off)/fc_programmed;
    for (t = 0; t < 3;t++) {
#if 1
      xc_correlate_step(capbuf, xc_sqr, xc_incoherent_single, xc_incoherent, t, foi, f_off, fs_programmed*k_factor, ds_comb_arm);
#else
      temp = ROM_TABLES.pss_td[t];
      temp = fshift(temp,f_off,fs_programmed*k_factor);
      temp = conj(temp)/137;
#ifdef _OPENMP
#pragma omp parallel for shared(temp,capbuf,xc) private(k,acc,m)
#endif
      for (k=0;k<n_cap-136;k++) {
        acc=0;
        for (m=0;m<137;m++) {
          // Correlations are performed at the 2x rate which effectively
          // performs filtering and correlating at the same time. Thus,
          // this algorithm can handle huge frequency offsets limited only
          // by the bandwidth of the capture device.
          // Correlations can also be done at the 1x rate if filtering is
          // peformed first, but this will limit the set of frequency offsets
          // that this algorithm can detect. 1x rate correlations will,
          // however, be nearly twice as fast as the 2x correlations
          // performed here.
          acc+=temp(m)*capbuf(k+m);
        }
        xc[t][k][foi]=acc;
      }

      for (uint16 idx=0;idx<9600;idx++) {
        // Because of the large supported frequency offsets and the large
        // amount of time represented by the capture buffer, the length
        // in samples, of a frame varies by the frequency offset.
        //double actual_time_offset=m*.005*k_factor;
        //double actual_start_index=itpp::round_i(actual_time_offset*FS_LTE/16);
        xc_incoherent_single[t][idx][foi] = 0;
        for (uint16 m = 0; m < n_comb_xc; m++) {
          uint32 actual_start_index = itpp::round_i(m*.005*k_factor*fs_programmed);
          xc_incoherent_single[t][idx][foi] += xc[t][idx + actual_start_index][foi].real() * xc[t][idx + actual_start_index][foi].real() + xc[t][idx + actual_start_index][foi].imag() * xc[t][idx + actual_start_index][foi].imag();
        }
        xc_incoherent_single[t][idx][foi]/= n_comb_xc;
      }

      for (uint16 idx=0;idx<9600;idx++) {
        xc_incoherent[t][idx][foi] = xc_incoherent_single[t][idx][foi];
        for (uint8 k=1;k<=ds_comb_arm;k++) {
          xc_incoherent[t][idx][foi] += (xc_incoherent_single[t][itpp_ext::matlab_mod(idx-k,9600)][foi] + xc_incoherent_single[t][itpp_ext::matlab_mod(idx+k,9600)][foi]);
        }
        xc_incoherent[t][idx][foi] /= ds_com_arm_weight;
      }
#endif
    }
  }

  // Estimate received signal power
  // const uint32 n_cap=length(capbuf);
  n_comb_sp = floor_i((n_cap-136-137)/9600);
  const uint32 n_sp = n_comb_sp*9600;

  // Set aside space for the vector and initialize with NAN's.
  sp = vec(n_sp);
  xc_incoherent_collapsed_pow = mat(3,9600);
  xc_incoherent_collapsed_frq = imat(3,9600);
#ifndef NDEBUG
  sp = NAN;
  xc_incoherent_collapsed_pow = NAN;
  xc_incoherent_collapsed_frq = -1;
#endif
  sp[0] = 0;
  // Estimate power for first time offset
  for (uint16 t=0;t<274;t++) {
    sp[0] += pow(capbuf[t].real(),2) + pow(capbuf[t].imag(),2);
  }
  sp[0] = sp[0] / 274;
  // Estimate RX power for remaining time offsets.
  for (uint32 t=1;t<n_sp;t++) {
    sp[t] = sp[t-1] + (-pow(capbuf[t-1].real(),2)-pow(capbuf[t-1].imag(),2)+pow(capbuf[t+274-1].real(),2)+pow(capbuf[t+274-1].imag(),2))/274;
  }

  // Combine incoherently
  sp_incoherent = sp.left(9600);
  for (uint16 t=1; t < n_comb_sp;t++) {
    sp_incoherent += sp.mid(t*9600, 9600);
  }
  sp_incoherent = sp_incoherent / n_comb_sp;

  // Shift to the right by 137 samples to align with the correlation peaks.
  tshift(sp_incoherent, 137);

  // Search for peaks among all the frequency offsets.
  // const int n_f=xc_incoherent[0][0].size();

  for (uint8 t=0;t<3;t++) {
    for (uint16 k=0;k<9600;k++) {
      double best_pow=xc_incoherent[t][k][0];
      uint16 best_idx=0;
      for (uint16 foi=1;foi<n_f;foi++) {
        if (xc_incoherent[t][k][foi]>best_pow) {
          best_pow=xc_incoherent[t][k][foi];
          best_idx=foi;
        }
      }
      xc_incoherent_collapsed_pow(t,k)=best_pow;
      xc_incoherent_collapsed_frq(t,k)=best_idx;
    }
  }

  gettimeofday(&tv2, NULL);
  printf("xcorr_pss2 : %ld us\n", (tv2.tv_sec-tv1.tv_sec)*1000000+(tv2.tv_usec-tv1.tv_usec));
}



