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
                                    unsigned int n_cap, uint8 ds_comb_arm, 
                                    unsigned int t, double f, double fs)
{
    __shared__ cufftComplex s_fshift_pss[256], s_capbuf[256 + 137];

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

    s_capbuf[tid] = d_capbuf[256 * bid + tid];

    if (tid < 137) {
        if (256 * bid + 256 + tid < n_cap) {
            s_capbuf[256 + tid] = d_capbuf[256 * bid + 256 + tid];
        } else {
            s_capbuf[256 + tid] = d_capbuf[tid];
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
    d_xc_sqr[256 * bid + tid] = (real * real + imag * imag) / (137.0*137.0);

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


__global__ void xc_incoherent_collapsed_kernel(float *d_xc_incoherent, 
                                               float *d_xc_incoherent_collapsed_pow, int *d_xc_incoherent_collapsed_frq,
                                               unsigned int n_f)
{
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    float best_pow = d_xc_incoherent[(0 * 3 + tid) * 9600 + bid];
    unsigned int best_index = 0;

    for (unsigned int foi = 1; foi < n_f; foi++) {
        if (d_xc_incoherent[(foi * 3 + tid) * 9600 + bid] > best_pow) {
            best_pow = d_xc_incoherent[(foi * 3 + tid) * 9600 + bid];
            best_index = foi;
        }
    }

    d_xc_incoherent_collapsed_pow[tid * 9600 + bid] = best_pow;
    d_xc_incoherent_collapsed_frq[tid * 9600 + bid] = best_index;
}


__global__ void sp_incoherent_kernel(cufftComplex *d_capbuf, float *d_sp_incoherent, unsigned int n_cap)
{
    __shared__ float s_sqr[512];

    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int n_comb_sp = (n_cap - 136 - 137) / 9600;
    unsigned int index = bid * 16 + tid;
    float value;

    if (tid < 274 + 16) {
       value = d_capbuf[index].x * d_capbuf[index].x + d_capbuf[index].y * d_capbuf[index].y;
       for (unsigned int m = 1; m < n_comb_sp; m++) {
           value += (d_capbuf[index + 9600 * m].x * d_capbuf[index + 9600 * m].x + d_capbuf[index + 9600 * m].y * d_capbuf[index + 9600 * m].y);
       }
       s_sqr[tid] = value;
    } else {
       s_sqr[tid] = 0.0f;
    }

    __syncthreads();

    if (tid < 16) {
        value = s_sqr[tid];
        for (unsigned int k = 1; k < 274; k++) {
             value += s_sqr[tid + k];
        }
        index += 137;
        if (index >= 9600)
            index -= 9600;
        d_sp_incoherent[index] = value / (274.0 * n_comb_sp);
    }

    __syncthreads();
}

void xcorr_pss2(const cvec & capbuf,
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
                uint16 & n_comb_sp)
{
    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);

    unsigned int n_cap = capbuf.length();
    unsigned int n_f = f_search_set.length();
    n_comb_xc = (n_cap - 100) / 9600;
    n_comb_sp = (n_cap - 136 - 137) / 9600;

    cufftComplex *h_capbuf = (cufftComplex *)NULL, *d_capbuf = (cufftComplex *)NULL;
    double *h_f = (double *)NULL, *d_f = (double *)NULL;
    float *h_xc_sqr = (float *)NULL, *d_xc_sqr = (float *)NULL;
    float *h_xc_incoherent_single = (float *)NULL, *d_xc_incoherent_single = (float *)NULL;
    float *h_xc_incoherent = (float *)NULL, *d_xc_incoherent = (float *)NULL;
    float *h_xc_incoherent_collapsed_pow = (float *)NULL, *d_xc_incoherent_collapsed_pow = (float *)NULL;
    int *h_xc_incoherent_collapsed_frq = (int *)NULL, *d_xc_incoherent_collapsed_frq = (int *)NULL;
    float *h_sp_incoherent = (float *)NULL, *d_sp_incoherent = (float *)NULL;
 
    h_capbuf = (cufftComplex *)malloc(n_cap * sizeof(cufftComplex));
    h_f = (double *)malloc(n_f * sizeof(double));
    h_xc_incoherent_single = (float *)malloc(3 * n_f * 9600 * sizeof(float));
    h_xc_incoherent = (float *)malloc(3 * n_f * 9600 * sizeof(float));
    h_xc_incoherent_collapsed_pow = (float *)malloc(3 * 9600 * sizeof(float));
    h_xc_incoherent_collapsed_frq = (int *)malloc(3 * 9600 * sizeof(int));
    h_sp_incoherent = (float *)malloc(9600 * sizeof(float));
    h_xc_sqr = (float *)malloc(n_cap * sizeof(float));
 
    checkCudaErrors(cudaMalloc(&d_capbuf, n_cap * sizeof(cufftComplex)));
    checkCudaErrors(cudaMalloc(&d_f, n_f * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_xc_sqr, n_cap * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_xc_incoherent_single, 3 * n_f * 9600 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_xc_incoherent, 3 * n_f * 9600 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_xc_incoherent_collapsed_pow, 3 * 9600 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_xc_incoherent_collapsed_frq, 3 * 9600 * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_sp_incoherent, 9600 * sizeof(float)));

    for (unsigned int i = 0; i < n_cap; i++) {
        h_capbuf[i].x = capbuf[i].real();
        h_capbuf[i].y = capbuf[i].imag();
    }

    for (unsigned int i = 0; i < n_f; i++) {
        h_f[i] = CUDART_PI * 2 * f_search_set[i] * fc_programmed / (fs_programmed * (fc_requested - f_search_set[i]));
    }

    checkCudaErrors(cudaMemcpy(d_capbuf, h_capbuf, n_cap * sizeof(cufftComplex), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_f, h_f, n_f * sizeof(double), cudaMemcpyHostToDevice));

    for (unsigned int foi = 0; foi < n_f; foi++) {
        for (unsigned int t = 0; t < 3; t++) {
            xc_correlate_kernel<<<600, 256>>>(d_capbuf, d_xc_sqr, 
                                              &d_xc_incoherent_single[(foi * 3 + t)*9600], &d_xc_incoherent[(foi * 3 + t)*9600],
                                              n_cap, ds_comb_arm, 
                                              t, f_search_set[foi], (fc_requested - f_search_set[foi]) * fs_programmed /fc_programmed);
            checkCudaErrors(cudaDeviceSynchronize());
        }
    }
    checkCudaErrors(cudaDeviceSynchronize());

    xc_incoherent_collapsed_kernel<<<9600, 3>>>(d_xc_incoherent, d_xc_incoherent_collapsed_pow, d_xc_incoherent_collapsed_frq, n_f);
    checkCudaErrors(cudaDeviceSynchronize());

    sp_incoherent_kernel<<<600, 512>>>(d_capbuf, d_sp_incoherent, n_cap);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(h_xc_incoherent_single, d_xc_incoherent_single, 3 * n_f * 9600 * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_xc_incoherent_collapsed_pow, d_xc_incoherent_collapsed_pow, 3 * 9600 * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_xc_incoherent_collapsed_frq, d_xc_incoherent_collapsed_frq, 3 * 9600 * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_sp_incoherent, d_sp_incoherent, 9600 * sizeof(float), cudaMemcpyDeviceToHost));

    sp_incoherent = vec(9600);
    xc_incoherent_collapsed_pow = mat(3, 9600);
    xc_incoherent_collapsed_frq = imat(3, 9600);
    xc_incoherent_single = vector < vector < vector < float > > > (3,vector< vector < float > >(9600, vector < float > (n_f)));

    for (unsigned int foi = 0; foi < n_f; foi++) {
        for (unsigned int t = 0; t < 3; t++) {
            for (unsigned int k = 0; k < 9600; k++) {
                xc_incoherent_single[t][k][foi] = h_xc_incoherent_single[(foi*3+t)*9600+k];
            }
        }
    }

    for (unsigned int t = 0; t < 3; t++) {
        for (unsigned int k = 0; k < 9600; k++) {
            xc_incoherent_collapsed_pow(t,k) = h_xc_incoherent_collapsed_pow[t * 9600 + k];
            xc_incoherent_collapsed_frq(t,k) = h_xc_incoherent_collapsed_frq[t * 9600 + k];
        }
    }

    for (unsigned int i = 0; i < 9600; i++) {
        sp_incoherent[i] = h_sp_incoherent[i];
    }

    free(h_capbuf);
    free(h_f);
    free(h_xc_incoherent_single);
    free(h_xc_incoherent);
    free(h_xc_incoherent_collapsed_pow);
    free(h_xc_incoherent_collapsed_frq);
    free(h_sp_incoherent);
    free(h_xc_sqr);

    checkCudaErrors(cudaFree(d_capbuf));
    checkCudaErrors(cudaFree(d_f));
    checkCudaErrors(cudaFree(d_xc_sqr));
    checkCudaErrors(cudaFree(d_xc_incoherent_single));
    checkCudaErrors(cudaFree(d_xc_incoherent));
    checkCudaErrors(cudaFree(d_xc_incoherent_collapsed_pow));
    checkCudaErrors(cudaFree(d_xc_incoherent_collapsed_frq));
    checkCudaErrors(cudaFree(d_sp_incoherent));

    gettimeofday(&tv2, NULL);
    printf("xcorr_pss2 : %ld us\n", (tv2.tv_sec-tv1.tv_sec)*1000000+(tv2.tv_usec-tv1.tv_usec));
}

