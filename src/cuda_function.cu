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

#define SIGNAL_SIZE 128
#define SQRT2_INV               (0.7071067817811865475)
#define SQRT62_INV              (0.1270001270001905)
#define SQRT128_INV             (0.0883883476483184)
#define COMPLEX_MUL_REAL(a, b)  ((a).x * (b).x - (a).y * (b).y)
#define COMPLEX_MUL_IMAG(a, b)  ((a).x * (b).y + (a).y * (b).x)

#undef SWAP
#define SWAP(x,y)               do { typeof (x) __tmp = (x); (x) = (y); (y) = (__tmp); } while (0)

__constant__ cufftDoubleComplex pss_fd[3][62];
__constant__ cufftDoubleComplex pss_td[3][256];
__constant__ unsigned int sss_fd[168][3][2][2];

__constant__ cufftDoubleComplex d_tw128[SIGNAL_SIZE];
__constant__ short d_radix2_bitreverse[SIGNAL_SIZE];
__constant__ short d_radix4_bitreverse[SIGNAL_SIZE];

cufftDoubleComplex h_pss_fd[3][62];
cufftDoubleComplex h_pss_td[3][256];
unsigned int h_sss_fd[168][3][2][2];

cufftDoubleComplex h_tw128[SIGNAL_SIZE];
short h_radix2_bitreverse[SIGNAL_SIZE];
short h_radix4_bitreverse[SIGNAL_SIZE];

__device__ void kernel_fft_radix2(cufftDoubleComplex *c_io, int N);

/*
 *  Wrapper of cudaDeviceReset()
 */
extern "C" void cuda_reset_device()
{
    cudaDeviceReset();
}


/*
 *  Copy constant data to CUDA device.
 *
 *  To-Do : use CUDA to generate PSS and SSS.
 */
extern "C" void cuda_copy_constant_data_to_device()
{
    int i, j, k, t, len;

    for (t = 0; t < 3; t++) {
        len = ROM_TABLES.pss_td[t].length();
        for (i = 0; i < len; i++) {
            h_pss_td[t][i].x = ROM_TABLES.pss_td[t][i].real();
            h_pss_td[t][i].y = ROM_TABLES.pss_td[t][i].imag();
        }
        for (; i < 256; i++) {
            h_pss_td[t][i].x = 0.0f;
            h_pss_td[t][i].y = 0.0f;
        }
    }

    // PSS frequency domain data

    for (t = 0; t < 3; t++) {
        for (i = 0; i < 62; i++) {
            h_pss_fd[t][i].x = ROM_TABLES.pss_fd[t][i].real();
            h_pss_fd[t][i].y = ROM_TABLES.pss_fd[t][i].imag();
        }
    }

    for (t = 0; t < 168; t++) {
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 2; j++) {
                unsigned int word1 = 0, word2 = 0;
                for (k = 0; k < 31; k++) {
                    if (ROM_TABLES.sss_fd(t,i,j)[k] == -1)
                        word1 |= (1 << k);
                    if (ROM_TABLES.sss_fd(t,i,j)[k+31] == -1)
                        word2 |= (1 << k);
                }
                h_sss_fd[t][i][j][0] = word1;
                h_sss_fd[t][i][j][1] = word2;
            }
        }
    }

    checkCudaErrors(cudaMemcpyToSymbol(pss_td, h_pss_td, sizeof(h_pss_td)));
    checkCudaErrors(cudaMemcpyToSymbol(pss_fd, h_pss_fd, sizeof(h_pss_fd)));
    checkCudaErrors(cudaMemcpyToSymbol(sss_fd, h_sss_fd, sizeof(h_sss_fd)));
}

/*
 * Return bit-reverse of number n in nbits bits.
 */
extern "C" unsigned int reverse_bit(unsigned int n, int nbits)
{
    unsigned int reverse_num = 0;

    for (int i = 0; i < nbits; i++) {
        if (n & (1 << i))
            reverse_num |= (1 << ((nbits - 1) - i));
    }

    return reverse_num;
}

/*
 * Returns radix-4/2 reverse of number n in nbits bits.
 * s denotes the radix-2 stage is in the last one (1) or the first one (0).
 *
 * Example : s=1, assuming 4442 -> 2444
 *           s=0, assuming 2444 -> 4442
 */
extern "C" unsigned int reverse_radix_4_and_2(unsigned int n, int nbits, int s)
{
    unsigned int reverse_num = 0;
    int i = 0;

    if (nbits & 1) {
        if (s == 0) {
            reverse_num |= ((n >> (nbits - 1)) & 1);
        } else {
            i = 1;
        }
    }

    for (; i <= nbits - 2; i += 2) {
        reverse_num |= (((n >> i) & 3) << ((nbits - 2) - i));
    }

    if (nbits & 1) {
        if (s == 1) {
            reverse_num |= ((n & 1) << (nbits - 1));
        }
    }

    return reverse_num;
}


/*
 * Generate twiddle factor of length N and copy them to CUDA device.
 * Generate bit reverse table of length N and copy them to CUDA device.
 *
 * Now support data of length N less or equal to 128.
 */
extern "C" void generate_twiddle_factor(int N)
{
    int nbits = ceil(log(1.0 * N) / log(2.0));

    for (int n = 0; n < N; n++) {
        double theta = (CUDART_PI * 2 * n) / N;
        h_tw128[n].x = cos(theta);
        h_tw128[n].y = -sin(theta);
        h_radix2_bitreverse[n] = reverse_bit(n, nbits);
        h_radix4_bitreverse[n] = reverse_radix_4_and_2(n, nbits, 1);
    }

    checkCudaErrors(cudaMemcpyToSymbol(d_tw128, &h_tw128, sizeof(h_tw128)));
    checkCudaErrors(cudaMemcpyToSymbol(d_radix2_bitreverse, &h_radix2_bitreverse, sizeof(h_radix2_bitreverse)));
    checkCudaErrors(cudaMemcpyToSymbol(d_radix4_bitreverse, &h_radix4_bitreverse, sizeof(h_radix4_bitreverse)));
}



/*
 * Calculate angle of complex number with real and imag.
 */
__device__ double angle(float real, float imag)
{
    if (real > 0.0) {
        return atan(imag / real);
    } else if (real < 0.0) {
        if (imag >= 0.0) {
            return atan(imag / real) + CUDART_PI;
        } else {
            return atan(imag / real) - CUDART_PI;
        }
    } else if (imag > 0.0) {
        return CUDART_PI / 2;
    } else if (imag < 0.0) {
        return -CUDART_PI / 2;
    } else {
        return CUDART_NAN;
    }
}



/*
 *  Step 1 of xc_correlate()
 */
__global__ void xc_correlate_step1_kernel(cufftDoubleComplex *d_capbuf, double *d_xc_sqr,
                                          const unsigned int n_cap, const unsigned int t,
                                          const double f_off, const double fc_requested, const double fc_programmed, const double fs_programmed)
{
    __shared__ cufftDoubleComplex s_fshift_pss[256], s_capbuf[256 + 137];

    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;

    const double k_factor = (fc_requested - f_off)/fc_programmed;
    const double k = CUDART_PI * f_off / (fs_programmed * k_factor / 2);
    double shift = k * tid;
    double x1 = cos(shift), y1 = sin(shift);
    double x2 = pss_td[t][tid].x, y2 = pss_td[t][tid].y;
    double real, imag;

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

    real = COMPLEX_MUL_REAL(s_fshift_pss[0], s_capbuf[tid]);
    imag = COMPLEX_MUL_IMAG(s_fshift_pss[0], s_capbuf[tid]);
    for (unsigned int i = 1; i < 137; i++) {
        real += COMPLEX_MUL_REAL(s_fshift_pss[i], s_capbuf[tid + i]);
        imag += COMPLEX_MUL_IMAG(s_fshift_pss[i], s_capbuf[tid + i]);
    }
    d_xc_sqr[256 * bid + tid] = (real * real + imag * imag) / (137.0*137.0);
}



/*
 * Step 2 of xc_correlate()
 */
__global__ void xc_correlate_step2_kernel(double *d_xc_sqr, double *d_xc_incoherent_single, const unsigned int n_cap,
                                          const double f_off, const double fc_requested, const double fc_programmed, const double fs_programmed)
{
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const double k_factor = (fc_requested - f_off)/fc_programmed;
    const unsigned int index = 16 * bid + tid;
    const unsigned int max_m = (n_cap - 100 - 136) / 9600;
    double xc_incoherent_single_val;

    xc_incoherent_single_val = d_xc_sqr[index];
    for (unsigned int m = 1; m < max_m; m++) {
        unsigned int span = lround(m * 0.005 * k_factor * fs_programmed);
        xc_incoherent_single_val += d_xc_sqr[index + span];
    }
    d_xc_incoherent_single[index] = xc_incoherent_single_val / max_m;
}



/*
 *  Step 3 of xc_correlate()
 */
__global__ void xc_correlate_step3_kernel(double *d_xc_incoherent_single, double *d_xc_incoherent, uint8 ds_comb_arm)
{
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int index = 16 * bid + tid;
    double xc_incoherent_value = 0.0;

    if (index < ds_comb_arm) {
       for (unsigned int i = 0; i <= index + ds_comb_arm; i++) {
           xc_incoherent_value += d_xc_incoherent_single[i];
       }
       for (unsigned int i = 9600 + index - ds_comb_arm; i < 9600; i++) {
           xc_incoherent_value += d_xc_incoherent_single[i];
       }
    } else if (index < 9600 - ds_comb_arm) {
       for (unsigned int i = index - ds_comb_arm; i <= index + ds_comb_arm; i++) {
           xc_incoherent_value += d_xc_incoherent_single[i];
       }
    } else {
       for (unsigned int i = index - ds_comb_arm; i < 9600; i++) {
           xc_incoherent_value += d_xc_incoherent_single[i];
       }
       for (unsigned int i = 0; i <= index + ds_comb_arm - 9600; i++) {
           xc_incoherent_value += d_xc_incoherent_single[i];
       }
    }

    d_xc_incoherent[index] = xc_incoherent_value / (ds_comb_arm * 2 + 1);
}



/*
 *
 */
__global__ void xc_incoherent_collapsed_kernel(double *d_xc_incoherent,
                                               double *d_xc_incoherent_collapsed_pow, int *d_xc_incoherent_collapsed_frq,
                                               unsigned int n_f)
{
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    double best_pow = d_xc_incoherent[(0 * 3 + tid) * 9600 + bid];
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



/*
 *
 */
__global__ void sp_incoherent_kernel(cufftDoubleComplex *d_capbuf, double *d_sp_incoherent, double *d_Z_th1, unsigned int n_cap, double Z_th1_factor)
{
    __shared__ double s_sqr[512];

    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int n_comb_sp = (n_cap - 136 - 137) / 9600;
    unsigned int index = bid * 16 + tid;
    double value;

    if (tid < 274 + 16) {
        value = d_capbuf[index].x * d_capbuf[index].x + d_capbuf[index].y * d_capbuf[index].y;
        for (unsigned int m = 1; m < n_comb_sp; m++) {
            value += (d_capbuf[index + 9600 * m].x * d_capbuf[index + 9600 * m].x + d_capbuf[index + 9600 * m].y * d_capbuf[index + 9600 * m].y);
        }
        s_sqr[tid] = value;
    } else {
        s_sqr[tid] = 0.0;
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
        d_Z_th1[index] = d_sp_incoherent[index] * Z_th1_factor;
    }
}



/*
 *
 */
__global__ void peak_search_kernel(double *d_xc_incoherent_collapsed_pow, int *d_xc_incoherent_collapsed_frq, double *d_f_search_set,
                                   double *d_Z_th1, double *d_xc_incoherent_single, short *d_aux, int ds_comb_arm)
{
    __shared__ unsigned int finished;
    __shared__ double thresh1, thresh2, peak_pow;
    __shared__ short peak_pos, peak_ind, peak_n_id_2;

    const int tid = threadIdx.x;
    short pos, pos_ind, pos_n_id_2;
    double pos_pow;

    for (unsigned int i = tid; i < 9600 * 3; i += 1024) {
        d_aux[i] = i;
    }

    __syncthreads();

    do {
        for (unsigned int k = 0; k < 9600 * 3; k += 2048) {
            for (unsigned int s = 1024; s > 0; s >>= 1) {
                if ((tid < s) && (k + tid + s < 9600 * 3)) {
                    int pos1 = d_aux[k + tid];
                    int pos2 = d_aux[k + tid + s];

                    if (d_xc_incoherent_collapsed_pow[pos1] < d_xc_incoherent_collapsed_pow[pos2]) {
                        d_aux[k + tid] = pos2;
                        d_aux[k + tid + s] = pos1;
                    }
                }
            }
            __syncthreads();
        }

        for (unsigned int s = 8; s > 0; s >>= 1) {
            if ((tid < s) && ((tid + s) * 2048 < 9600 * 3)) {
                int pos1 = d_aux[tid * 2048];
                int pos2 = d_aux[(tid + s) * 2048];

                if (d_xc_incoherent_collapsed_pow[pos1] < d_xc_incoherent_collapsed_pow[pos2]) {
                    d_aux[tid * 2048] = pos2;
                    d_aux[(tid + s) * 2048] = pos1;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            peak_pos = d_aux[0];
            peak_pow = d_xc_incoherent_collapsed_pow[peak_pos];

            if (peak_pow < d_Z_th1[peak_ind]) {
                finished = 1;
            } else {

                finished = 0;
                peak_n_id_2 = peak_pos / 9600;
                peak_ind = peak_pos - peak_n_id_2 * 9600;

                int freq_idx = d_xc_incoherent_collapsed_frq[peak_pos];
                double freq = d_f_search_set[freq_idx];
                double best_pow = -CUDART_INF;
                short best_ind = -1;
                int t = peak_ind - ds_comb_arm;

                if (t < 0) t += 9600;

                for (unsigned int i = 0; i < 2 * ds_comb_arm + 1; i++) {
                    if (d_xc_incoherent_single[(freq_idx * 3 + peak_n_id_2) * 9600 + t] > best_pow) {
                        best_ind = t;
                        best_pow = d_xc_incoherent_single[(freq_idx * 3 + peak_n_id_2) * 9600 + t];
                    }
                    t++;
                    if (t >= 9600) t-= 9600;
                }

                thresh1 = peak_pow * 0.1584893192461113; // udb10(-8.0) = pow(10.0,-8.0/10.0);
                thresh2 = peak_pow * 0.0630957344480193; // udb10(-12.0) = pow(10.0,-12.0/10.0);
            }
        }

        __syncthreads();

        if (!finished) {

            for (unsigned int i = tid; i < 3 * 9600; i += 1024) {

                pos = d_aux[i];

                pos_n_id_2 = pos / 9600;
                pos_ind = pos - pos_n_id_2 * 9600;
                pos_pow = d_xc_incoherent_collapsed_pow[pos];

                if (pos_pow < thresh2) {
                    d_xc_incoherent_collapsed_pow[pos] = 0.0;
                }

                /* 9600 - 274 <= pos_ind - peak_ind + 9600 <= 9600 + 274 */

                if (((9600 - 274) <= (pos_ind - peak_ind + 9600)) && 
                    ((pos_ind - peak_ind + 9600) <= (9600 + 274))) {

                    if (peak_n_id_2 == pos_n_id_2) {
                       d_xc_incoherent_collapsed_pow[pos] = 0.0;
                    } else if (pos_pow < thresh1) {
                       d_xc_incoherent_collapsed_pow[pos] = 0.0;
                    }
                }
            }
        }

        __syncthreads();

    } while (!finished);
}



/*
 *
 */
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

    cufftDoubleComplex *h_capbuf = (cufftDoubleComplex *)NULL, *d_capbuf = (cufftDoubleComplex *)NULL;
    double *h_f = (double *)NULL, *d_f = (double *)NULL;
    double *h_f_search_set = (double *)NULL, *d_f_search_set = (double *)NULL;
    double *h_xc_sqr = (double *)NULL, *d_xc_sqr = (double *)NULL;
    double *h_xc_incoherent_single = (double *)NULL, *d_xc_incoherent_single = (double *)NULL;
    double *h_xc_incoherent = (double *)NULL, *d_xc_incoherent = (double *)NULL;
    double *h_xc_incoherent_collapsed_pow = (double *)NULL, *d_xc_incoherent_collapsed_pow = (double *)NULL;
    int *h_xc_incoherent_collapsed_frq = (int *)NULL, *d_xc_incoherent_collapsed_frq = (int *)NULL;
    double *h_sp_incoherent = (double *)NULL, *d_sp_incoherent = (double *)NULL;
    double *h_Z_th1 = (double *)NULL, *d_Z_th1 = (double *)NULL;
    short *d_aux = (short *)NULL;

    h_capbuf = (cufftDoubleComplex *)malloc(n_cap * sizeof(cufftDoubleComplex));
    h_f = (double *)malloc(n_f * sizeof(double));
    h_f_search_set = (double *)malloc(n_f * sizeof(double));
    h_xc_incoherent_single = (double *)malloc(3 * n_f * 9600 * sizeof(double));
    h_xc_incoherent = (double *)malloc(3 * n_f * 9600 * sizeof(double));
    h_xc_incoherent_collapsed_pow = (double *)malloc(3 * 9600 * sizeof(double));
    h_xc_incoherent_collapsed_frq = (int *)malloc(3 * 9600 * sizeof(int));
    h_sp_incoherent = (double *)malloc(9600 * sizeof(double));
    h_Z_th1 = (double *)malloc(9600 * sizeof(double));
    h_xc_sqr = (double *)malloc(n_cap * sizeof(double));

    checkCudaErrors(cudaMalloc(&d_capbuf, n_cap * sizeof(cufftDoubleComplex)));
    checkCudaErrors(cudaMalloc(&d_f, n_f * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_f_search_set, n_f * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_xc_sqr, n_cap * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_xc_incoherent_single, 3 * n_f * 9600 * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_xc_incoherent, 3 * n_f * 9600 * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_xc_incoherent_collapsed_pow, 3 * 9600 * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_xc_incoherent_collapsed_frq, 3 * 9600 * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_sp_incoherent, 9600 * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_Z_th1, 9600 * sizeof(double)));

    checkCudaErrors(cudaMalloc(&d_aux, 9600 * 3 * sizeof(short)));

    for (unsigned int i = 0; i < n_cap; i++) {
        h_capbuf[i].x = capbuf[i].real();
        h_capbuf[i].y = capbuf[i].imag();
    }

    checkCudaErrors(cudaMemcpy(d_capbuf, h_capbuf, n_cap * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice));

    for (unsigned int i = 0; i < n_f; i++) {
        h_f[i] = CUDART_PI * 2 * f_search_set[i] * fc_programmed / (fs_programmed * (fc_requested - f_search_set[i]));
        h_f_search_set[i] = f_search_set[i];
    }

    checkCudaErrors(cudaMemcpy(d_f, h_f, n_f * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_f_search_set, h_f_search_set, n_f * sizeof(double), cudaMemcpyHostToDevice));

    /* xc_correlate, xc_combine, xc_delay_spread */
    for (unsigned int foi = 0; foi < n_f; foi++) {
        for (unsigned int t = 0; t < 3; t++) {

            xc_correlate_step1_kernel<<<600, 256>>>(d_capbuf, d_xc_sqr, n_cap, t,
                                                    f_search_set[foi], fc_requested, fc_programmed, fs_programmed);
            checkCudaErrors(cudaDeviceSynchronize());

            xc_correlate_step2_kernel<<<600, 16>>>(d_xc_sqr, &d_xc_incoherent_single[(foi * 3 + t)*9600], n_cap,
                                                   f_search_set[foi], fc_requested, fc_programmed, fs_programmed);
            checkCudaErrors(cudaDeviceSynchronize());

            xc_correlate_step3_kernel<<<600, 16>>>(&d_xc_incoherent_single[(foi * 3 + t)*9600], &d_xc_incoherent[(foi * 3 + t)*9600],
                                                   ds_comb_arm);
            checkCudaErrors(cudaDeviceSynchronize());
        }
    }
    checkCudaErrors(cudaDeviceSynchronize());

    /* xc_peak_freq */
    xc_incoherent_collapsed_kernel<<<9600, 3>>>(d_xc_incoherent, d_xc_incoherent_collapsed_pow, d_xc_incoherent_collapsed_frq, n_f);
    checkCudaErrors(cudaDeviceSynchronize());

    /* sp_est, Z_th1 */
    const uint8 thresh1_n_nines = 12;
    double R_th1 = chi2cdf_inv(1 - pow(10.0, -thresh1_n_nines), 2 * n_comb_xc * (2 * ds_comb_arm + 1));
    double rx_cutoff = (6 * 12 * 15e3 / 2 + 4*15e3) / (FS_LTE / 16 / 2);
    double Z_th1_factor = R_th1 / rx_cutoff / 137 / 2 / n_comb_xc / (2 * ds_comb_arm + 1);

    sp_incoherent_kernel<<<600, 512>>>(d_capbuf, d_sp_incoherent, d_Z_th1, n_cap, Z_th1_factor);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(h_xc_incoherent_single, d_xc_incoherent_single, 3 * n_f * 9600 * sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_xc_incoherent_collapsed_pow, d_xc_incoherent_collapsed_pow, 3 * 9600 * sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_xc_incoherent_collapsed_frq, d_xc_incoherent_collapsed_frq, 3 * 9600 * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_sp_incoherent, d_sp_incoherent, 9600 * sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_Z_th1, d_Z_th1, 9600 * sizeof(double), cudaMemcpyDeviceToHost));

    peak_search_kernel<<<1, 1024>>>(d_xc_incoherent_collapsed_pow, d_xc_incoherent_collapsed_frq, d_f_search_set,
                                    d_Z_th1, d_xc_incoherent_single, d_aux, ds_comb_arm);
    checkCudaErrors(cudaDeviceSynchronize());

    /* copy data for subsequent functions */
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
    free(h_f_search_set);
    free(h_xc_incoherent_single);
    free(h_xc_incoherent);
    free(h_xc_incoherent_collapsed_pow);
    free(h_xc_incoherent_collapsed_frq);
    free(h_sp_incoherent);
    free(h_xc_sqr);

    checkCudaErrors(cudaFree(d_capbuf));
    checkCudaErrors(cudaFree(d_f));
    checkCudaErrors(cudaFree(d_f_search_set));
    checkCudaErrors(cudaFree(d_xc_sqr));
    checkCudaErrors(cudaFree(d_xc_incoherent_single));
    checkCudaErrors(cudaFree(d_xc_incoherent));
    checkCudaErrors(cudaFree(d_xc_incoherent_collapsed_pow));
    checkCudaErrors(cudaFree(d_xc_incoherent_collapsed_frq));
    checkCudaErrors(cudaFree(d_sp_incoherent));

    checkCudaErrors(cudaFree(d_aux));
}



/*
 *  Step 1 of sss_detect_getce_sss()
 */
__global__ void sss_detect_getce_sss_multiblocks_step1_kernel(cufftDoubleComplex *d_capbuf, int n_pss,
                                                              unsigned short n_id_2_est, double peak_loc,
                                                              double fc_requested, double fc_programmed, double fs_programmed, double peak_freq,
                                                              // output
                                                              cufftDoubleComplex *d_h_sm, double *d_pss_np, cufftDoubleComplex *d_sss_nrm_raw, cufftDoubleComplex *d_sss_ext_raw)
{
    __shared__ cufftDoubleComplex s_pss_dft[128], s_nrm_sss_dft[128], s_ext_sss_dft[128];
    __shared__ cufftDoubleComplex h_raw[62], *p_pss_fd;
    __shared__ float pss_np;

    const unsigned int bid = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int output_offset = bid * 62;
    const double k_factor = (fc_requested - peak_freq) / fc_programmed;
    const unsigned int pss_dft_location = lround(peak_loc + bid * k_factor * 9600 + 9 - 2);
    const unsigned int nrm_sss_dft_location = pss_dft_location - 128 - 9;
    const unsigned int ext_sss_dft_location = pss_dft_location - 128 - 32;
    cufftDoubleComplex shift, acc;
    double noise_r, noise_i;

    double k = CUDART_PI * (-peak_freq) / ((fs_programmed * k_factor) / 2);

    if (tid == 0) {
        p_pss_fd = (cufftDoubleComplex *)&pss_fd[n_id_2_est];
        pss_np = 0.0;
    }

    shift.x = cos(k * tid);
    shift.y = sin(k * tid);

    // implement extract_psss(capbuf.mid(pss_dft_location, 128), -cell_in.freq, k_factor, fs_programmed)
    if (tid < 2) {
        s_pss_dft[126 + tid].x = COMPLEX_MUL_REAL(d_capbuf[pss_dft_location + tid], shift);
        s_pss_dft[126 + tid].y = COMPLEX_MUL_IMAG(d_capbuf[pss_dft_location + tid], shift);
        s_nrm_sss_dft[126 + tid].x = COMPLEX_MUL_REAL(d_capbuf[nrm_sss_dft_location + tid], shift);
        s_nrm_sss_dft[126 + tid].y = COMPLEX_MUL_IMAG(d_capbuf[nrm_sss_dft_location + tid], shift);
        s_ext_sss_dft[126 + tid].x = COMPLEX_MUL_REAL(d_capbuf[ext_sss_dft_location + tid], shift);
        s_ext_sss_dft[126 + tid].y = COMPLEX_MUL_IMAG(d_capbuf[ext_sss_dft_location + tid], shift);
    } else if (tid < 128) {
        s_pss_dft[tid - 2].x = COMPLEX_MUL_REAL(d_capbuf[pss_dft_location + tid], shift);
        s_pss_dft[tid - 2].y = COMPLEX_MUL_IMAG(d_capbuf[pss_dft_location + tid], shift);
        s_nrm_sss_dft[tid - 2].x = COMPLEX_MUL_REAL(d_capbuf[nrm_sss_dft_location + tid], shift);
        s_nrm_sss_dft[tid - 2].y = COMPLEX_MUL_IMAG(d_capbuf[nrm_sss_dft_location + tid], shift);
        s_ext_sss_dft[tid - 2].x = COMPLEX_MUL_REAL(d_capbuf[ext_sss_dft_location + tid], shift);
        s_ext_sss_dft[tid - 2].y = COMPLEX_MUL_IMAG(d_capbuf[ext_sss_dft_location + tid], shift);
    }

    __syncthreads();

    if (tid == 0) {
        kernel_fft_radix2(s_pss_dft, 128);
    } else if (tid == 32) {
        kernel_fft_radix2(s_nrm_sss_dft, 128);
    } else if (tid == 64) {
        kernel_fft_radix2(s_ext_sss_dft, 128);
    }

    __syncthreads();

    // index   :  0,  1,       30, 31, 32, 33,    , 61
    // dft_out : 97, 98, .... 127,  1,  2,  3, ..., 31
    // pss_fd  :  0,  1, ....  30, 31, 32, 33, ..., 61

    if (tid < 31) {
        h_raw[tid].x = SQRT128_INV * (s_pss_dft[tid + 97].x * p_pss_fd[tid].x + s_pss_dft[tid + 97].y * p_pss_fd[tid].y);
        h_raw[tid].y = SQRT128_INV * (s_pss_dft[tid + 97].y * p_pss_fd[tid].x - s_pss_dft[tid + 97].x * p_pss_fd[tid].y);
    } else if (tid < 62) {
        // concat(dft_out.right(31), dft_out.mid(1,31)) * conj(pss_fd)
        h_raw[tid].x = SQRT128_INV * (s_pss_dft[tid - 30].x * p_pss_fd[tid].x + s_pss_dft[tid - 30].y * p_pss_fd[tid].y);
        h_raw[tid].y = SQRT128_INV * (s_pss_dft[tid - 30].y * p_pss_fd[tid].x - s_pss_dft[tid - 30].x * p_pss_fd[tid].y);
    }

    __syncthreads();

    // Smoothing... Basic...

    //                  t  = 0, 1, 2, 3,  4,  5,  6,  7,  8,  9, 10,        , 53, 54, 55, 56, 57, 58, 59, 60, 61
    // lt  = MAX(0, t - 6) = 0, 0, 0, 0,  0,  0,  0,  1,  2,  3,  4, ...... , 47, 48, 49, 50, 51, 52, 53, 54, 55
    // rt  = MIN(61,t + 6) = 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, ...... , 59, 60, 61, 61, 61, 61, 61, 61, 61
    // len = rt-lt + 1     = 7, 8, 9,10, 11, 12, 13, 13, 13, 13, 13, ...... , 13, 13, 13, 12, 11, 10,  9,  8,  7

    acc.x = 0.0; acc.y = 0.0;
    if (tid < 6) {
        for (unsigned int i = 0; i <= tid + 6; i++) {
            acc.x += h_raw[i].x;
            acc.y += h_raw[i].y;
        }
        acc.x /= (tid + 7);
        acc.y /= (tid + 7);
    } else if (tid < 56) {
        for (unsigned int i = tid - 6; i <= tid + 6; i++) {
            acc.x += h_raw[i].x;
            acc.y += h_raw[i].y;
        }
        acc.x /= 13;
        acc.y /= 13;
    } else if (tid < 62) {
        for (unsigned int i = tid - 6; i <= 61; i++) {
            acc.x += h_raw[i].x;
            acc.y += h_raw[i].y;
        }
        acc.x /= (68 - tid);
        acc.y /= (68 - tid);
    }

    // Estimate noise power.
    // pss_np = sigpower(h_sm.get_row(tid) - h_raw_fo_pss.get_row(tid));

    if (tid < 62) {
        d_h_sm[output_offset + tid].x = acc.x;
        d_h_sm[output_offset + tid].y = acc.y;

        noise_r = acc.x - h_raw[tid].x;
        noise_i = acc.y - h_raw[tid].y;
        atomicAdd(&pss_np, (float)(noise_r * noise_r + noise_i * noise_i));
    }

    __syncthreads();

    if (tid == 0) {
        d_pss_np[bid] = pss_np / 62;
    }

    __syncthreads();

    // implment : extract_psss(capbuf.mid(ext_sss_dft_location,128),-peak_freq,k_factor,fs_programmed)

    //  index   :  0,  1,       30, 31, 32, 33,    , 61
    //  dft_out : 97, 98, .... 127,  1,  2,  3, ..., 31

    if (tid < 31) {
        d_sss_nrm_raw[output_offset + tid].x = SQRT128_INV * s_nrm_sss_dft[tid + 97].x;
        d_sss_nrm_raw[output_offset + tid].y = SQRT128_INV * s_nrm_sss_dft[tid + 97].y; 
        d_sss_ext_raw[output_offset + tid].x = SQRT128_INV * s_ext_sss_dft[tid + 97].x;
        d_sss_ext_raw[output_offset + tid].y = SQRT128_INV * s_ext_sss_dft[tid + 97].y; 
    } else if (tid < 62) {
        d_sss_nrm_raw[output_offset + tid].x = SQRT128_INV * s_nrm_sss_dft[tid - 30].x;
        d_sss_nrm_raw[output_offset + tid].y = SQRT128_INV * s_nrm_sss_dft[tid - 30].y;
        d_sss_ext_raw[output_offset + tid].x = SQRT128_INV * s_ext_sss_dft[tid - 30].x;
        d_sss_ext_raw[output_offset + tid].y = SQRT128_INV * s_ext_sss_dft[tid - 30].y;
    }
}



/*
 *  Step 2 of sss_detect_getce_sss()
 */
__global__ void sss_detect_getce_sss_multiblocks_step2_kernel(int n_pss,
                                                              cufftDoubleComplex *d_h_sm, double *d_pss_np, cufftDoubleComplex *d_sss_nrm_raw, cufftDoubleComplex *d_sss_ext_raw,
                                                              // output
                                                              float *d_sss_h12_np_est, cufftDoubleComplex *d_sss_h12_nrm_est, cufftDoubleComplex *d_sss_h12_ext_est)
{
    __shared__ cufftDoubleComplex h_sm[50], sss_nrm_raw[50], sss_ext_raw[50];
    __shared__ double pss_np_inv[50];
    __shared__ double sss_np_est[2];
    __shared__ float sss_h12_nrm_est_real[2], sss_h12_nrm_est_imag[2];
    __shared__ float sss_h12_ext_est_real[2], sss_h12_ext_est_imag[2];
    __shared__ float sum_of_sqr_sm_pss_np_inv[2];

    const unsigned int bid = blockIdx.x;
    const unsigned int tid = threadIdx.x;

    if (tid > 50) {
        printf("Caution : number of thread is larger than reserved space !\n");
        return;
    }

    pss_np_inv[tid] = 1.0 / d_pss_np[tid];
    h_sm[tid] = d_h_sm[tid * 62 + bid];
    sss_nrm_raw[tid] = d_sss_nrm_raw[tid * 62 + bid];
    sss_ext_raw[tid] = d_sss_ext_raw[tid * 62 + bid];

    if (tid < 2) {
        sss_h12_nrm_est_real[tid] = 0.0;
        sss_h12_nrm_est_imag[tid] = 0.0;
        sss_h12_ext_est_real[tid] = 0.0;
        sss_h12_ext_est_imag[tid] = 0.0;
        sum_of_sqr_sm_pss_np_inv[tid] = 0.0;
    }

    __syncthreads();

    //  tid = 0 ... (n_pss-1)

    //  vec pss_np_inv_h1 = 1.0 / pss_np(itpp_ext::matlab_range(0, 2, n_pss - 1));
    //  vec pss_np_inv_h2 = 1.0 / pss_np(itpp_ext::matlab_range(1, 2, n_pss - 1));
    //
    //  for (uint8 t = 0; t < 62; t++) {
    //
    //      // First half (h1) and second half (h2) channel estimates.
    // 
    //      cvec h_sm_h1 = h_sm.get_col(t).get(itpp_ext::matlab_range(0, 2, n_pss - 1));
    //      cvec h_sm_h2 = h_sm.get_col(t).get(itpp_ext::matlab_range(1, 2, n_pss - 1));
    //
    //      sum(elem_mult(sqr(h_sm_h1), pss_np_inv_h1))
    //      sum(elem_mult(sqr(h_sm_h2), pss_np_inv_h2))
    //
    //      ....
    //  }

    double sqr_sm_pss_np_inv = (h_sm[tid].x * h_sm[tid].x + h_sm[tid].y * h_sm[tid].y) * pss_np_inv[tid];

    atomicAdd(&sum_of_sqr_sm_pss_np_inv[tid & 1], (float)(sqr_sm_pss_np_inv));

    __syncthreads();

    //  Implement the following expressions :
    //
    //  sss_h1_np_est(t) = 1 / (1 + sum(elem_mult(sqr(h_sm_h1), pss_np_inv_h1)));
    //  sss_h2_np_est(t) = 1 / (1 + sum(elem_mult(sqr(h_sm_h2), pss_np_inv_h2)));

    if (tid < 2) {
        d_sss_h12_np_est[bid + tid * 62] = sss_np_est[tid] = 1.0 / (1 + sum_of_sqr_sm_pss_np_inv[tid]);
    }

    __syncthreads();

    //  index   :  0,  1,       30, 31, 32, 33,    , 61
    //  dft_out : 97, 98, .... 127,  1,  2,  3, ..., 31
    //
    //  vec pss_np_inv_h1 = 1.0 / pss_np(itpp_ext::matlab_range(0,2,n_pss-1));
    //  vec pss_np_inv_h2 = 1.0 / pss_np(itpp_ext::matlab_range(1,2,n_pss-1));
    //
    //  for (uint8 t = 0; t < 62; t++) {
    //
    //      sss_h1_np_est(t) = 1 / (1 + sum(elem_mult(sqr(h_sm_h1), pss_np_inv_h1)));
    //      sss_h2_np_est(t) = 1 / (1 + sum(elem_mult(sqr(h_sm_h2), pss_np_inv_h2)));
    //
    //      ...
    //      sss_h1_ext_est(t) = sss_h1_np_est(t) * sum(elem_mult(conj(h_sm_h1), to_cvec(pss_np_inv_h1), sss_ext_raw.get_col(t).get(itpp_ext::matlab_range(0,2,n_pss-1))));
    //      sss_h2_ext_est(t) = sss_h2_np_est(t) * sum(elem_mult(conj(h_sm_h2), to_cvec(pss_np_inv_h2), sss_ext_raw.get_col(t).get(itpp_ext::matlab_range(1,2,n_pss-1))));
    //  }

    double nrm_real, nrm_imag, ext_real, ext_imag;

    nrm_real = nrm_imag = ext_real = ext_imag = sss_np_est[tid & 1] * pss_np_inv[tid];

    nrm_real *= (h_sm[tid].x * sss_nrm_raw[tid].x + h_sm[tid].y * sss_nrm_raw[tid].y);
    nrm_imag *= (h_sm[tid].x * sss_nrm_raw[tid].y - h_sm[tid].y * sss_nrm_raw[tid].x);
    ext_real *= (h_sm[tid].x * sss_ext_raw[tid].x + h_sm[tid].y * sss_ext_raw[tid].y);
    ext_imag *= (h_sm[tid].x * sss_ext_raw[tid].y - h_sm[tid].y * sss_ext_raw[tid].x);

    atomicAdd(&sss_h12_nrm_est_real[tid & 1], (float)nrm_real);
    atomicAdd(&sss_h12_nrm_est_imag[tid & 1], (float)nrm_imag);
    atomicAdd(&sss_h12_ext_est_real[tid & 1], (float)ext_real);
    atomicAdd(&sss_h12_ext_est_imag[tid & 1], (float)ext_imag);

    __syncthreads(); 

    if (tid < 2) {
        d_sss_h12_nrm_est[bid + tid * 62].x = sss_h12_nrm_est_real[tid];
        d_sss_h12_nrm_est[bid + tid * 62].y = sss_h12_nrm_est_imag[tid];
 
        d_sss_h12_ext_est[bid + tid * 62].x = sss_h12_ext_est_real[tid];
        d_sss_h12_ext_est[bid + tid * 62].y = sss_h12_ext_est_imag[tid];
    }
}



/*
 *  Use one thread-block to implement function of sss_detect_getce_sss()
 */
__global__ void sss_detect_getce_sss_singleblock_kernel(cufftDoubleComplex *d_capbuf, int n_pss,
                                                        unsigned short n_id_2_est, int n_symb_dl, double peak_loc,
                                                        double fc_requested, double fc_programmed, double fs_programmed, double peak_freq,
                                                        // output
                                                        float *d_sss_h12_np_est, cufftDoubleComplex *d_sss_h12_nrm_est, cufftDoubleComplex *d_sss_h12_ext_est)
{
    __shared__ cufftDoubleComplex shift[128], *p_pss_fd;
    __shared__ cufftComplex sss_h12_nrm_est[62 * 2], sss_h12_ext_est[62 * 2];
    __shared__ float sss_h12_np_est[62 * 2];

    const unsigned int tid = threadIdx.x;
    const double k_factor = (fc_requested - peak_freq) / fc_programmed;
    const unsigned int pss_dft_location = lround(peak_loc + tid * k_factor * 9600 + 9 - 2);
    cufftDoubleComplex s_capbuf[128], h_raw[62], h_sm[62], acc;
    double pss_np;
    float real, imag;

    double k = CUDART_PI * (-peak_freq) / ((fs_programmed * k_factor) / 2);

    if (tid == 0) {
        p_pss_fd = (cufftDoubleComplex *)&pss_fd[n_id_2_est];
    }

    for (unsigned int i = tid; i < 128; i += n_pss) {
        shift[i].x = cos(k * i);
        shift[i].y = sin(k * i);
    }

    for (unsigned int i = tid; i < 62 * 2; i += n_pss) {
        sss_h12_np_est[i] = 0.0;
        sss_h12_nrm_est[i].x = 0.0; sss_h12_nrm_est[i].y = 0.0;
        sss_h12_ext_est[i].x = 0.0; sss_h12_ext_est[i].y = 0.0;
    }

    __syncthreads();

    // implement extract_psss(capbuf.mid(pss_dft_location, 128), -cell_in.freq, k_factor, fs_programmed)
    for (unsigned int t = pss_dft_location + 2, i = 2; i < 128; i++, t++) {
        /* 128 points data shift, so k multiply from 0 to 125 */

        s_capbuf[i - 2].x = COMPLEX_MUL_REAL(d_capbuf[t], shift[i]);
        s_capbuf[i - 2].y = COMPLEX_MUL_IMAG(d_capbuf[t], shift[i]);
    }

    // time shift 2 points
    s_capbuf[126].x = COMPLEX_MUL_REAL(d_capbuf[pss_dft_location], shift[0]);
    s_capbuf[126].y = COMPLEX_MUL_IMAG(d_capbuf[pss_dft_location], shift[0]);
    s_capbuf[127].x = COMPLEX_MUL_REAL(d_capbuf[pss_dft_location + 1], shift[1]);
    s_capbuf[127].y = COMPLEX_MUL_IMAG(d_capbuf[pss_dft_location + 1], shift[1]);

    kernel_fft_radix2(s_capbuf, 128);

    // index   :  0,  1,       30, 31, 32, 33,    , 61
    // dft_out : 97, 98, .... 127,  1,  2,  3, ..., 31
    // pss_fd  :  0,  1, ....  30, 31, 32, 33, ..., 61

    for (unsigned int i = 1; i <= 31; i++) {
        // concat(dft_out.right(31), dft_out.mid(1,31)) * conj(pss_fd)

        h_raw[i + 30].x = SQRT128_INV * (s_capbuf[i].x * p_pss_fd[i + 30].x + s_capbuf[i].y * p_pss_fd[i + 30].y);
        h_raw[i + 30].y = SQRT128_INV * (s_capbuf[i].y * p_pss_fd[i + 30].x - s_capbuf[i].x * p_pss_fd[i + 30].y);

        h_raw[i - 1].x = SQRT128_INV * (s_capbuf[i + 96].x * p_pss_fd[i - 1].x + s_capbuf[i + 96].y * p_pss_fd[i - 1].y);
        h_raw[i - 1].y = SQRT128_INV * (s_capbuf[i + 96].y * p_pss_fd[i - 1].x - s_capbuf[i + 96].x * p_pss_fd[i - 1].y);
    }

    // Smoothing... Basic...

    //                  t  = 0, 1, 2, 3,  4,  5,  6,  7,  8,  9, 10,        , 53, 54, 55, 56, 57, 58, 59, 60, 61
    // lt  = MAX(0, t - 6) = 0, 0, 0, 0,  0,  0,  0,  1,  2,  3,  4, ...... , 47, 48, 49, 50, 51, 52, 53, 54, 55
    // rt  = MIN(61,t + 6) = 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, ...... , 59, 60, 61, 61, 61, 61, 61, 61, 61
    // len = rt-lt + 1     = 7, 8, 9,10, 11, 12, 13, 13, 13, 13, 13, ...... , 13, 13, 13, 12, 11, 10,  9,  8,  7

    /* To be optimized in future */

    acc.x = 0.0; acc.y = 0.0;
    for (unsigned int i = 0; i < 6; i++) {
        acc.x += h_raw[i].x;
        acc.y += h_raw[i].y;
    }

    for (unsigned int i = 0; i <= 6; i++) {
        acc.x += h_raw[i + 6].x;
        acc.y += h_raw[i + 6].y;

        h_sm[i].x = acc.x / (i + 7);
        h_sm[i].y = acc.y / (i + 7);
    }

    for (unsigned int i = 7; i <= 55; i++) {
        acc.x += (h_raw[i + 6].x - h_raw[i - 7].x);
        acc.y += (h_raw[i + 6].y - h_raw[i - 7].y);

        h_sm[i].x = acc.x / 13;
        h_sm[i].y = acc.y / 13;
    }

    for (unsigned int i = 56; i < 62; i++) {
        acc.x -= h_raw[i - 7].x;
        acc.y -= h_raw[i - 7].y;

        h_sm[i].x = acc.x / (61 + 7 - i);
        h_sm[i].y = acc.y / (61 + 7 - i);
    }

    // Estimate noise power.
    // pss_np = sigpower(h_sm.get_row(tid) - h_raw_fo_pss.get_row(tid));
    pss_np = 0.0;
    for (unsigned int i = 0; i < 62; i++) {
        double noise_r = h_sm[i].x - h_raw[i].x;
        double noise_i = h_sm[i].y - h_raw[i].y;
        pss_np += (noise_r * noise_r + noise_i * noise_i);
    }

    pss_np = pss_np / 62.0;

    __syncthreads();

    //  vec pss_np_inv_h1 = 1.0 / pss_np(itpp_ext::matlab_range(0, 2, n_pss - 1));
    //  vec pss_np_inv_h2 = 1.0 / pss_np(itpp_ext::matlab_range(1, 2, n_pss - 1));
    //
    //  for (uint8 t = 0; t < 62; t++) {
    //
    //      // First half (h1) and second half (h2) channel estimates.
    // 
    //      cvec h_sm_h1 = h_sm.get_col(t).get(itpp_ext::matlab_range(0, 2, n_pss - 1));
    //      cvec h_sm_h2 = h_sm.get_col(t).get(itpp_ext::matlab_range(1, 2, n_pss - 1));
    //
    //      sum(elem_mult(sqr(h_sm_h1), pss_np_inv_h1))
    //      sum(elem_mult(sqr(h_sm_h2), pss_np_inv_h2))
    //
    //      ....
    //  }

    for (unsigned int i = 0, col = (tid & 1) * 62; i < 62; i++, col++) {
         float sss_np_est_value;

         sss_np_est_value = ((h_sm[i].x * h_sm[i].x) + (h_sm[i].y * h_sm[i].y)) / pss_np;
         atomicAdd(&sss_h12_np_est[col], sss_np_est_value);
    }

    __syncthreads();

    //  Implement the following expressions :
    //
    //  sss_h1_np_est(t) = 1 / (1 + sum(elem_mult(sqr(h_sm_h1), pss_np_inv_h1)));
    //  sss_h2_np_est(t) = 1 / (1 + sum(elem_mult(sqr(h_sm_h2), pss_np_inv_h2)));

    for (unsigned int i = tid; i < 62 * 2; i += n_pss) {
        sss_h12_np_est[i] = 1.0 / (1.0 + sss_h12_np_est[i]);
    }

    __syncthreads();

    // implment : extract_psss(capbuf.mid(ext_sss_dft_location,128),-peak_freq,k_factor,fs_programmed)

    //  index   :  0,  1,       30, 31, 32, 33,    , 61
    //  dft_out : 97, 98, .... 127,  1,  2,  3, ..., 31

    const unsigned int ext_sss_dft_location = pss_dft_location - 128 - 32;
    for (unsigned int t = ext_sss_dft_location + 2, i = 2; i < 128; i++, t++) {
        /* 128 points data shift, so k multiply from 0 to 125 */

        s_capbuf[i - 2].x = COMPLEX_MUL_REAL(d_capbuf[t], shift[i]);
        s_capbuf[i - 2].y = COMPLEX_MUL_IMAG(d_capbuf[t], shift[i]);
    }

    // time shift 2 points
    s_capbuf[126].x = COMPLEX_MUL_REAL(d_capbuf[ext_sss_dft_location], shift[0]);
    s_capbuf[126].y = COMPLEX_MUL_IMAG(d_capbuf[ext_sss_dft_location], shift[0]);
    s_capbuf[127].x = COMPLEX_MUL_REAL(d_capbuf[ext_sss_dft_location + 1], shift[1]);
    s_capbuf[127].y = COMPLEX_MUL_IMAG(d_capbuf[ext_sss_dft_location + 1], shift[1]);

    kernel_fft_radix2(s_capbuf, 128);

    //  index   :  0,  1,       30, 31, 32, 33,    , 61
    //  dft_out : 97, 98, .... 127,  1,  2,  3, ..., 31
    //
    //  vec pss_np_inv_h1 = 1.0 / pss_np(itpp_ext::matlab_range(0,2,n_pss-1));
    //  vec pss_np_inv_h2 = 1.0 / pss_np(itpp_ext::matlab_range(1,2,n_pss-1));
    //
    //  for (uint8 t = 0; t < 62; t++) {
    //
    //      sss_h1_np_est(t) = 1 / (1 + sum(elem_mult(sqr(h_sm_h1), pss_np_inv_h1)));
    //      sss_h2_np_est(t) = 1 / (1 + sum(elem_mult(sqr(h_sm_h2), pss_np_inv_h2)));
    //
    //      ...
    //      sss_h1_ext_est(t) = sss_h1_np_est(t) * sum(elem_mult(conj(h_sm_h1), to_cvec(pss_np_inv_h1), sss_ext_raw.get_col(t).get(itpp_ext::matlab_range(0,2,n_pss-1))));
    //      sss_h2_ext_est(t) = sss_h2_np_est(t) * sum(elem_mult(conj(h_sm_h2), to_cvec(pss_np_inv_h2), sss_ext_raw.get_col(t).get(itpp_ext::matlab_range(1,2,n_pss-1))));
    //  }

    const double pss_np_inv_SQRT128_INV = (1 / pss_np) * SQRT128_INV;

    for (unsigned int i = 0, col = (tid & 1) * 62; i < 31; i++, col++) {

         real = imag = sss_h12_np_est[col] * pss_np_inv_SQRT128_INV;

         real *= (h_sm[i].x * s_capbuf[97 + i].x + h_sm[i].y * s_capbuf[97 + i].y);
         imag *= (h_sm[i].x * s_capbuf[97 + i].y - h_sm[i].y * s_capbuf[97 + i].x);

         atomicAdd(&sss_h12_ext_est[col].x, real);
         atomicAdd(&sss_h12_ext_est[col].y, imag);

         real = imag = sss_h12_np_est[col + 31] * pss_np_inv_SQRT128_INV;

         real *= (h_sm[i + 31].x * s_capbuf[1 + i].x + h_sm[i + 31].y * s_capbuf[1 + i].y);
         imag *= (h_sm[i + 31].x * s_capbuf[1 + i].y - h_sm[i + 31].y * s_capbuf[1 + i].x);

         atomicAdd(&sss_h12_ext_est[col + 31].x, real);
         atomicAdd(&sss_h12_ext_est[col + 31].y, imag);
    }

    __syncthreads();

    // implment : extract_psss(capbuf.mid(nrm_sss_dft_location,128),-peak_freq,k_factor,fs_programmed)

    const unsigned int nrm_sss_dft_location = pss_dft_location - 128 - 9;
    for (unsigned int t = nrm_sss_dft_location + 2, i = 2; i < 128; i++, t++) {
        s_capbuf[i - 2].x = COMPLEX_MUL_REAL(d_capbuf[t], shift[i]);
        s_capbuf[i - 2].y = COMPLEX_MUL_IMAG(d_capbuf[t], shift[i]);
    }

    // time shift 2 points
    s_capbuf[126].x = COMPLEX_MUL_REAL(d_capbuf[nrm_sss_dft_location], shift[0]);
    s_capbuf[126].y = COMPLEX_MUL_IMAG(d_capbuf[nrm_sss_dft_location], shift[0]);
    s_capbuf[127].x = COMPLEX_MUL_REAL(d_capbuf[nrm_sss_dft_location + 1], shift[1]);
    s_capbuf[127].y = COMPLEX_MUL_IMAG(d_capbuf[nrm_sss_dft_location + 1], shift[1]);

    kernel_fft_radix2(s_capbuf, 128);

    // sss_h1_nrm_est(t) = sss_h1_np_est(t)*sum(elem_mult(conj(h_sm_h1), to_cvec(pss_np_inv_h1), sss_nrm_raw.get_col(t).get(itpp_ext::matlab_range(0,2,n_pss-1))));
    // sss_h2_nrm_est(t) = sss_h2_np_est(t)*sum(elem_mult(conj(h_sm_h2), to_cvec(pss_np_inv_h2), sss_nrm_raw.get_col(t).get(itpp_ext::matlab_range(1,2,n_pss-1))));

    for (unsigned int i = 0, col = (tid & 1) * 62; i < 31; i++, col++) {

         real = imag = sss_h12_np_est[col] * pss_np_inv_SQRT128_INV;

         real *= (h_sm[i].x * s_capbuf[97 + i].x + h_sm[i].y * s_capbuf[97 + i].y);
         imag *= (h_sm[i].x * s_capbuf[97 + i].y - h_sm[i].y * s_capbuf[97 + i].x);

         atomicAdd(&sss_h12_nrm_est[col].x, real);
         atomicAdd(&sss_h12_nrm_est[col].y, imag);

         real = imag = sss_h12_np_est[col + 31] * pss_np_inv_SQRT128_INV;

         real *= (h_sm[i + 31].x * s_capbuf[1 + i].x + h_sm[i + 31].y * s_capbuf[1 + i].y);
         imag *= (h_sm[i + 31].x * s_capbuf[1 + i].y - h_sm[i + 31].y * s_capbuf[1 + i].x);

         atomicAdd(&sss_h12_nrm_est[col + 31].x, real);
         atomicAdd(&sss_h12_nrm_est[col + 31].y, imag);
    }

    /* perhaps in the future, we can combine sss_detect_ml_kernel() into this one */
    for (unsigned int i = tid; i < 62 * 2; i += n_pss) {
        d_sss_h12_np_est[i]  = sss_h12_np_est[i];
        d_sss_h12_ext_est[i].x = sss_h12_ext_est[i].x;
        d_sss_h12_ext_est[i].y = sss_h12_ext_est[i].y;
    }

    __syncthreads();

    for (unsigned int i = tid; i < 62 * 2; i += n_pss) {
        d_sss_h12_nrm_est[i].x = sss_h12_nrm_est[i].x;
        d_sss_h12_nrm_est[i].y = sss_h12_nrm_est[i].y;
    }
}



/*
 *
 */
__device__ void sss_detect_ml_helper_function(float *sss_np_est, cufftDoubleComplex *sss_est, int *sss_try_orig,
                                              /* output */
                                              double *log_lik)
{
    __shared__ cufftDoubleComplex buffer1[124], coeff;
    __shared__ double buffer2[124];

    const unsigned int tid = threadIdx.x;
    double real, imag, r;

    /* elem_mult(conj(sss_h12_est), sss_h12_try) */
    buffer1[tid].x =   sss_est[tid].x * sss_try_orig[tid];
    buffer1[tid].y = - sss_est[tid].y * sss_try_orig[tid];
    
    __syncthreads();

    for (int s = 128 / 2; s > 0; s >>= 1) {
        if ((tid < s) && (tid + s < 124)) {
            buffer1[tid].x += buffer1[tid + s].x;
            buffer1[tid].y += buffer1[tid + s].y;
        }
        __syncthreads();
    }

    if (tid == 0) {
        real = buffer1[0].x; imag = buffer1[0].y;
        r = sqrt(real * real + imag * imag);
        coeff.x =  real / r;
        coeff.y = -imag / r;
    }

    __syncthreads();

    real = sss_try_orig[tid] * coeff.x - sss_est[tid].x;
    imag = sss_try_orig[tid] * coeff.y - sss_est[tid].y;
    buffer2[tid] = - (real * real + imag * imag) / sss_np_est[tid];

    __syncthreads();

    for (unsigned int s = 128 / 2; s > 0; s >>= 1) {
        if ((tid < s) && (tid + s < 124)) {
            buffer2[tid] += buffer2[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *log_lik = buffer2[0];
    }
}



/*
 *
 */
__global__ void sss_detect_ml_kernel(float *d_sss_h12_np_est, cufftDoubleComplex *d_sss_h12_nrm_est, cufftDoubleComplex *d_sss_h12_ext_est,
                                     int n_id_2,
                                     // output
                                     double *d_log_lik_nrm, double *d_log_lik_ext)
{
    __shared__ float sss_h12_np_est[124];
    __shared__ int sss_h12_try[124], sss_h21_try[124], *sss_try;
    __shared__ cufftDoubleComplex sss_h12_nrm_est[124], sss_h12_ext_est[124], *sss_h12_est;
    __shared__ unsigned int word12[4], word21[4];
    __shared__ double *d_log_lik;

    const unsigned int bid = blockIdx.x;
    const unsigned int tid = threadIdx.x;

    if (tid == 0) {
        const int bid_mod4 = bid % 4;
        const int bid_div4 = bid / 4;

        word12[0] = word21[2] = sss_fd[bid_div4][n_id_2][0][0];
        word12[1] = word21[3] = sss_fd[bid_div4][n_id_2][0][1];

        word12[2] = word21[0] = sss_fd[bid_div4][n_id_2][1][0];
        word12[3] = word21[1] = sss_fd[bid_div4][n_id_2][1][1];

        // sss_detect_ml_helper_function(sss_h12_np_est, sss_h12_nrm_est, sss_h12_try, &d_log_lik_nrm[bid * 2 + 0]);
        // sss_detect_ml_helper_function(sss_h12_np_est, sss_h12_nrm_est, sss_h21_try, &d_log_lik_nrm[bid * 2 + 1]);
        // sss_detect_ml_helper_function(sss_h12_np_est, sss_h12_ext_est, sss_h12_try, &d_log_lik_ext[bid * 2 + 0]);
        // sss_detect_ml_helper_function(sss_h12_np_est, sss_h12_ext_est, sss_h21_try, &d_log_lik_ext[bid * 2 + 1]);

        sss_h12_est = (bid_mod4 / 2) ? sss_h12_ext_est : sss_h12_nrm_est;
        sss_try = (bid_mod4 & 1) ? sss_h21_try : sss_h12_try;
        d_log_lik = (bid_mod4 / 2) ? &d_log_lik_ext[(bid_mod4 & 1) * 168 + bid_div4] : &d_log_lik_nrm[(bid_mod4 & 1) * 168 + bid_div4]; 
    }

    __syncthreads();

    sss_h12_np_est[tid] = d_sss_h12_np_est[tid];
    sss_h12_nrm_est[tid].x = d_sss_h12_nrm_est[tid].x;
    sss_h12_nrm_est[tid].y = d_sss_h12_nrm_est[tid].y;
    sss_h12_ext_est[tid].x = d_sss_h12_ext_est[tid].x;
    sss_h12_ext_est[tid].y = d_sss_h12_ext_est[tid].y;

    /* to_cvec(sss_h12_try_orig) and to_cvec(sss_h21_try_orig) */
    /* tid = 0...30, 31...61, 62...92, 93...123 */

    sss_h12_try[tid] = 1 - 2 * ((word12[tid / 31] >> (tid % 31)) & 1);
    sss_h21_try[tid] = 1 - 2 * ((word21[tid / 31] >> (tid % 31)) & 1);

    __syncthreads();

    sss_detect_ml_helper_function(sss_h12_np_est, sss_h12_est, sss_try, d_log_lik);
}



/*
 *
 */
__device__ void log_lik_maximum(double *d_in, double *d_max, int *d_idx)
{
    __shared__ double shared[168 * 4];
    __shared__ int index[168 * 4];

    const unsigned int tid = threadIdx.x;
    const unsigned int tid_mod168 = tid % 168;
    const unsigned int tid_div168 = tid / 168;

    shared[tid] = d_in[tid];
    index[tid] = tid_mod168;

    __syncthreads();

    for (unsigned int s = 256 / 2; s > 0; s >>= 1) {
         if ((tid_mod168 < s) && (tid_mod168 + s < 168)) {
             if (shared[tid] < shared[tid + s]) {
                 double value;
                 value = shared[tid];
                 shared[tid] = shared[tid + s];
                 shared[tid + s] = value;

                 int idx;
                 idx = index[tid];
                 index[tid] = index[tid + s];
                 index[tid + s] = idx;
             }
         }
         __syncthreads();
    }

    if (tid_mod168 == 0) {
        d_max[tid_div168] = shared[tid_div168 * 168];
        d_idx[tid_div168] = index[tid_div168 * 168];
    }
}



/*
 *
 */
__device__ void log_lik_mean_var(double *d_in, double *d_mean, double *d_var)
{
    __shared__ double shared[168 * 4], shared_sqr[168 * 4];
    const unsigned int tid = threadIdx.x;
    double mean;

    shared[tid] = d_in[tid];
    shared_sqr[tid] = d_in[tid] * d_in[tid];

    __syncthreads();

    for (unsigned int s = 1024 / 2; s > 0; s >>= 1) {
        if ((tid < s) && (tid + s < 168 * 4)) {
             shared[tid] += shared[tid + s];
             shared_sqr[tid] += shared_sqr[tid + s];
        }
        __syncthreads();
    }

    mean = shared[0] / (168 * 4);

    if (tid == 0) {
        *d_mean = mean;
        *d_var = (shared_sqr[tid] / (168 * 4)) - mean * mean;
    }
}



/*
 *
 */
__global__ void sss_detect_ml_decision_kernel(double *d_log_lik, int thresh2_n_sigma, int ind,
                                              double fc_requested, double fc_programmed, double fs_programmed, double freq,
                                              // output
                                              int *d_n_id_1_est, double *d_frame_start, int *d_cp_type)
{
    __shared__ double log_lik_max[4], log_lik_mean, log_lik_var;
    __shared__ int log_lik_idx[4];
 
    const unsigned int tid = threadIdx.x;

    log_lik_maximum(&d_log_lik[0], &log_lik_max[0], &log_lik_idx[0]);
    __syncthreads();

    log_lik_mean_var(&d_log_lik[0], &log_lik_mean, &log_lik_var);
    __syncthreads(); 

    if (tid == 0) {
        double max_value = log_lik_max[0];
        int max_idx = 0;

        for (int i = 1; i <= 3; i++) {
            if (log_lik_max[i] > max_value) {
                max_value = log_lik_max[i];
                max_idx = i;
            }
        }

        *d_n_id_1_est = -1;

        const double k_factor = (fc_requested - freq) / fc_programmed;
        double frame_start = ind + (128 + 9 - 960 - 2) * 16 / FS_LTE * fs_programmed * k_factor;
        if (max_idx & 1)
            frame_start = frame_start + 9600 * k_factor * 16 / FS_LTE * fs_programmed * k_factor;

#undef WRAP
#define WRAP(x,sm,lg) (fmod((x)-(sm),(lg)-(sm))+(sm))
        frame_start = WRAP(frame_start, -0.5, (2*9600.0-0.5)*16/FS_LTE*fs_programmed*k_factor);

        if (max_value < log_lik_mean + sqrt(log_lik_var) * thresh2_n_sigma)
            return;
        *d_cp_type = (max_idx / 2) ? 2 /* extend cp */ : 1 /* normal cp */;
        *d_n_id_1_est = log_lik_idx[max_idx];
        *d_frame_start = frame_start;
    }
}



/*
 *
 */
__global__ void pss_sss_foe_multiblocks_step1_kernel(cufftDoubleComplex *d_capbuf, int n_sss,
                                                     unsigned short n_id_cell, int n_symb_dl, double first_sss_dft_location, unsigned int pss_sss_dist, int sn,
                                                     double fc_requested, double fc_programmed, double fs_programmed, double freq,
                                                     // output
                                                     cufftDoubleComplex *d_M)
{
    __shared__ cufftDoubleComplex s_pss_dft[128], s_sss_dft[128];
    __shared__ cufftDoubleComplex h_raw_fo_pss[62], sss_raw_fo[62];
    __shared__ cufftDoubleComplex coeff;
    __shared__ float pss_np, M_real, M_imag;

    const unsigned int bid = blockIdx.x;
    const unsigned int tid = threadIdx.x;

    cufftDoubleComplex shift, acc, *p_pss_fd;
    unsigned int sss_fd_bit;
    const double k_factor = (fc_requested - freq) / fc_programmed;
    unsigned int *p_sss_fd;     
    const int n_id_1 = n_id_cell / 3, n_id_2 = n_id_cell % 3;
    double real, imag;

    // Determine where we can find both PSS and SSS
    // Q : original code snippet for EXTENDED_CP : pss_sss_dist=round_i((128+32)*k_factor); ???
    const unsigned int sss_dft_location = lround(first_sss_dft_location + bid * (9600 * 16 / FS_LTE * fs_programmed * k_factor));

    // Find the PSS and use it to estimate the channel.
    const unsigned int pss_dft_location = sss_dft_location + pss_sss_dist;

    if (tid == 0) {
        pss_np = M_real = M_imag = 0.0;

        // exp(J*pi*(-cell_in.freq)/(FS_LTE/16/2)*(-pss_sss_dist))

        coeff.x = cos(CUDART_PI * freq / (FS_LTE / 16 / 2) * pss_sss_dist);
        coeff.y = sin(CUDART_PI * freq / (FS_LTE / 16 / 2) * pss_sss_dist);
    }

    p_pss_fd = (cufftDoubleComplex *)&pss_fd[n_id_2];
    p_sss_fd = (unsigned int *)&sss_fd[n_id_1][n_id_2][(bid + sn) & 1];

    double k = CUDART_PI * (-freq) / ((fs_programmed * k_factor) / 2);

    shift.x = cos(k * tid);
    shift.y = sin(k * tid);

    __syncthreads();

    if (tid < 2) {
        s_pss_dft[tid + 126].x = COMPLEX_MUL_REAL(d_capbuf[pss_dft_location + tid], shift);
        s_pss_dft[tid + 126].y = COMPLEX_MUL_IMAG(d_capbuf[pss_dft_location + tid], shift);
        s_sss_dft[tid + 126].x = COMPLEX_MUL_REAL(d_capbuf[sss_dft_location + tid], shift);
        s_sss_dft[tid + 126].y = COMPLEX_MUL_IMAG(d_capbuf[sss_dft_location + tid], shift);
    } else if (tid < 128) {
        s_pss_dft[tid - 2].x = COMPLEX_MUL_REAL(d_capbuf[pss_dft_location + tid], shift);
        s_pss_dft[tid - 2].y = COMPLEX_MUL_IMAG(d_capbuf[pss_dft_location + tid], shift);
        s_sss_dft[tid - 2].x = COMPLEX_MUL_REAL(d_capbuf[sss_dft_location + tid], shift);
        s_sss_dft[tid - 2].y = COMPLEX_MUL_IMAG(d_capbuf[sss_dft_location + tid], shift);
    }

    __syncthreads();

    if (tid == 0) {
        kernel_fft_radix2(s_pss_dft, 128);
    } else if (tid == 64) {
        kernel_fft_radix2(s_sss_dft, 128);
    }
    
    __syncthreads();

    // h_raw_fo_pss : concat(dft_out.right(31), dft_out.mid(1,31)) .* conj(pss_fd)
    // sss_raw_fo   : concat(dft_out.right(31), dft_out.mid(1,31)) .* exp(J*pi*-cell_in.freq/(FS_LTE/16/2)*-pss_sss_dist) .* sss_fd(n_id_1,n_id_2, 0 or 1)

    // index   :  0,  1,       30, 31, 32, 33,    , 61
    // dft_out : 97, 98, .... 127,  1,  2,  3, ..., 31
    // pss_fd  :  0,  1, ....  30, 31, 32, 33, ..., 61

    if (tid < 31) {
        sss_fd_bit = (p_sss_fd[0] >> tid) & 1;

        h_raw_fo_pss[tid].x = SQRT128_INV * (s_pss_dft[tid + 97].x * p_pss_fd[tid].x + s_pss_dft[tid + 97].y * p_pss_fd[tid].y);
        h_raw_fo_pss[tid].y = SQRT128_INV * (s_pss_dft[tid + 97].y * p_pss_fd[tid].x - s_pss_dft[tid + 97].x * p_pss_fd[tid].y);
        sss_raw_fo[tid].x = SQRT128_INV * COMPLEX_MUL_REAL(s_sss_dft[tid + 97], coeff);
        sss_raw_fo[tid].y = SQRT128_INV * COMPLEX_MUL_IMAG(s_sss_dft[tid + 97], coeff);
    } else if (tid < 62) {
        sss_fd_bit = ((p_sss_fd[1] >> (tid - 31)) & 1);

        h_raw_fo_pss[tid].x = SQRT128_INV * (s_pss_dft[tid - 30].x * p_pss_fd[tid].x + s_pss_dft[tid - 30].y * p_pss_fd[tid].y);
        h_raw_fo_pss[tid].y = SQRT128_INV * (s_pss_dft[tid - 30].y * p_pss_fd[tid].x - s_pss_dft[tid - 30].x * p_pss_fd[tid].y);
        sss_raw_fo[tid].x = SQRT128_INV * COMPLEX_MUL_REAL(s_sss_dft[tid - 30], coeff);
        sss_raw_fo[tid].y = SQRT128_INV * COMPLEX_MUL_IMAG(s_sss_dft[tid - 30], coeff);
    }

    __syncthreads();

    // Smoothing... Basic...

    //                  t  = 0, 1, 2, 3,  4,  5,  6,  7,  8,  9, 10,        , 53, 54, 55, 56, 57, 58, 59, 60, 61
    // lt  = MAX(0, t - 6) = 0, 0, 0, 0,  0,  0,  0,  1,  2,  3,  4, ...... , 47, 48, 49, 50, 51, 52, 53, 54, 55
    // rt  = MIN(61,t + 6) = 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, ...... , 59, 60, 61, 61, 61, 61, 61, 61, 61
    // len = rt-lt + 1     = 7, 8, 9,10, 11, 12, 13, 13, 13, 13, 13, ...... , 13, 13, 13, 12, 11, 10,  9,  8,  7

    acc.x = 0.0; acc.y = 0.0;
    if (tid < 6) {
        for (unsigned int i = 0; i <= tid + 6; i++) {
            acc.x += h_raw_fo_pss[i].x;
            acc.y += h_raw_fo_pss[i].y;
        }
        acc.x /= (tid + 7);
        acc.y /= (tid + 7);
    } else if (tid < 56) {
        for (unsigned int i = tid - 6; i <= tid + 6; i++) {
            acc.x += h_raw_fo_pss[i].x;
            acc.y += h_raw_fo_pss[i].y;
        }
        acc.x /= 13;
        acc.y /= 13;
    } else if (tid < 62) {
        for (unsigned int i = tid - 6; i <= 61; i++) {
            acc.x += h_raw_fo_pss[i].x;
            acc.y += h_raw_fo_pss[i].y;
        }
        acc.x /= (68 - tid);
        acc.y /= (68 - tid);
    }

    // Estimate noise power.
    // pss_np = sigpower(h_sm.get_row(tid) - h_raw_fo_pss.get_row(tid));

    if (tid < 62) {
        double noise_r = acc.x - h_raw_fo_pss[tid].x;
        double noise_i = acc.y - h_raw_fo_pss[tid].y;
        atomicAdd(&pss_np, (float)(noise_r * noise_r + noise_i * noise_i));
    }

    __syncthreads();

    // sss_raw_fo : concat(dft_out.right(31), dft_out.mid(1,31)) .* conj(pss_fd)

    // index   :  0,  1,       30, 31, 32, 33,    , 61
    // dft_out : 97, 98, .... 127,  1,  2,  3, ..., 31
    // pss_fd  :  0,  1, ....  30, 31, 32, 33, ..., 61

    // Compare PSS to SSS. With no frequency offset, arg(M) is zero.

    if (tid < 62) {
        // sss_fd_bit = 1 : a+bj -> -a-bj -> -a+bj
        // sss_fd_bit = 0 : a+bj ->  a+bj ->  a-bj

        if (sss_fd_bit)
            sss_raw_fo[tid].x = - sss_raw_fo[tid].x;
        else
            sss_raw_fo[tid].y = - sss_raw_fo[tid].y;
 
        // elem_mult(conj(sss_raw_fo), h_raw_fo_pss, to_cvec(elem_mult(sqr(h_sm), 1.0/(2*sqr(h_sm)*pss_np+sqr(pss_np))

        double sqr_sm = (acc.x * acc.x + acc.y * acc.y);
        imag = real = (sqr_sm * 62) / ((2 * sqr_sm * pss_np) + (pss_np * pss_np / 62));

        real *= COMPLEX_MUL_REAL(sss_raw_fo[tid], h_raw_fo_pss[tid]);
        imag *= COMPLEX_MUL_IMAG(sss_raw_fo[tid], h_raw_fo_pss[tid]);

        atomicAdd(&M_real, (float)real);
        atomicAdd(&M_imag, (float)imag);
    }

    __syncthreads();

    if (tid == 0) {
        d_M[bid].x = 1.0 * M_real;
        d_M[bid].y = 1.0 * M_imag;
    }
}



/*
 *
 */
__global__ void pss_sss_foe_multiblocks_step2_kernel(cufftDoubleComplex *d_M, int n_sss, unsigned int pss_sss_dist,
                                                     double fc_requested, double fc_programmed, double fs_programmed, double freq,
                                                     // output
                                                     double *d_adjust_f)
{
    __shared__ cufftDoubleComplex s_M[64];

    const unsigned int tid = threadIdx.x;
    const double k_factor = (fc_requested - freq) / fc_programmed;

    if (tid > 64) {
        printf("Caution : number of thread is larger than reserved space !\n");
        return;
    }

    if (tid < n_sss) {
        s_M[tid] = d_M[tid];
    }

    __syncthreads();

    for (unsigned int s = 64; s > 0; s >>= 1) {
        if ((tid < s) && ((tid + s) < n_sss) && ((tid + s) < 50)) {
            s_M[tid].x += s_M[tid + s].x;
            s_M[tid].y += s_M[tid + s].y;
        }
        __syncthreads();
    }
    

    if (tid == 0) {
        *d_adjust_f = angle((float)s_M[0].x, (float)s_M[0].y) / (2 * CUDART_PI) / (1 / (fs_programmed * k_factor) * pss_sss_dist);
    }
}



// Perform FOE using only the PSS and SSS.
// The PSS correlation peak gives us the frequency offset within 2.5kHz.
// The PSS/SSS can be used to estimate the frequency offset within a
// much finer resolution.

/*
 *
 */
__global__ void pss_sss_foe_singleblock_kernel(cufftDoubleComplex *d_capbuf, int n_sss,
                                               unsigned short n_id_cell, int n_symb_dl, double first_sss_dft_location, unsigned int pss_sss_dist, int sn,
                                               double fc_requested, double fc_programmed, double fs_programmed, double freq,
                                               // output
                                               double *d_adjust_f)
{
    __shared__ float M_real, M_imag;
    __shared__ cufftDoubleComplex shift[128], coeff, *p_pss_fd;
    cufftDoubleComplex s_capbuf[128];
    cufftDoubleComplex h_raw_fo_pss[62], h_sm[62];
    cufftDoubleComplex acc, M;
    unsigned int *p_sss_fd;

    const double k_factor = (fc_requested - freq) / fc_programmed;
    const unsigned int tid = threadIdx.x;
    const int n_id_1 = n_id_cell / 3, n_id_2 = n_id_cell % 3;
    float real, imag;
    double pss_np;

    // Determine where we can find both PSS and SSS
    // Q : original code snippet for EXTENDED_CP : pss_sss_dist=round_i((128+32)*k_factor); ???
    const unsigned int sss_dft_location = lround(first_sss_dft_location + tid * (9600 * 16 / FS_LTE * fs_programmed * k_factor));

    // Find the PSS and use it to estimate the channel.
    const unsigned int pss_dft_location = sss_dft_location + pss_sss_dist;

    if (tid == 0) {
        M_real = 0.0; M_imag = 0.0;

        // exp(J*pi*(-cell_in.freq)/(FS_LTE/16/2)*(-pss_sss_dist))

        coeff.x = cos(CUDART_PI * freq / (FS_LTE / 16 / 2) * pss_sss_dist);
        coeff.y = sin(CUDART_PI * freq / (FS_LTE / 16 / 2) * pss_sss_dist);

        p_pss_fd = (cufftDoubleComplex *)&pss_fd[n_id_2];
    }

    p_sss_fd = (unsigned int *)&sss_fd[n_id_1][n_id_2][(tid + sn) & 1];

    double k = CUDART_PI * (-freq) / ((fs_programmed * k_factor) / 2);

    for (unsigned int i = tid; i < 128; i += n_sss) {
        shift[i].x = cos(k * i);
        shift[i].y = sin(k * i);
    }

    __syncthreads();

    // implement extract_psss(capbuf.mid(pss_dft_location, 128), -cell_in.freq, k_factor, fs_programmed)
    for (unsigned int t = pss_dft_location + 2, i = 2; i < 128; i++, t++) {
        /* 128 points data shift, so k multiply from 0 to 125 */

        s_capbuf[i - 2].x = COMPLEX_MUL_REAL(d_capbuf[t], shift[i]);
        s_capbuf[i - 2].y = COMPLEX_MUL_IMAG(d_capbuf[t], shift[i]);
    }

    // time shift 2 points
    s_capbuf[126].x = COMPLEX_MUL_REAL(d_capbuf[pss_dft_location], shift[0]);
    s_capbuf[126].y = COMPLEX_MUL_IMAG(d_capbuf[pss_dft_location], shift[0]);
    s_capbuf[127].x = COMPLEX_MUL_REAL(d_capbuf[pss_dft_location + 1], shift[1]);
    s_capbuf[127].y = COMPLEX_MUL_IMAG(d_capbuf[pss_dft_location + 1], shift[1]);

    kernel_fft_radix2(s_capbuf, 128);

    // h_raw_fo_pss : concat(dft_out.right(31), dft_out.mid(1,31)) .* conj(pss_fd)

    // index   :  0,  1,       30, 31, 32, 33,    , 61
    // dft_out : 97, 98, .... 127,  1,  2,  3, ..., 31
    // pss_fd  :  0,  1, ....  30, 31, 32, 33, ..., 61

    for (unsigned int i = 1; i <= 31; i++) {
        // concat(dft_out.right(31), dft_out.mid(1,31)) * conj(pss_fd)

        h_raw_fo_pss[i + 30].x = SQRT128_INV * (s_capbuf[i].x * p_pss_fd[i + 30].x + s_capbuf[i].y * p_pss_fd[i + 30].y);
        h_raw_fo_pss[i + 30].y = SQRT128_INV * (s_capbuf[i].y * p_pss_fd[i + 30].x - s_capbuf[i].x * p_pss_fd[i + 30].y);

        h_raw_fo_pss[i - 1].x = SQRT128_INV * (s_capbuf[i + 96].x * p_pss_fd[i - 1].x + s_capbuf[i + 96].y * p_pss_fd[i - 1].y);
        h_raw_fo_pss[i - 1].y = SQRT128_INV * (s_capbuf[i + 96].y * p_pss_fd[i - 1].x - s_capbuf[i + 96].x * p_pss_fd[i - 1].y);
    }

    // Smoothing... Basic...

    //                  t  = 0, 1, 2, 3,  4,  5,  6,  7,  8,  9, 10,        , 53, 54, 55, 56, 57, 58, 59, 60, 61
    // lt  = MAX(0, t - 6) = 0, 0, 0, 0,  0,  0,  0,  1,  2,  3,  4, ...... , 47, 48, 49, 50, 51, 52, 53, 54, 55
    // rt  = MIN(61,t + 6) = 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, ...... , 59, 60, 61, 61, 61, 61, 61, 61, 61
    // len = rt-lt + 1     = 7, 8, 9,10, 11, 12, 13, 13, 13, 13, 13, ...... , 13, 13, 13, 12, 11, 10,  9,  8,  7

    acc.x = 0.0; acc.y = 0.0;
    pss_np = 0.0;
    for (unsigned int i = 0; i < 6; i++) {
        acc.x += h_raw_fo_pss[i].x;
        acc.y += h_raw_fo_pss[i].y;
    }

    for (unsigned int i = 0; i <= 6; i++) {
        acc.x += h_raw_fo_pss[i + 6].x;
        acc.y += h_raw_fo_pss[i + 6].y;

        h_sm[i].x = acc.x / (i + 7);
        h_sm[i].y = acc.y / (i + 7);
    }

    for (unsigned int i = 7; i <= 55; i++) {
        acc.x += (h_raw_fo_pss[i + 6].x - h_raw_fo_pss[i - 7].x);
        acc.y += (h_raw_fo_pss[i + 6].y - h_raw_fo_pss[i - 7].y);

        h_sm[i].x = acc.x / 13;
        h_sm[i].y = acc.y / 13;
    }

    for (unsigned int i = 56; i < 62; i++) {
        acc.x -= h_raw_fo_pss[i - 7].x;
        acc.y -= h_raw_fo_pss[i - 7].y;

        h_sm[i].x = acc.x / (61 + 7 - i);
        h_sm[i].y = acc.y / (61 + 7 - i);
    }

    // Estimate noise power.
    // pss_np = sigpower(h_sm.get_row(tid) - h_raw_fo_pss.get_row(tid));
    pss_np = 0.0;
    for (unsigned int i = 0; i < 62; i++) {
        double noise_r = h_sm[i].x - h_raw_fo_pss[i].x;
        double noise_i = h_sm[i].y - h_raw_fo_pss[i].y;
        pss_np += (noise_r * noise_r + noise_i * noise_i);
    }
    pss_np /= 62;

    // Calculate the SSS in the frequency domain

    // extract_psss(capbuf.mid(sss_dft_location,128),-cell_in.freq,k_factor,fs_programmed)
    for (unsigned int t = sss_dft_location + 2, i = 2; i < 128; i++, t++) {
        /* 128 points data shift, so k multiply from 0 to 125 */

        s_capbuf[i - 2].x = COMPLEX_MUL_REAL(d_capbuf[t], shift[i]);
        s_capbuf[i - 2].y = COMPLEX_MUL_IMAG(d_capbuf[t], shift[i]);
    }

    s_capbuf[126].x = COMPLEX_MUL_REAL(d_capbuf[sss_dft_location], shift[0]);
    s_capbuf[126].y = COMPLEX_MUL_IMAG(d_capbuf[sss_dft_location], shift[0]);
    s_capbuf[127].x = COMPLEX_MUL_REAL(d_capbuf[sss_dft_location + 1], shift[1]);
    s_capbuf[127].y = COMPLEX_MUL_IMAG(d_capbuf[sss_dft_location + 1], shift[1]);

    kernel_fft_radix2(s_capbuf, 128);

    // sss_raw_fo : concat(dft_out.right(31), dft_out.mid(1,31)) .* conj(pss_fd)

    // index   :  0,  1,       30, 31, 32, 33,    , 61
    // dft_out : 97, 98, .... 127,  1,  2,  3, ..., 31
    // pss_fd  :  0,  1, ....  30, 31, 32, 33, ..., 61

    // Compare PSS to SSS. With no frequency offset, arg(M) is zero.

    M.x = 0.0; M.y = 0.0;
    for (unsigned int i = 1, word1 = p_sss_fd[0], word2 = p_sss_fd[1]; i <= 31; i++, word1 >>= 1, word2 >>= 1) {
        cufftDoubleComplex sss_raw_fo;
        double sm_power;

        // index = 0, 1 ... 31 of sss_raw_fo

        sm_power = (h_sm[i - 1].x * h_sm[i - 1].x + h_sm[i - 1].y * h_sm[i - 1].y);
        sm_power = SQRT128_INV * sm_power / (2 * sm_power * pss_np + pss_np * pss_np);

        if (word1 & 1) sm_power *= -1.0;

        // dft_out .* sss_fd

        // conj( .* exp(J*pi*(-cell_in.freq)/(FS_LTE/16/2)*(-pss_sss_dist)))

        sss_raw_fo.x = COMPLEX_MUL_REAL(s_capbuf[i + 96], coeff);
        sss_raw_fo.y = -COMPLEX_MUL_IMAG(s_capbuf[i + 96], coeff);

        // .* h_raw_fo_pss

        M.x += sm_power * COMPLEX_MUL_REAL(sss_raw_fo, h_raw_fo_pss[i - 1]);
        M.y += sm_power * COMPLEX_MUL_IMAG(sss_raw_fo, h_raw_fo_pss[i - 1]);

        // index = 32, 33 ... 61 of sss_raw_fo

        sm_power = (h_sm[i + 30].x * h_sm[i + 30].x + h_sm[i + 30].y * h_sm[i + 30].y);
        sm_power = SQRT128_INV * sm_power / (2 * sm_power * pss_np + pss_np * pss_np);

        if (word2 & 1) sm_power *= -1.0;

        // dft_out .* sss_fd

        // conj( .* exp(J*pi*(-cell_in.freq)/(FS_LTE/16/2)*(-pss_sss_dist)))

        sss_raw_fo.x = COMPLEX_MUL_REAL(s_capbuf[i], coeff);
        sss_raw_fo.y = -COMPLEX_MUL_IMAG(s_capbuf[i], coeff);

        // .* h_raw_fo_pss

        M.x += sm_power * COMPLEX_MUL_REAL(sss_raw_fo, h_raw_fo_pss[i + 30]);
        M.y += sm_power * COMPLEX_MUL_IMAG(sss_raw_fo, h_raw_fo_pss[i + 30]);
    }

    real = 1.0 * M.x;
    imag = 1.0 * M.y;

    atomicAdd(&M_real, real);
    atomicAdd(&M_imag, imag);

    __syncthreads();

    if (tid == 0) {
        *d_adjust_f = angle(M_real, M_imag) / (2 * CUDART_PI) / (1 / (fs_programmed * k_factor) * pss_sss_dist);
    }
}



/**
 * Implement 36.211 7.2.
 * Generate Pseudo-random sequence and store the result into array rather than queue.
 * Bits are stored from LSB to MSB.
 *
 * \param init_in       Initial value of pseudo-random sequence generator
 * \param seqLn         How many consecutive pseudo-random number to be generated.
 * \param initOffset    The starting position of pseudo-random number to be generated
 * \param pSeqOut       Pointer to output array of UNSG32 which holds generated pseudo-random sequence
 */
__device__ void pn_seq_lsb_to_msb(unsigned int d_init_in, unsigned int d_seq_len, unsigned int d_init_offset, unsigned int *d_pseq_out)
{
    unsigned int x1, x2, tmp_val;
    unsigned int i;

    const unsigned int m2_v1600[31] = {
        0x0099110E, 0x004C8887, 0x40264444, 0x20132222, 0x10099111, 0x4804C88F, 0x64026440, 0x32013220,
        0x19009910, 0x0C804C88, 0x06402644, 0x03201322, 0x01900991, 0x40C804CF, 0x60640260, 0x30320130,
        0x18190098, 0x0C0C804C, 0x06064026, 0x03032013, 0x4181900E, 0x20C0C807, 0x50606404, 0x28303202,
        0x14181901, 0x4A0C0C87, 0x65060644, 0x32830322, 0x19418191, 0x4CA0C0CF, 0x66506060
    };

    unsigned int init_in = d_init_in;
    unsigned int seq_len = d_seq_len;
    unsigned int init_offset = d_init_offset;
    unsigned int *pseq_out = d_pseq_out;

    /* x1 is independent of c_init,
       so it can be pre-calculated at N=1600.
       x2 depends on c_init,
       so it need to multiply M^1570 to obtain its value at N=1600 */
    x1 = 0x54D21B24;
    x2 = 0;
    for (i = 0; i < 31; i++) {
        tmp_val = init_in & m2_v1600[i];

        /* determine there are even or odd number of bits set in tmp_val */
        tmp_val ^= (tmp_val >> 16);
        tmp_val ^= (tmp_val >> 8);
        tmp_val ^= (tmp_val >> 4);
        tmp_val &= 0xF;
        tmp_val = ((0x6996 >> tmp_val) & 1);

        x2 |= (tmp_val << (31 - i));
    }

    for (i = 0; i < init_offset; i++) {
        x1 >>= 1;
        x1  |= (((0x55AA >> (x1 & 0xF)) & 1) << 31);   /* bit0 of 0x55AA is x(0)^x(3) of [3:0] */
        x2 >>= 1;
        x2  |= (((0x6996 >> (x2 & 0xF)) & 1) << 31);   /* bit0 of 0x55AA is x(0)^x(3) */
    }

    tmp_val = 0;
    for (i = 0; i < seq_len; i++) {
        /* store from LSB to MSB */
        tmp_val |= ((x1 ^ x2) >> (31 - (i % 32)));
        if (((i + 1) % 32) == 0) {
            *pseq_out++ = tmp_val;
            tmp_val    = 0;
        }

        x1 >>= 1;
        x1 |= (((0x55AA >> (x1 & 0xF)) & 1) << 31);
        x2 >>= 1;
        x2 |= (((0x6996 >> (x2 & 0xF)) & 1) << 31);
    }

    if (seq_len && (seq_len % 32))
        *pseq_out = tmp_val;
}

/**
 * Implement 36.211 7.2.
 * Generate Pseudo-random sequence and store the result into array rather than queue.
 * Bits are stored from MSB to LSB.
 *
 * \param init_in       Initial value of pseudo-random sequence generator
 * \param seqLn         How many consecutive pseudo-random number to be generated.
 * \param initOffset    The starting position of pseudo-random number to be generated
 * \param pSeqOut       Pointer to output array of unsigned int which holds generated pseudo-random sequence
 */
__device__ void pn_seq_msb_to_lsb(unsigned int d_init_in, unsigned int d_seq_len, unsigned int d_init_offset, unsigned int *d_pseq_out)
{
    unsigned int x1, x2, tmp_val;
    unsigned int i;

    const unsigned int m2_v1600[31] = {
        0x0099110E, 0x004C8887, 0x40264444, 0x20132222, 0x10099111, 0x4804C88F, 0x64026440, 0x32013220,
        0x19009910, 0x0C804C88, 0x06402644, 0x03201322, 0x01900991, 0x40C804CF, 0x60640260, 0x30320130,
        0x18190098, 0x0C0C804C, 0x06064026, 0x03032013, 0x4181900E, 0x20C0C807, 0x50606404, 0x28303202,
        0x14181901, 0x4A0C0C87, 0x65060644, 0x32830322, 0x19418191, 0x4CA0C0CF, 0x66506060
    };

    unsigned int init_in = d_init_in;
    unsigned int seq_len = d_seq_len;
    unsigned int init_offset = d_init_offset;
    unsigned int *pseq_out = d_pseq_out;

    /* x1 is independent of c_init,
       so it can be pre-calculated at N=1600.
       x2 depends on c_init,
       so it need to multiply M^1570 to obtain its value at N=1600 */
    x1 = 0x54D21B24;
    x2 = 0;
    for (i = 0; i < 31; i++) {
        tmp_val = init_in & m2_v1600[i];

        /* determine there are even or odd number of bits set in tmp_val */
        tmp_val ^= (tmp_val >> 16);
        tmp_val ^= (tmp_val >> 8);
        tmp_val ^= (tmp_val >> 4);
        tmp_val &= 0xF;
        tmp_val = ((0x6996 >> tmp_val) & 1);

        x2 |= (tmp_val << (31 - i));
    }

    for (i = 0; i < init_offset; i++) {
        x1 >>= 1;
        x1 |= (((0x55AA >> (x1 & 0xF)) & 1) << 31);
        x2 >>= 1;
        x2 |= (((0x6996 >> (x2 & 0xF)) & 1) << 31);
    }

    tmp_val = 0;
    for (i = 0; i < seq_len; i++) {
        /* store from MSB to LSB */
        tmp_val |= (((x1 ^ x2) & 0x80000000) >> (i % 32));
        if (((i + 1) % 32) == 0) {
            *pseq_out++ = tmp_val;
            tmp_val    = 0;
        }

        x1 >>= 1;
        x1 |= (((0x55AA >> (x1 & 0xF)) & 1) << 31);
        x2 >>= 1;
        x2 |= (((0x6996 >> (x2 & 0xF)) & 1) << 31);
    }

    if (seq_len && (seq_len % 32))
        *pseq_out = tmp_val;
}

__device__ void kernel_fft_radix2(cufftDoubleComplex *c_io, int N)
{
    int n, s, l, i;
    cufftDoubleComplex *d_tw = &d_tw128[0];

    for (n = N >> 1, l = 1; n >= 1; n >>= 1, l <<= 1) {
        for (i = 0; i < l; i++) {
            for (s = 0; s < n; s++) {
                cufftDoubleComplex a, b, aa, bb, tw;

                a = c_io[s + n*0 + i*n*2];
                b = c_io[s + n*1 + i*n*2];
                tw = d_tw[s * l];

                aa.x = a.x + b.x;
                aa.y = a.y + b.y;

                bb.x = (a.x - b.x) * tw.x - (a.y - b.y) * tw.y;
                bb.y = (a.y - b.y) * tw.x + (a.x - b.x) * tw.y;

                c_io[s + n*0 + i*n*2] = aa;
                c_io[s + n*1 + i*n*2] = bb;
            }
        }
    }

    // bit reverse
    for (n = 0; n < N; n++) {
        cufftDoubleComplex c;
        int idx = d_radix2_bitreverse[n];

        if (idx <= n)
            continue;

        c = c_io[idx];
        c_io[idx] = c_io[n];
        c_io[n] = c;
    }
}



/*
 *
 */
__global__ void extract_tfg_multiblocks_kernel(cufftDoubleComplex *d_capbuf, cufftDoubleComplex *d_tfg, double *d_tfg_timestamp, double *d_adjust_f,
                                               unsigned short n_id_cell, int n_symb_dl, double frame_start,
                                               double fc_requested, double fc_programmed, double fs_programmed, double freq)
{
    __shared__ cufftDoubleComplex s_capbuf[128];

    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;

    int dft_location_i;
    double freq_fine = freq + *d_adjust_f;
    const double k_factor = (fc_requested - freq_fine) / fc_programmed;
    double dft_location = frame_start + ((n_symb_dl == 6) ? 32 : 10) * 16 / FS_LTE * fs_programmed * k_factor;
    cufftDoubleComplex shift, coeff;

    if (dft_location - .01 * fs_programmed * k_factor > -0.5) {
        dft_location = dft_location - .01 * fs_programmed * k_factor;
    }

    dft_location += ((bid / n_symb_dl) * 960 + (bid % n_symb_dl) * (n_symb_dl == 6 ? 160 : 137))  * 16 / FS_LTE * fs_programmed * k_factor;
    dft_location_i = lround(dft_location);

    // cvec capbuf = fshift(capbuf_raw, -freq_fine, fs_programmed * k_factor);
    double k = CUDART_PI * (-freq_fine) / (fs_programmed * k_factor / 2);
    double theta = k * (dft_location_i + tid);
    shift.x = cos(theta);
    shift.y = sin(theta);

    s_capbuf[tid].x = COMPLEX_MUL_REAL(d_capbuf[dft_location_i + tid], shift);
    s_capbuf[tid].y = COMPLEX_MUL_IMAG(d_capbuf[dft_location_i + tid], shift);

    __syncthreads();

    // DFT of 128 points
    // cvec dft_out = dft(capbuf.mid(round_i(dft_location), 128));

    if (tid == 0) {
        d_tfg_timestamp[bid] = dft_location;

        kernel_fft_radix2(s_capbuf, 128);
    }

    __syncthreads();

    //  92,  93,  94, ... , 127,  1,  2,  3, ..., 36 -> concat(dft_out.right(36), dft_out.mid(1, 36))
    //   0,   1,   2,     ,  35, 36, 37, 38,    , 71
    // -36, -35, -34, ... ,  -1,  1,  2,  3, .... 36

    // concat(dft_out.right(36), dft_out.mid(1,36));
    // exp((-J * 2 * pi * late / 128) * cn)

    double late = dft_location_i - dft_location;

    if (tid < 36) {
        coeff.x =  cos(2 * CUDART_PI * late * (36 - tid) / 128);
        coeff.y =  sin(2 * CUDART_PI * late * (36 - tid) / 128);

        d_tfg[bid * 72 + tid].x = SQRT128_INV * COMPLEX_MUL_REAL(s_capbuf[tid + 92], coeff);
        d_tfg[bid * 72 + tid].y = SQRT128_INV * COMPLEX_MUL_IMAG(s_capbuf[tid + 92], coeff);
    } else if (tid < 72) {
        coeff.x =  cos(2 * CUDART_PI * late * (tid - 35) / 128);
        coeff.y = -sin(2 * CUDART_PI * late * (tid - 35) / 128);

        d_tfg[bid * 72 + tid].x = SQRT128_INV * COMPLEX_MUL_REAL(s_capbuf[tid - 35], coeff);
        d_tfg[bid * 72 + tid].y = SQRT128_INV * COMPLEX_MUL_IMAG(s_capbuf[tid - 35], coeff);
    }
}



/*
 *
 */
__global__ void extract_tfg_singleblock_kernel(cufftDoubleComplex *d_capbuf, cufftDoubleComplex *d_tfg, double *d_tfg_timestamp, double *d_adjust_f,
                                               unsigned short n_id_cell, int n_symb_dl, double frame_start,
                                               double fc_requested, double fc_programmed, double fs_programmed, double freq)
{
    const unsigned int tid = threadIdx.x;

    cufftDoubleComplex s_capbuf[128];
    int dft_location_i;
    double freq_fine = freq + *d_adjust_f;
    const double k_factor = (fc_requested - freq_fine) / fc_programmed;
    double dft_location = frame_start + ((n_symb_dl == 6) ? 32 : 10) * 16 / FS_LTE * fs_programmed * k_factor;

    if (dft_location - .01 * fs_programmed * k_factor > -0.5) {
        dft_location = dft_location - .01 * fs_programmed * k_factor;
    }

    dft_location += ((tid / n_symb_dl) * 960 + (tid % n_symb_dl) * (n_symb_dl == 6 ? 160 : 137))  * 16 / FS_LTE * fs_programmed * k_factor;
    dft_location_i = lround(dft_location);
    d_tfg_timestamp[tid] = dft_location;

    // cvec capbuf = fshift(capbuf_raw, -freq_fine, fs_programmed * k_factor);
    double k = CUDART_PI * (-freq_fine) / (fs_programmed * k_factor / 2);

    for (unsigned int t = dft_location_i, i = 0; i < 128; i++, t++) {
        cufftDoubleComplex shift;

        shift.x = cos(k * t);
        shift.y = sin(k * t);

        s_capbuf[i].x = COMPLEX_MUL_REAL(d_capbuf[t], shift);
        s_capbuf[i].y = COMPLEX_MUL_IMAG(d_capbuf[t], shift);
    }

    // DFT of 128 points
    // cvec dft_out = dft(capbuf.mid(round_i(dft_location), 128));

    kernel_fft_radix2(s_capbuf, 128);


    //  92,  93,  94, ... , 127,  1,  2,  3, ..., 36 -> concat(dft_out.right(36), dft_out.mid(1, 36))
    //   0,   1,   2,     ,  35, 36, 37, 38,    , 71
    // -36, -35, -34, ... ,  -1,  1,  2,  3, .... 36

    // concat(dft_out.right(36), dft_out.mid(1,36));
    // exp((-J * 2 * pi * late / 128) * cn)

    double late = dft_location_i - dft_location;

    for (unsigned int i = 1; i <= 36; i++) {
        cufftDoubleComplex coeff;
        coeff.x =  cos(2 * CUDART_PI * late * i / 128);
        coeff.y = -sin(2 * CUDART_PI * late * i / 128);

        d_tfg[tid * 72 + 35 + i].x = SQRT128_INV * COMPLEX_MUL_REAL(s_capbuf[i], coeff);
        d_tfg[tid * 72 + 35 + i].y = SQRT128_INV * COMPLEX_MUL_IMAG(s_capbuf[i], coeff);

        coeff.y = -coeff.y;

        d_tfg[tid * 72 + 36 - i].x = SQRT128_INV * COMPLEX_MUL_REAL(s_capbuf[128 - i], coeff);
        d_tfg[tid * 72 + 36 - i].y = SQRT128_INV * COMPLEX_MUL_IMAG(s_capbuf[128 - i], coeff);
    }
}



/*
 *
 */
__global__ void tfoec_kernel(cufftDoubleComplex *d_tfg, cufftDoubleComplex *d_rs_extracted, double *d_tfg_timestamp,
                             unsigned short n_id_cell, int n_symb_dl,
                             double fc_requested, double fc_programmed, double fs_programmed,
                             // output
                             double *d_residual_f)
{
    __shared__ unsigned int rs_dl[20 * 2];
    __shared__ float foe_real, foe_imag;
    __shared__ float toe_real, toe_imag;


    const unsigned int tid = threadIdx.x;
    double dft_location;
    double late;

    // generate random sequences for symbol 0, 1, 3/4 of 20 slots

    if (tid < 20 * 2) {
        int slot = tid / 2;
        int l = (tid & 1) * (n_symb_dl - 3);
        int cinit = ((7 * (slot + 1) + l + 1) * (2 * n_id_cell + 1) << 10) + 2 * n_id_cell + (n_symb_dl == 7 ? 1 : 0);

        pn_seq_lsb_to_msb(cinit, 6 * 2 * 2, (55 - 3) * 2 * 2, &rs_dl[tid]);
    }

    if (tid == 0) {
        foe_real = 0.0; foe_imag = 0.0;
        toe_real = 0.0; toe_imag = 0.0;
    }

    dft_location = d_tfg_timestamp[tid];

    __syncthreads();

    if (tid < 122 * 2) {
        int slot = tid / 2;
        int second_sym = tid & 1;
        int l = second_sym * (n_symb_dl - 3);
        int rs_bits = rs_dl[(slot % 20) * 2 + second_sym];
        int v_offset = ((n_id_cell % 6) + second_sym * 3) % 6;

        // elem_mult(rs_extracted.get_row(t), conj(rs_dl.get_rs(mod(t, 20), sym_num)))
        for (unsigned int i = 0; i < 12; i++, rs_bits >>= 2, v_offset += 6) {
            cufftDoubleComplex std_rs, rcvd_rs;

            // rs_symb = 1 / sqrt(2) ((1 - 2 * c(0)) + j (1 - 2 * c(1))

            std_rs.x = SQRT2_INV * (1.0 - ((rs_bits & 1) * 2));
            std_rs.y = SQRT2_INV * (1.0 - (rs_bits & 2));

            rcvd_rs = d_tfg[(slot * n_symb_dl + l) * 72 + v_offset];

            std_rs.y = -std_rs.y;

            d_rs_extracted[((tid & 1) * 122 + (tid / 2)) * 12 + i].x = COMPLEX_MUL_REAL(rcvd_rs, std_rs);
            d_rs_extracted[((tid & 1) * 122 + (tid / 2)) * 12 + i].y = COMPLEX_MUL_IMAG(rcvd_rs, std_rs);
        }
    }

    __syncthreads();

    if (tid < 121) {
        // CUDA 2.0+ capability support atomic addition of 32 bit floating point numbers
        // CUDA 6.0+ capability support atomic addition of 64 bit floating point numbers

        // sum(elem_mult(conj(col(0,n_slot-2)), col(1,-1)));

        float real = 0.0, imag = 0.0;
        for (unsigned int i = 0; i < 12; i++) {
            cufftDoubleComplex rs_1, rs_2;

            rs_1 = d_rs_extracted[(0 + tid + 0) * 12 + i];
            rs_2 = d_rs_extracted[(0 + tid + 1) * 12 + i];

            rs_1.y = -rs_1.y;
            real += COMPLEX_MUL_REAL(rs_1, rs_2);
            imag += COMPLEX_MUL_IMAG(rs_1, rs_2);

            rs_1 = d_rs_extracted[(122 + tid + 0) * 12 + i];
            rs_2 = d_rs_extracted[(122 + tid + 1) * 12 + i];

            rs_1.y = -rs_1.y;
            real += COMPLEX_MUL_REAL(rs_1, rs_2);
            imag += COMPLEX_MUL_IMAG(rs_1, rs_2);
        }

        atomicAdd(&foe_real, real);
        atomicAdd(&foe_imag, imag);
    }

    __syncthreads();

    double residual_f = angle(foe_real, foe_imag) / (2*CUDART_PI) / 0.0005;
    double k_factor_residual = (fc_requested - residual_f) / fc_programmed;
    late = dft_location - k_factor_residual * dft_location;

    if (tid == 0) *d_residual_f = residual_f;

    // -36, -35, -34, ... ,  -1,  1,  2,  3, .... 36
    // exp((-J * 2 * pi * late / 128) * cn)

    // tfg.get_row(t)*exp(J*2*pi* -residual_f*tfg_comp_timestamp(t)/(FS_LTE/16))
    // elem_mult(tfg_comp.get_row(t), exp((-J*2*pi*late/128)*cn))
    for (unsigned int i = 1; i <= 36; i++) {

        cufftDoubleComplex coeff;
        double real, imag;

        coeff.x = cos(2 * CUDART_PI * ((-residual_f) * dft_location / (FS_LTE / 16) - (late * i / 128)));
        coeff.y = sin(2 * CUDART_PI * ((-residual_f) * dft_location / (FS_LTE / 16) - (late * i / 128)));

        real = COMPLEX_MUL_REAL(d_tfg[tid * 72 + 35 + i], coeff);
        imag = COMPLEX_MUL_IMAG(d_tfg[tid * 72 + 35 + i], coeff);

        d_tfg[tid * 72 + 35 + i].x = real;
        d_tfg[tid * 72 + 35 + i].y = imag;

        coeff.x = cos(2 * CUDART_PI * ((-residual_f) * dft_location / (FS_LTE / 16) + (late * i / 128)));
        coeff.y = sin(2 * CUDART_PI * ((-residual_f) * dft_location / (FS_LTE / 16) + (late * i / 128)));

        real = COMPLEX_MUL_REAL(d_tfg[tid * 72 + 36 - i], coeff);
        imag = COMPLEX_MUL_IMAG(d_tfg[tid * 72 + 36 - i], coeff);

        d_tfg[tid * 72 + 36 - i].x = real;
        d_tfg[tid * 72 + 36 - i].y = imag;
    }

    __syncthreads();

    // Perform TOE.
    // Implemented by comparing subcarrier k of one OFDM symbol with subcarrier
    // k+3 of another OFDM symbol. This is why FOE must be performed first.
    // Slightly less performance but faster execution time could be obtained
    // by comparing subcarrier k with subcarrier k+6 of the same OFDM symbol.

    if (tid < 2 * 122 - 1) {

        int slot1 = tid / 2;
        int second_sym1 = tid & 1;
        int l1 = second_sym1 * (n_symb_dl - 3);
        int rs_bits1 = rs_dl[(slot1 % 20) * 2 + second_sym1];
        int v_offset1 = ((n_id_cell % 6) + second_sym1 * 3) % 6;

        int slot2 = (tid + 1) / 2;
        int second_sym2 = (tid + 1) & 1;
        int l2 = second_sym2 * (n_symb_dl - 3);
        int rs_bits2 = rs_dl[(slot2 % 20) * 2 + second_sym2];
        int v_offset2 = ((n_id_cell % 6) + second_sym2 * 3) % 6;

        float real, imag;

        cufftDoubleComplex toe1, toe2;
        cufftDoubleComplex std_rs, rcvd_rs;
        cufftDoubleComplex r1v, r2v, r2v_prev;

        toe1.x = 0.0; toe1.y = 0.0;
        toe2.x = 0.0; toe2.y = 0.0;
        r2v_prev.x = 0.0; r2v_prev.y = 0.0;

        if (v_offset2 < v_offset1) {
            SWAP(slot1, slot2);
            SWAP(l1, l2);
            SWAP(rs_bits1, rs_bits2);
            SWAP(v_offset1, v_offset2);
        }

        for (unsigned int i = 0; i < 12; i++, rs_bits1 >>= 2, v_offset1 += 6, rs_bits2 >>= 2, v_offset2 += 6) {

            // rs_symb = 1 / sqrt(2) ((1 - 2 * c(0)) + j (1 - 2 * c(1))

            std_rs.x = SQRT2_INV * (1.0 - ((rs_bits1 & 1) * 2));
            std_rs.y = SQRT2_INV * (1.0 - ((rs_bits1 & 2)));

            rcvd_rs = d_tfg[(slot1 * n_symb_dl + l1) * 72 + v_offset1];

            std_rs.y = -std_rs.y;

            r1v.x = COMPLEX_MUL_REAL(rcvd_rs, std_rs);
            r1v.y = -COMPLEX_MUL_IMAG(rcvd_rs, std_rs); // this r1v is actually conj(r1v)

            std_rs.x = SQRT2_INV * (1.0 - ((rs_bits2 & 1) * 2));
            std_rs.y = SQRT2_INV * (1.0 - ((rs_bits2 & 2)));

            std_rs.y = -std_rs.y;

            rcvd_rs = d_tfg[(slot2 * n_symb_dl + l2) * 72 + v_offset2];

            r2v.x = COMPLEX_MUL_REAL(rcvd_rs, std_rs);
            r2v.y = COMPLEX_MUL_IMAG(rcvd_rs, std_rs);

            // elem_mult(conj(r1v), r2v)

            toe1.x += COMPLEX_MUL_REAL(r1v, r2v);
            toe1.y += COMPLEX_MUL_IMAG(r1v, r2v);

            r1v.y = -r1v.y;
            r2v.y = -r2v.y;   // this r2v is actually conj(r2v)

            // elem_mult(conj(r2v(i-1)), r1v(i))

            toe2.x += COMPLEX_MUL_REAL(r1v, r2v_prev);
            toe2.y += COMPLEX_MUL_IMAG(r1v, r2v_prev);

            r2v_prev = r2v;
        }

        real = 1.0 * (toe1.x + toe2.x);
        imag = 1.0 * (toe1.y + toe2.y);
        atomicAdd(&toe_real, real);
        atomicAdd(&toe_imag, imag);
    }

    __syncthreads();

    // double delay = -arg(toe)/3/(2*pi/128);
    double delay = -angle(toe_real, toe_imag) / 3 / (2 * CUDART_PI / 128);

    // Perform TOC
    for (unsigned int i = 1; i <= 36; i++) {

        cufftDoubleComplex coeff;
        double real, imag;

        coeff.x = cos(2 * CUDART_PI * delay * i / 128);
        coeff.y = sin(2 * CUDART_PI * delay * i / 128);

        real = COMPLEX_MUL_REAL(d_tfg[tid * 72 + 35 + i], coeff);
        imag = COMPLEX_MUL_IMAG(d_tfg[tid * 72 + 35 + i], coeff);

        d_tfg[tid * 72 + 35 + i].x = real;
        d_tfg[tid * 72 + 35 + i].y = imag;

        coeff.y = -coeff.y;

        real = COMPLEX_MUL_REAL(d_tfg[tid * 72 + 36 - i], coeff);
        imag = COMPLEX_MUL_IMAG(d_tfg[tid * 72 + 36 - i], coeff);

        d_tfg[tid * 72 + 36 - i].x = real;
        d_tfg[tid * 72 + 36 - i].y = imag;
    }
}



/*
 *
 */
__global__ void chan_est_kernel(cufftDoubleComplex *d_tfg, int num_slot, int ant_port,
                                unsigned short n_id_cell, int n_symb_dl,
                                // output
                                cufftDoubleComplex *d_ce_filt, double *d_err_pwr_acc)
{
    __shared__ unsigned int rs_dl[3];
    __shared__ cufftDoubleComplex rcvd_rs[12 * 3], tfg_rs, conj_std_rs;
    __shared__ float ce_err_pwr;

    const unsigned int bid = blockIdx.x;
    const unsigned int tid = threadIdx.x;

    /*  */
    int rs_no = bid;
    int port = ant_port;
    int rs_out = bid;

    /*  */

    const int port01 = 1 - (port / 2), port23 = port / 2;
    const int rs_per_slot = 1 << port01;
    const int total_rs = num_slot * rs_per_slot;
    const int rs_slot = (rs_no / rs_per_slot) % 20;
    // const int l_rs = port01 * ((rs_no & 1) * (n_symb_dl - 3)) + port23;
    const int v = port01 * ((port & 1) ^ (rs_no & 1)) * 3 + port23 * ((port & 1) + (rs_slot & 1)) * 3;
    const int k_offset = (v + (n_id_cell % 6)) % 6;

    int pos, rs_count;
    cufftDoubleComplex filt;


    if (tid < 3) {
        const int cinit_rs_no = rs_no - 1 + tid;
        const int cinit_rs_slot = (cinit_rs_no / rs_per_slot) % 20;
        const int cinit_l_rs = port01 * ((cinit_rs_no & 1) * (n_symb_dl - 3)) + port23;
        const int cinit = ((7 * (cinit_rs_slot + 1) + cinit_l_rs + 1) * (2 * n_id_cell + 1) << 10) + 2 * n_id_cell + (n_symb_dl == 7 ? 1 : 0);

        pn_seq_lsb_to_msb(cinit, 6 * 2 * 2, (55 - 3) * 2 * 2, &rs_dl[tid]);

        ce_err_pwr = 0.0;
    }

    __syncthreads();

    if (tid < 36) {
        const int copy_rs_no = rs_no - 1 + (tid / 12);
        const int copy_rs_slot = copy_rs_no / rs_per_slot;
        const int copy_l_rs = port01 * ((copy_rs_no & 1) * (n_symb_dl - 3)) + port23;
        const int copy_v = port01 * ((port & 1) ^ (copy_rs_no & 1)) * 3 + port23 * ((port & 1) + (copy_rs_slot & 1)) * 3;
        const int copy_k_offset = (copy_v + (n_id_cell % 6)) % 6;

        const unsigned int rs_bits = (rs_dl[tid / 12] >> (2 * (tid % 12))) & 0x3;
        const int copy_pos = (tid / 12) * 12 + (tid % 12);


        if ((0 <= copy_rs_no) && (copy_rs_no < total_rs)) {
            tfg_rs.x = d_tfg[(copy_rs_slot * n_symb_dl + copy_l_rs) * 72 + copy_k_offset + 6 * (tid % 12)].x;
            tfg_rs.y = d_tfg[(copy_rs_slot * n_symb_dl + copy_l_rs) * 72 + copy_k_offset + 6 * (tid % 12)].y;

            conj_std_rs.x = SQRT2_INV * (1.0 - ((rs_bits & 0x1) * 2));
            conj_std_rs.y = - SQRT2_INV * (1.0 - (rs_bits & 0x2));

            rcvd_rs[copy_pos].x = COMPLEX_MUL_REAL(tfg_rs, conj_std_rs);
            rcvd_rs[copy_pos].y = COMPLEX_MUL_IMAG(tfg_rs, conj_std_rs);
        } else {
            rcvd_rs[copy_pos].x = 0.0;
            rcvd_rs[copy_pos].y = 0.0;
        }
    }

    __syncthreads();

    //  0   1   2   3                   9   10    11
    //    0   1   2   3                   9    10    11
    //  0   1   2   3                   9   10    11
    //    0   1   2   3                   9    10    11
    //  0   1   2   3                   9   10    11
    //
    //  k_offset < 3
    //
    //  (i,j=0)    ->                           |(i-1,j)(i,j)(i+1,j)|(i,j+1)
    //  (i,j=1-10) -> (i-1,j-1)(i+1,j-1)|(i,j-1)|(i-1,j)(i,j)(i+1,j)|(i,j+1)
    //  (i,j=11)   -> (i-1,j-1)(i+1,j-1)|(i,j-1)|(i-1,j)(i,j)(i+1,j)|
    //
    //  k_offset >= 3`
    //
    //  (i,j=0)    ->        |(i-1,j)(i,j)(i+1,j)|(i,j+1)|(i-1,j+1)(i+1,j+1)
    //  (i,j=1-10) -> (i,j-1)|(i-1,j)(i,j)(i+1,j)|(i,j+1)|(i-1,j+1)(i+1,j+1)
    //  (i,j=11)   -> (i,j-1)|(i-1,j)(i,j)(i+1,j)|
    //

    if (tid < 12) {
        pos = 12 + tid;

        filt.x = rcvd_rs[pos].x + rcvd_rs[pos - 12].x + rcvd_rs[pos + 12].x;
        filt.y = rcvd_rs[pos].y + rcvd_rs[pos - 12].y + rcvd_rs[pos + 12].y;
        rs_count = 3;
        if (0 < tid) {
            filt.x += rcvd_rs[pos - 1].x;
            filt.y += rcvd_rs[pos - 1].y;
            rs_count += 1;
            if (k_offset < 3) {
                filt.x += (rcvd_rs[pos - 12 - 1].x + rcvd_rs[pos + 12 - 1].x);
                filt.y += (rcvd_rs[pos - 12 - 1].y + rcvd_rs[pos + 12 - 1].y);
                rs_count += 2;
            }
        }
        if (tid < 11) {
            filt.x += rcvd_rs[pos + 1].x;
            filt.y += rcvd_rs[pos + 1].y;
            rs_count += 1;
            if (3 <= k_offset) {
                filt.x += (rcvd_rs[pos - 12 + 1].x + rcvd_rs[pos + 12 + 1].x);
                filt.y += (rcvd_rs[pos - 12 + 1].y + rcvd_rs[pos + 12 + 1].y);
                rs_count += 2;
            }
        }
        if ((rs_no == 0) || (rs_no == total_rs - 1)) {
            if (((k_offset < 3) && (tid == 0)) || ((k_offset >= 3) && (tid == 11))) {
                rs_count -= 1;
            } else {
                rs_count -= 2;
            }
        }
        filt.x /= rs_count;
        filt.y /= rs_count;

        d_ce_filt[rs_out * 12 + tid].x = filt.x;
        d_ce_filt[rs_out * 12 + tid].y = filt.y;

        double error_r = rcvd_rs[pos].x - filt.x;
        double error_i = rcvd_rs[pos].y - filt.y;

        atomicAdd(&ce_err_pwr, (float)(error_r * error_r + error_i * error_i));
    }

    __syncthreads();

    if (tid == 0) {
        d_err_pwr_acc[bid] = ce_err_pwr;
    }
}




/*
 *
 */
__global__ void chan_est_four_port_step1_kernel(cufftDoubleComplex *d_tfg, int num_slot,
                                                unsigned short n_id_cell, int n_symb_dl,
                                                // output
                                                cufftDoubleComplex *d_ce_filt, double *d_err_pwr_acc)
{
    __shared__ unsigned int rs_dl[3];
    __shared__ cufftDoubleComplex rcvd_rs[12 * 3], tfg_rs, conj_std_rs;
    __shared__ float ce_err_pwr;

    const unsigned int bid = blockIdx.x;
    const unsigned int tid = threadIdx.x;

    /* port 0 and port 1 has (2 * num_slot) symbols, port 3 and port 4 has (num_slot) symbols */

    int rs_no = (bid < (4 * num_slot)) ? (bid % (2 * num_slot)) : (bid % num_slot);
    int port = (bid < (4 * num_slot)) ? (bid / (num_slot * 2)) : ((bid / num_slot) - 2);
    int rs_out = bid;

    /*  */

    const int port01 = 1 - (port / 2), port23 = port / 2;
    const int rs_per_slot = 1 << port01;
    const int total_rs = num_slot * rs_per_slot;
    const int rs_slot = (rs_no / rs_per_slot) % 20;
    const int l_rs = port01 * ((rs_no & 1) * (n_symb_dl - 3)) + port23;
    const int v = port01 * ((port & 1) ^ (rs_no & 1)) * 3 + port23 * ((port & 1) + (rs_slot & 1)) * 3;
    const int k_offset = (v + (n_id_cell % 6)) % 6;

    int pos, rs_count;
    cufftDoubleComplex filt;


    if (tid < 3) {
        const int cinit_rs_no = rs_no - 1 + tid;
        const int cinit_rs_slot = (cinit_rs_no / rs_per_slot) % 20;
        const int cinit_l_rs = port01 * ((cinit_rs_no & 1) * (n_symb_dl - 3)) + port23;
        const int cinit = ((7 * (cinit_rs_slot + 1) + cinit_l_rs + 1) * (2 * n_id_cell + 1) << 10) + 2 * n_id_cell + (n_symb_dl == 7 ? 1 : 0);

        pn_seq_lsb_to_msb(cinit, 6 * 2 * 2, (55 - 3) * 2 * 2, &rs_dl[tid]);

        ce_err_pwr = 0.0;
    }

    __syncthreads();

    if (tid < 36) {
        const int copy_rs_no = rs_no - 1 + (tid / 12);
        const int copy_rs_slot = copy_rs_no / rs_per_slot;
        const int copy_l_rs = port01 * ((copy_rs_no & 1) * (n_symb_dl - 3)) + port23;
        const int copy_v = port01 * ((port & 1) ^ (copy_rs_no & 1)) * 3 + port23 * ((port & 1) + (copy_rs_slot & 1)) * 3;
        const int copy_k_offset = (copy_v + (n_id_cell % 6)) % 6;

        const unsigned int rs_bits = (rs_dl[tid / 12] >> (2 * (tid % 12))) & 0x3;
        const int copy_pos = (tid / 12) * 12 + (tid % 12);


        if ((0 <= copy_rs_no) && (copy_rs_no < total_rs)) {
            tfg_rs.x = d_tfg[(copy_rs_slot * n_symb_dl + copy_l_rs) * 72 + copy_k_offset + 6 * (tid % 12)].x;
            tfg_rs.y = d_tfg[(copy_rs_slot * n_symb_dl + copy_l_rs) * 72 + copy_k_offset + 6 * (tid % 12)].y;

            conj_std_rs.x = SQRT2_INV * (1.0 - ((rs_bits & 0x1) * 2));
            conj_std_rs.y = - SQRT2_INV * (1.0 - (rs_bits & 0x2));

            rcvd_rs[copy_pos].x = COMPLEX_MUL_REAL(tfg_rs, conj_std_rs);
            rcvd_rs[copy_pos].y = COMPLEX_MUL_IMAG(tfg_rs, conj_std_rs);
        } else {
            rcvd_rs[copy_pos].x = 0.0;
            rcvd_rs[copy_pos].y = 0.0;
        }
    }

    __syncthreads();

    //  0   1   2   3                   9   10    11
    //    0   1   2   3                   9    10    11
    //  0   1   2   3                   9   10    11
    //    0   1   2   3                   9    10    11
    //  0   1   2   3                   9   10    11
    //
    //  k_offset < 3
    //
    //  (i,j=0)    ->                           |(i-1,j)(i,j)(i+1,j)|(i,j+1)
    //  (i,j=1-10) -> (i-1,j-1)(i+1,j-1)|(i,j-1)|(i-1,j)(i,j)(i+1,j)|(i,j+1)
    //  (i,j=11)   -> (i-1,j-1)(i+1,j-1)|(i,j-1)|(i-1,j)(i,j)(i+1,j)|
    //
    //  k_offset >= 3`
    //
    //  (i,j=0)    ->        |(i-1,j)(i,j)(i+1,j)|(i,j+1)|(i-1,j+1)(i+1,j+1)
    //  (i,j=1-10) -> (i,j-1)|(i-1,j)(i,j)(i+1,j)|(i,j+1)|(i-1,j+1)(i+1,j+1)
    //  (i,j=11)   -> (i,j-1)|(i-1,j)(i,j)(i+1,j)|
    //

    if (tid < 12) {
        pos = 12 + tid;

        filt.x = rcvd_rs[pos].x + rcvd_rs[pos - 12].x + rcvd_rs[pos + 12].x;
        filt.y = rcvd_rs[pos].y + rcvd_rs[pos - 12].y + rcvd_rs[pos + 12].y;
        rs_count = 3;
        if (0 < tid) {
            filt.x += rcvd_rs[pos - 1].x;
            filt.y += rcvd_rs[pos - 1].y;
            rs_count += 1;
            if (k_offset < 3) {
                filt.x += (rcvd_rs[pos - 12 - 1].x + rcvd_rs[pos + 12 - 1].x);
                filt.y += (rcvd_rs[pos - 12 - 1].y + rcvd_rs[pos + 12 - 1].y);
                rs_count += 2;
            }
        }
        if (tid < 11) {
            filt.x += rcvd_rs[pos + 1].x;
            filt.y += rcvd_rs[pos + 1].y;
            rs_count += 1;
            if (3 <= k_offset) {
                filt.x += (rcvd_rs[pos - 12 + 1].x + rcvd_rs[pos + 12 + 1].x);
                filt.y += (rcvd_rs[pos - 12 + 1].y + rcvd_rs[pos + 12 + 1].y);
                rs_count += 2;
            }
        }
        if ((rs_no == 0) || (rs_no == total_rs - 1)) {
            if (((k_offset < 3) && (tid == 0)) || ((k_offset >= 3) && (tid == 11))) {
                rs_count -= 1;
            } else {
                rs_count -= 2;
            }
        }

        filt.x /= rs_count;
        filt.y /= rs_count;

        d_ce_filt[rs_out * 12 + tid].x = filt.x;
        d_ce_filt[rs_out * 12 + tid].y = filt.y;

        double error_r = rcvd_rs[pos].x - filt.x;
        double error_i = rcvd_rs[pos].y - filt.y;

        atomicAdd(&ce_err_pwr, (float)(error_r * error_r + error_i * error_i));
    }

    __syncthreads();

    if (tid == 0) {
        d_err_pwr_acc[bid] = ce_err_pwr;
    }
}




/*
 *
 */
__global__ void chan_est_four_port_step2_kernel(double *d_err_pwr_acc, int num_slot,
                                                // output
                                                double *d_np)
{
    extern __shared__ double err_pwr_acc[];

    const unsigned int bid = blockIdx.x;
    const unsigned int tid = threadIdx.x;

    const int port = bid;
    const int no_sym = port < 2 ? 2 * num_slot : num_slot;
    const int rs_no = port < 2 ? port * 2 : 4 + (port - 2);
    double *d_err_pwr_acc_start = &d_err_pwr_acc[rs_no * num_slot];

    if (tid < no_sym) {
        err_pwr_acc[tid] = d_err_pwr_acc_start[tid];
    }

    __syncthreads();

    for (unsigned int s = 128; s > 0; s >>= 1) {
        if ((tid < s) && (tid + s < no_sym)) {
            err_pwr_acc[tid] += err_pwr_acc[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_np[bid] = err_pwr_acc[0] / (no_sym * 12);
    }
}



/*
 *  for debug
 */
__global__ void print_kernel_float(float *d_data, int len)
{
    printf("\n\nprint_kernel_float\n");
    for (int i = 0; i < len;i++)
        printf("(%f)", d_data[i]);
    printf("\n\n");
}


__global__ void print_kernel_double(double *d_data, int len)
{
    printf("\n\nprint_kernel_double\n");
    for (int i = 0; i < len;i++)
        printf("(%f)",d_data[i]);
    printf("\n\n");
}

__global__ void print_kernel_complex(cufftDoubleComplex *d_data, int len)
{
    printf("\n\nprint_kernel_complex\n");
    for (int i = 0; i < len;i++)
        printf("(%lf,%lf)", d_data[i].x, d_data[i].y);
    printf("\n\n");
}


/*
 *
 */
extern "C" Cell extract_tfg_and_tfoec(
    const Cell & cell,
    const cvec & capbuf_raw,
    const double & fc_requested,
    const double & fc_programmed,
    const double & fs_programmed,
    // Output
    cmat & my_tfg_comp)
{
    const unsigned int n_cap = capbuf_raw.length();

    Cell cell_out(cell);

    cufftDoubleComplex *h_capbuf = (cufftDoubleComplex *)NULL, *d_capbuf = (cufftDoubleComplex *)NULL;
    cufftDoubleComplex *h_tfg = (cufftDoubleComplex *)NULL, *d_tfg = (cufftDoubleComplex *)NULL;
    cufftDoubleComplex *d_rs_extracted = (cufftDoubleComplex *)NULL;
    double h_frame_start;
    int h_n_id_1_est, h_cp_type;
    double h_residual_f, *d_residual_f = (double *)NULL;
    double h_adjust_f, *d_adjust_f = (double *)NULL;
    double *d_tfg_timestamp = (double *)NULL;
    float *d_sss_h12_np_est = (float *)NULL;
    cufftDoubleComplex *d_sss_h12_est = (cufftDoubleComplex *)NULL;
    double *d_log_lik = (double *)NULL, *d_frame_start = (double *)NULL;
    int *d_n_id_1_est = (int *)NULL, *d_cp_type = (int *)NULL;

    cufftDoubleComplex *d_h_sm = (cufftDoubleComplex *)NULL;
    cufftDoubleComplex *d_sss_raw = (cufftDoubleComplex *)NULL;
    double *d_pss_np_inv = (double *)NULL;

    cufftDoubleComplex *d_M = (cufftDoubleComplex *)NULL;
    double *d_err_pwr_acc = (double *)NULL;

    checkCudaErrors(cudaMalloc(&d_rs_extracted, (2 + 2 + 1 + 1) * 122 * 12 * sizeof(cufftDoubleComplex)));
    checkCudaErrors(cudaMalloc(&d_adjust_f, sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_residual_f, sizeof(double)));

    checkCudaErrors(cudaMalloc(&d_n_id_1_est, sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_cp_type, sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_frame_start, sizeof(double)));

    checkCudaErrors(cudaMalloc(&d_err_pwr_acc, (2 + 2 + 1 + 1) * 122  * sizeof(double)));

    h_capbuf = (cufftDoubleComplex *)malloc(n_cap * sizeof(cufftDoubleComplex));

    for (unsigned int i = 0; i < n_cap; i++) {
        h_capbuf[i].x = capbuf_raw[i].real();
        h_capbuf[i].y = capbuf_raw[i].imag();
    }
    checkCudaErrors(cudaMalloc(&d_capbuf, n_cap * sizeof(cufftDoubleComplex)));
    checkCudaErrors(cudaMemcpy(d_capbuf, h_capbuf, n_cap * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice));

    /* prologue of function sss_detect() of searcher.cpp */

    const double peak_freq = cell_out.freq;
    double k_factor = (fc_requested - peak_freq) / fc_programmed; 
    double peak_loc = (cell.ind + 9 < 162 ? cell_out.ind + 9600 * k_factor : cell_out.ind);
    const unsigned int n_id_2_est = cell_out.n_id_2;
    const int n_pss = ceil((n_cap - 125 - 9 - peak_loc) / (k_factor * 9600));

    checkCudaErrors(cudaMalloc(&d_h_sm, 62 * n_pss * sizeof(cufftDoubleComplex)));
    checkCudaErrors(cudaMalloc(&d_sss_raw, n_pss * 62 * 2 * sizeof(cufftDoubleComplex)));
    checkCudaErrors(cudaMalloc(&d_pss_np_inv, n_pss * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_sss_h12_np_est, 62 * 2 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_sss_h12_est, 62 * 2 * 2 * sizeof(cufftDoubleComplex)));
    checkCudaErrors(cudaMalloc(&d_log_lik, 168 * 2 * 2 * sizeof(double)));

    sss_detect_getce_sss_multiblocks_step1_kernel<<<n_pss, 128>>>(d_capbuf, n_pss,
                                                                  n_id_2_est, peak_loc,
                                                                  fc_requested, fc_programmed, fs_programmed, peak_freq,
                                                                  // output
                                                                  d_h_sm, d_pss_np_inv, &d_sss_raw[0], &d_sss_raw[n_pss * 62]);
    checkCudaErrors(cudaDeviceSynchronize());

    sss_detect_getce_sss_multiblocks_step2_kernel<<<62, n_pss>>>(n_pss,
                                                                 d_h_sm, d_pss_np_inv, &d_sss_raw[0], &d_sss_raw[n_pss * 62],
                                                                 // output
                                                                 d_sss_h12_np_est, &d_sss_h12_est[0], &d_sss_h12_est[62 * 2]);
    checkCudaErrors(cudaDeviceSynchronize());

    sss_detect_ml_kernel<<<168*4, 124>>>(d_sss_h12_np_est, &d_sss_h12_est[0], &d_sss_h12_est[62 * 2], n_id_2_est,
                                         &d_log_lik[0], &d_log_lik[168 * 2]);
    checkCudaErrors(cudaDeviceSynchronize());

#define THRESH2_N_SIGMA 3
    sss_detect_ml_decision_kernel<<<1, 168*4>>>(&d_log_lik[0], THRESH2_N_SIGMA, cell.ind,
                                                fc_requested, fc_programmed, fs_programmed, peak_freq,
                                                // output
                                                d_n_id_1_est, d_frame_start, d_cp_type);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(&h_cp_type, d_cp_type, sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&h_n_id_1_est, d_n_id_1_est, sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&h_frame_start, d_frame_start, sizeof(double), cudaMemcpyDeviceToHost));

    cell_out.frame_start = h_frame_start;
    cell_out.n_id_1 = h_n_id_1_est;
    cell_out.cp_type = (cp_type_t::cp_type_t)h_cp_type;

    const int n_symb_dl = (h_cp_type == (int)cp_type_t::NORMAL ? 7 : 6);
    const int n_ofdm_sym = (6*10*2+2)*n_symb_dl;
    const double frame_start = h_frame_start;

    h_tfg = (cufftDoubleComplex *)malloc(n_ofdm_sym * 12 * 6 * sizeof(cufftDoubleComplex));

    checkCudaErrors(cudaMalloc(&d_tfg, n_ofdm_sym * 12 * 6 * sizeof(cufftDoubleComplex)));
    checkCudaErrors(cudaMalloc(&d_tfg_timestamp, n_ofdm_sym * sizeof(double)));

    /* prologue of function pss_sss_foe() of searcher.cpp */

    const unsigned int pss_sss_dist = lround((128 + (n_symb_dl == 7 ? 9 : 32)) * 16 / FS_LTE * fs_programmed * k_factor);
    double first_sss_dft_location = frame_start + (960 - 128 - (n_symb_dl == 7 ? 9 : 32)-128) * 16 / FS_LTE * fs_programmed * k_factor;
    const int n_sss = ceil((n_cap - 127 - pss_sss_dist - 100) / (9600 * 16 / FS_LTE * fs_programmed * k_factor));
    int sn = 0;
    first_sss_dft_location = fmod(first_sss_dft_location + 0.5, 19200.0) - 0.5;
    if (first_sss_dft_location - 9600 * k_factor > -0.5) {
        first_sss_dft_location -= 9600 * k_factor;
        sn = 1;
    }

    checkCudaErrors(cudaMalloc(&d_M, n_sss * sizeof(cufftDoubleComplex)));

    const unsigned int n_id_cell = cell_out.n_id_cell();

    pss_sss_foe_multiblocks_step1_kernel<<<n_sss, 128>>>(d_capbuf, n_sss,
                                                         n_id_cell, n_symb_dl, first_sss_dft_location, pss_sss_dist, sn,
                                                         fc_requested, fc_programmed, fs_programmed, cell.freq,
                                                         // output
                                                         d_M);
    checkCudaErrors(cudaDeviceSynchronize());

    pss_sss_foe_multiblocks_step2_kernel<<<1, n_sss>>>(d_M, n_sss, pss_sss_dist,
                                                       fc_requested, fc_programmed, fs_programmed, cell.freq,
                                                       // output
                                                       d_adjust_f);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(&h_adjust_f, d_adjust_f, sizeof(double), cudaMemcpyDeviceToHost));

    cell_out.freq_fine = cell_out.freq + h_adjust_f;

    extract_tfg_multiblocks_kernel<<<n_ofdm_sym, 128>>>(d_capbuf, d_tfg, d_tfg_timestamp, d_adjust_f,
                                                        n_id_cell, n_symb_dl, frame_start,
                                                        fc_requested, fc_programmed, fs_programmed, cell_out.freq);
    checkCudaErrors(cudaDeviceSynchronize());

    tfoec_kernel<<<1, n_ofdm_sym>>>(d_tfg, d_rs_extracted, d_tfg_timestamp,
                                    n_id_cell, n_symb_dl,
                                    fc_requested, fc_programmed, fs_programmed,
                                    // output
                                    d_residual_f);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(&h_residual_f, d_residual_f, sizeof(double), cudaMemcpyDeviceToHost));

    cell_out.freq_superfine = cell_out.freq_fine + h_residual_f;

    cufftDoubleComplex *d_ce_filt = d_rs_extracted; // re-use memory allocated for intermediate variables
    double *d_np = d_log_lik; // re-use memory allocated for intermediate variables

    chan_est_four_port_step1_kernel<<<122 * 6, 36>>>(d_tfg, 122, n_id_cell, n_symb_dl, &d_ce_filt[122 * 0 * 12], d_err_pwr_acc);

    chan_est_four_port_step2_kernel<<<4, 122 * 2, 2 * 122 * sizeof(double)>>>(d_err_pwr_acc, 122, &d_np[0]);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(h_tfg, d_tfg, n_ofdm_sym * 12 * 6 * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    my_tfg_comp = cmat(n_ofdm_sym, 72);
    for (int i = 0; i < n_ofdm_sym; i++) {
        for (unsigned int j = 0; j < 72; j++) {
            my_tfg_comp(i,j).real() = h_tfg[i * 72 + j].x;
            my_tfg_comp(i,j).imag() = h_tfg[i * 72 + j].y;
        }
    }

    free(h_capbuf);
    free(h_tfg);

    checkCudaErrors(cudaFree(d_capbuf));
    checkCudaErrors(cudaFree(d_tfg));
    checkCudaErrors(cudaFree(d_rs_extracted));
    checkCudaErrors(cudaFree(d_adjust_f));
    checkCudaErrors(cudaFree(d_residual_f));
    checkCudaErrors(cudaFree(d_tfg_timestamp));
    checkCudaErrors(cudaFree(d_sss_h12_np_est));
    checkCudaErrors(cudaFree(d_sss_h12_est));
    checkCudaErrors(cudaFree(d_frame_start));
    checkCudaErrors(cudaFree(d_n_id_1_est));
    checkCudaErrors(cudaFree(d_cp_type));

    checkCudaErrors(cudaFree(d_h_sm));
    checkCudaErrors(cudaFree(d_sss_raw));
    checkCudaErrors(cudaFree(d_pss_np_inv));

    checkCudaErrors(cudaFree(d_M));

    checkCudaErrors(cudaFree(d_err_pwr_acc));
    return cell_out;
}

