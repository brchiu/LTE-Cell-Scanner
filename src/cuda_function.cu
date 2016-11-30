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
#define SQRT2_INV   (0.7071067817811865475)
#define SQRT12_INV  (0.0883883476483184)

__constant__ cufftDoubleComplex pss_td[3][256];
__constant__ cufftDoubleComplex d_tw128[SIGNAL_SIZE];
__constant__ short d_radix2_bitreverse[SIGNAL_SIZE];
__constant__ short d_radix4_bitreverse[SIGNAL_SIZE];

cufftDoubleComplex h_tw128[SIGNAL_SIZE];
short h_radix2_bitreverse[SIGNAL_SIZE];
short h_radix4_bitreverse[SIGNAL_SIZE];

extern "C" __device__ void kernel_fft_radix2(cufftDoubleComplex *c_io, int N);


extern "C" void cuda_reset_device()
{
    cudaDeviceReset();
}

extern "C" void copy_pss_to_device()
{
    int i, t, len;
    cufftDoubleComplex pss[3][256];

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

extern "C" unsigned int reverse_bit(unsigned int n, int nbits)
{
    unsigned int reverse_num = 0;

    for (int i = 0; i < nbits; i++) {
        if (n & (1 << i))
            reverse_num |= (1 << ((nbits - 1) - i));
    }

    return reverse_num;
}

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

#define COMPLEX_MUL_REAL(a, b)  ((a).x * (b).x - (a).y * (b).y)
#define COMPLEX_MUL_IMAG(a, b)  ((a).x * (b).y + (a).y * (b).x)

__global__ void xc_correlate_kernel(cufftDoubleComplex *d_capbuf, double *d_xc_sqr,
                                    double *d_xc_incoherent_single, double *d_xc_incoherent,
                                    unsigned int n_cap, uint8 ds_comb_arm,
                                    unsigned int t, double f, double fs)
{
    __shared__ cufftDoubleComplex s_fshift_pss[256], s_capbuf[256 + 137];

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

    double real, imag;

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
        double xc_incoherent_single_val = d_xc_sqr[index];
        for (m = 1; m < max_m; m++) {
            unsigned int span = m * 0.005 * fs;
            xc_incoherent_single_val += d_xc_sqr[index + span];
        }
        double xc_incoherent_value = d_xc_incoherent_single[index] = xc_incoherent_single_val / max_m;

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


__global__ void xc_incoherent_collapsed_kernel(double *d_xc_incoherent,
                                               double *d_xc_incoherent_collapsed_pow, int *d_xc_incoherent_collapsed_frq,
                                               unsigned int n_f)
{
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
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

__global__ void sp_incoherent_kernel(cufftDoubleComplex *d_capbuf, double *d_sp_incoherent, double *d_Z_th1, unsigned int n_cap, double Z_th1_factor)
{
    __shared__ double s_sqr[512];

    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int n_comb_sp = (n_cap - 136 - 137) / 9600;
    unsigned int index = bid * 16 + tid;
    double value;

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
        d_Z_th1[index] = d_sp_incoherent[index] * Z_th1_factor;
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

    cufftDoubleComplex *h_capbuf = (cufftDoubleComplex *)NULL, *d_capbuf = (cufftDoubleComplex *)NULL;
    double *h_f = (double *)NULL, *d_f = (double *)NULL;
    double *h_xc_sqr = (double *)NULL, *d_xc_sqr = (double *)NULL;
    double *h_xc_incoherent_single = (double *)NULL, *d_xc_incoherent_single = (double *)NULL;
    double *h_xc_incoherent = (double *)NULL, *d_xc_incoherent = (double *)NULL;
    double *h_xc_incoherent_collapsed_pow = (double *)NULL, *d_xc_incoherent_collapsed_pow = (double *)NULL;
    int *h_xc_incoherent_collapsed_frq = (int *)NULL, *d_xc_incoherent_collapsed_frq = (int *)NULL;
    double *h_sp_incoherent = (double *)NULL, *d_sp_incoherent = (double *)NULL;
    double *h_Z_th1 = (double *)NULL, *d_Z_th1 = (double *)NULL;

    h_capbuf = (cufftDoubleComplex *)malloc(n_cap * sizeof(cufftDoubleComplex));
    h_f = (double *)malloc(n_f * sizeof(double));
    h_xc_incoherent_single = (double *)malloc(3 * n_f * 9600 * sizeof(double));
    h_xc_incoherent = (double *)malloc(3 * n_f * 9600 * sizeof(double));
    h_xc_incoherent_collapsed_pow = (double *)malloc(3 * 9600 * sizeof(double));
    h_xc_incoherent_collapsed_frq = (int *)malloc(3 * 9600 * sizeof(int));
    h_sp_incoherent = (double *)malloc(9600 * sizeof(double));
    h_Z_th1 = (double *)malloc(9600 * sizeof(double));
    h_xc_sqr = (double *)malloc(n_cap * sizeof(double));

    checkCudaErrors(cudaMalloc(&d_capbuf, n_cap * sizeof(cufftDoubleComplex)));
    checkCudaErrors(cudaMalloc(&d_f, n_f * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_xc_sqr, n_cap * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_xc_incoherent_single, 3 * n_f * 9600 * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_xc_incoherent, 3 * n_f * 9600 * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_xc_incoherent_collapsed_pow, 3 * 9600 * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_xc_incoherent_collapsed_frq, 3 * 9600 * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_sp_incoherent, 9600 * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_Z_th1, 9600 * sizeof(double)));

    for (unsigned int i = 0; i < n_cap; i++) {
        h_capbuf[i].x = capbuf[i].real();
        h_capbuf[i].y = capbuf[i].imag();
    }

    checkCudaErrors(cudaMemcpy(d_capbuf, h_capbuf, n_cap * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice));

    for (unsigned int i = 0; i < n_f; i++) {
        h_f[i] = CUDART_PI * 2 * f_search_set[i] * fc_programmed / (fs_programmed * (fc_requested - f_search_set[i]));
    }

    checkCudaErrors(cudaMemcpy(d_f, h_f, n_f * sizeof(double), cudaMemcpyHostToDevice));

    /* xc_correlate, xc_combine, xc_delay_spread */
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

__global__ void extract_tfg_kernel(cufftDoubleComplex *d_capbuf, cufftDoubleComplex *d_tfg, cufftDoubleComplex *d_rs_extracted, double *d_tfg_timestamp,
                                   unsigned short n_id_cell, int n_symb_dl, double frame_start,
                                   double fc_requested, double fc_programmed, double fs_programmed, double freq_fine,
                                   // output
                                   double *d_residual_f)
{
    __shared__ unsigned int rs_dl[20 * 3];

    const unsigned int tid = threadIdx.x;

    cufftDoubleComplex s_capbuf[128];
    int dft_location_i;
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

    __syncthreads();

    // generate random sequences for symbol 0, 1, 3/4 of 20 slots

    if (tid < 20 * 3) {
        int slot = tid / 3;
        int l = (tid % 3 == 2) ? n_symb_dl - 3 : tid % 3;
        int cinit = ((7 * (slot + 1) + l + 1) * (2 * n_id_cell + 1) << 10) + 2 * n_id_cell + (n_symb_dl == 7 ? 1 : 0);

        pn_seq_lsb_to_msb(cinit, 6 * 2 * 2, (55 - 3) * 2 * 2, &rs_dl[tid]);
    }

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

        d_tfg[tid * 72 + 35 + i].x = SQRT12_INV * COMPLEX_MUL_REAL(s_capbuf[i], coeff);
        d_tfg[tid * 72 + 35 + i].y = SQRT12_INV * COMPLEX_MUL_IMAG(s_capbuf[i], coeff);

        coeff.y = -coeff.y;

        d_tfg[tid * 72 + 36 - i].x = SQRT12_INV * COMPLEX_MUL_REAL(s_capbuf[128 - i], coeff);
        d_tfg[tid * 72 + 36 - i].y = SQRT12_INV * COMPLEX_MUL_IMAG(s_capbuf[128 - i], coeff);
    }

    __syncthreads();
}


__global__ void tfoec_kernel(cufftDoubleComplex *d_capbuf, cufftDoubleComplex *d_tfg, cufftDoubleComplex *d_rs_extracted, double *d_tfg_timestamp,
                             unsigned short n_id_cell, int n_symb_dl, double frame_start,
                             double fc_requested, double fc_programmed, double fs_programmed, double freq_fine,
                             // output
                             double *d_residual_f)
{
    __shared__ unsigned int rs_dl[20 * 3];
    __shared__ float foe_real, foe_imag;
    __shared__ float toe_real, toe_imag;

    const unsigned int tid = threadIdx.x;
    double dft_location = d_tfg_timestamp[tid];
    double late;

    // generate random sequences for symbol 0, 1, 3/4 of 20 slots

    if (tid < 20 * 3) {
        int slot = tid / 3;
        int l = (tid % 3 == 2) ? n_symb_dl - 3 : tid % 3;
        int cinit = ((7 * (slot + 1) + l + 1) * (2 * n_id_cell + 1) << 10) + 2 * n_id_cell + (n_symb_dl == 7 ? 1 : 0);

        pn_seq_lsb_to_msb(cinit, 6 * 2 * 2, (55 - 3) * 2 * 2, &rs_dl[tid]);
    }

    foe_real = 0.0; foe_imag = 0.0;
    toe_real = 0.0; toe_imag = 0.0;
    __syncthreads();

    if (tid < 122 * 2) {
        int slot = tid / 2;
        int l = (tid & 1) ? (n_symb_dl - 3) : 0;
        int rs_bits = rs_dl[((tid / 2) % 20) * 3 + ((l == 0) ? 0 : 2)];
        int v_offset = ((n_id_cell % 6) + ((l == 0) ? 0 : 3)) % 6;

        // elem_mult(rs_extracted.get_row(t), conj(rs_dl.get_rs(mod(t, 20), sym_num)))
        for (unsigned int i = 0; i < 12; i++, rs_bits >>= 2, v_offset += 6) {
            cufftDoubleComplex std_rs, rcvd_rs;

            // rs_symb = 1 / sqrt(2) ((1 - 2 * c(0)) + j (1 - 2 * c(1))

            std_rs.x = SQRT2_INV * (1 - ((rs_bits & 1) * 2));
            std_rs.y = SQRT2_INV * (1 - ((rs_bits & 2)));

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

        __syncthreads();
    }

    __syncthreads();

    double residual_f = angle(foe_real, foe_imag) / (2*CUDART_PI) / 0.0005;
    double k_factor_residual = (fc_requested - residual_f) / fc_programmed;
    late = dft_location - k_factor_residual * dft_location;

    *d_residual_f = residual_f;

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
        int l1 = (tid & 1) ? (n_symb_dl - 3) : 0;
        int rs_bits1 = rs_dl[(slot1 % 20) * 3 + ((l1 == 0) ? 0 : 2)];
        int v_offset1 = ((n_id_cell % 6) + ((l1 == 0) ? 0 : 3)) % 6;

        int slot2 = (tid + 1) / 2;
        int l2 = ((tid + 1) & 1) ? (n_symb_dl - 3) : 0;
        int rs_bits2 = rs_dl[(slot2 % 20) * 3 + ((l2 == 0) ? 0 : 2)];
        int v_offset2 = ((n_id_cell % 6) + ((l2 == 0) ? 0 : 3)) % 6;

        float real, imag;

        cufftDoubleComplex toe1, toe2;
        cufftDoubleComplex std_rs, rcvd_rs;
        cufftDoubleComplex r1v, r2v, r2v_prev;

        toe1.x = 0.0; toe1.y = 0.0;
        toe2.x = 0.0; toe2.y = 0.0;
        r2v_prev.x = 0.0; r2v_prev.y = 0.0;

#define SWAP(x,y) \
    do { (tmp) = (x); (x) = (y); (y) = (tmp); \
    } while(0)

        if (v_offset2 < v_offset1) {
            int tmp;

            SWAP(slot1, slot2);
            SWAP(l1, l2);
            SWAP(rs_bits1, rs_bits2);
            SWAP(v_offset1, v_offset2);
        }

        for (unsigned int i = 0; i < 12; i++, rs_bits1 >>= 2, v_offset1 += 6, rs_bits2 >>= 2, v_offset2 += 6) {

            // rs_symb = 1 / sqrt(2) ((1 - 2 * c(0)) + j (1 - 2 * c(1))

            std_rs.x = SQRT2_INV * (1 - ((rs_bits1 & 1) * 2));
            std_rs.y = SQRT2_INV * (1 - ((rs_bits1 & 2)));

            rcvd_rs = d_tfg[(slot1 * n_symb_dl + l1) * 72 + v_offset1];

            std_rs.y = -std_rs.y;

            r1v.x = COMPLEX_MUL_REAL(rcvd_rs, std_rs);
            r1v.y = -COMPLEX_MUL_IMAG(rcvd_rs, std_rs); // this r1v is actually conj(r1v)

            std_rs.x = SQRT2_INV * (1 - ((rs_bits2 & 1) * 2));
            std_rs.y = SQRT2_INV * (1 - ((rs_bits2 & 2)));

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

    __syncthreads();
}

extern "C" Cell extract_tfg_and_tfoec(
    const Cell & cell,
    const cvec & capbuf_raw,
    const double & fc_requested,
    const double & fc_programmed,
    const double & fs_programmed,
    // Output
    cmat & my_tfg_comp)
{
    const double frame_start = cell.frame_start;
    const int n_symb_dl = cell.n_symb_dl();
    const int n_ofdm_sym = (6*10*2+2)*n_symb_dl;

    unsigned int n_cap = capbuf_raw.length();
    cufftDoubleComplex *h_capbuf = (cufftDoubleComplex *)NULL, *d_capbuf = (cufftDoubleComplex *)NULL;
    cufftDoubleComplex *h_tfg = (cufftDoubleComplex *)NULL, *d_tfg = (cufftDoubleComplex *)NULL;
    cufftDoubleComplex *d_rs_extracted = (cufftDoubleComplex *)NULL;
    double h_residual_f, *d_residual_f = (double *)NULL;
    double *h_tfg_timestamp = (double *)NULL, *d_tfg_timestamp = (double *)NULL;

    h_capbuf = (cufftDoubleComplex *)malloc(n_cap * sizeof(cufftDoubleComplex));
    h_tfg = (cufftDoubleComplex *)malloc(n_ofdm_sym * 12 * 6 * sizeof(cufftDoubleComplex));
    h_tfg_timestamp = (double *)malloc(n_ofdm_sym * sizeof(double));

    checkCudaErrors(cudaMalloc(&d_capbuf, n_cap * sizeof(cufftDoubleComplex)));
    checkCudaErrors(cudaMalloc(&d_tfg, n_ofdm_sym * 12 * 6 * sizeof(cufftDoubleComplex)));
    checkCudaErrors(cudaMalloc(&d_rs_extracted, 2 * 122 * 12 * sizeof(cufftDoubleComplex)));
    checkCudaErrors(cudaMalloc(&d_residual_f, sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_tfg_timestamp, n_ofdm_sym * sizeof(double)));

    for (unsigned int i = 0; i < n_cap; i++) {
        h_capbuf[i].x = capbuf_raw[i].real();
        h_capbuf[i].y = capbuf_raw[i].imag();
    }
    checkCudaErrors(cudaMemcpy(d_capbuf, h_capbuf, n_cap * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice));

    extract_tfg_kernel<<<1, n_ofdm_sym>>>(d_capbuf, d_tfg, d_rs_extracted, d_tfg_timestamp,
                                          cell.n_id_cell(), n_symb_dl, frame_start,
                                          fc_requested, fc_programmed, fs_programmed, cell.freq_fine,
                                          // output
                                          d_residual_f);

    tfoec_kernel<<<1, n_ofdm_sym>>>(d_capbuf, d_tfg, d_rs_extracted, d_tfg_timestamp,
                                    cell.n_id_cell(), cell.n_symb_dl(), frame_start,
                                    fc_requested, fc_programmed, fs_programmed, cell.freq_fine,
                                    // output
                                    d_residual_f);

    checkCudaErrors(cudaMemcpy(h_tfg, d_tfg, n_ofdm_sym * 12 * 6 * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&h_residual_f, d_residual_f, sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_tfg_timestamp, d_tfg_timestamp, n_ofdm_sym * sizeof(double), cudaMemcpyDeviceToHost));

    my_tfg_comp = cmat(n_ofdm_sym, 72);
    for (int i = 0; i < n_ofdm_sym; i++) {
        for (unsigned int j = 0; j < 72; j++) {
            my_tfg_comp.set(i, j, complex<double>(h_tfg[i * 72 + j].x, h_tfg[i * 72 + j].y));
        }
    }

    free(h_capbuf);
    free(h_tfg);

    checkCudaErrors(cudaFree(d_capbuf));
    checkCudaErrors(cudaFree(d_tfg));
    checkCudaErrors(cudaFree(d_rs_extracted));
    checkCudaErrors(cudaFree(d_residual_f));
    checkCudaErrors(cudaFree(d_tfg_timestamp));

    Cell cell_out(cell);
    cell_out.freq_superfine = cell_out.freq_fine + h_residual_f;
    return cell_out;
}



