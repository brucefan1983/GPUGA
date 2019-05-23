/*
    Copyright 2019 Zheyong Fan
    This file is part of GPUGA.
    GPUGA is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUGA is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUGA.  If not, see <http://www.gnu.org/licenses/>.
*/


/*----------------------------------------------------------------------------80
Calculate force, energy, and stress
------------------------------------------------------------------------------*/


#include "fitness.cuh"
#include "mic.cuh"
#include "error.cuh"
#include "common.cuh"

//Easy labels for indexing
#define A      0
#define Q      1
#define LAMBDA 2
#define B      3
#define MU     4
#define B2     5
#define MU2    6
#define BETA   7
#define EN     8 // special name for n to avoid conflict
#define H      9
#define R1     10
#define R2     11
#define PI_FACTOR 12
#define MINUS_HALF_OVER_N 13


void Fitness::update_potential(double* potential_parameters)
{
    int n_entries = num_types * num_types * num_types;
    double a = potential_parameters[0];
    double q = potential_parameters[1];
    double lambda = potential_parameters[2];
    double b = potential_parameters[3];
    double mu = potential_parameters[4];
    double b2 = potential_parameters[5];
    double mu2 = potential_parameters[6];
    double beta = potential_parameters[7];
    double r1 = potential_parameters[8];
    double r2 = potential_parameters[9];
    double n = 1.0;
    double h = -1.0/3.0;


    for (int i = 0; i < n_entries; i++)
    {
        cpu_ters[i*NUM_PARAMS + A] = a;
        cpu_ters[i*NUM_PARAMS + B] = b;
        cpu_ters[i*NUM_PARAMS + LAMBDA] = lambda;
        cpu_ters[i*NUM_PARAMS + MU] = mu;
        cpu_ters[i*NUM_PARAMS + BETA] = beta;
        cpu_ters[i*NUM_PARAMS + EN] = n;
        cpu_ters[i*NUM_PARAMS + H] = h;
        cpu_ters[i*NUM_PARAMS + R1] = r1;
        cpu_ters[i*NUM_PARAMS + R2] = r2;
        cpu_ters[i*NUM_PARAMS + B2] = b2;
        cpu_ters[i*NUM_PARAMS + MU2] = mu2;
        cpu_ters[i*NUM_PARAMS + Q] = q;
        cpu_ters[i*NUM_PARAMS + PI_FACTOR] = PI / (r2 - r1);
        cpu_ters[i*NUM_PARAMS + MINUS_HALF_OVER_N] = - 0.5 / n;
    }
    int mem = sizeof(double) * n_entries * NUM_PARAMS;
    CHECK(cudaMemcpy(ters, cpu_ters, mem, cudaMemcpyHostToDevice));
}


static __device__ void find_fr_and_frp
(int i, const double* __restrict__ ters, double d12, double &fr, double &frp)
{
    double exp_factor = LDG(ters,i + A) * exp(- LDG(ters,i + LAMBDA) * d12);
    double d_inv = 1.0 / d12;
    fr = (1.0 + LDG(ters,i + Q) * d_inv) * exp_factor;
    frp = - LDG(ters, i + LAMBDA)*fr - LDG(ters, i + Q)*d_inv*d_inv*exp_factor;
}


static __device__ void find_fa_and_fap
(int i, const double* __restrict__ ters, double d12, double &fa, double &fap)
{
    fa  = LDG(ters, i + B) * exp(- LDG(ters, i + MU) * d12);
    fap = - LDG(ters, i + MU) * fa;
    double tmp =  LDG(ters, i + B2) * exp(- LDG(ters, i + MU2) * d12);
    fa += tmp;
    fap -= LDG(ters, i + MU2) * tmp;
}


static __device__ void find_fc_and_fcp
(int i, const double* __restrict__ ters, double d12, double &fc, double &fcp)
{
    if (d12 < LDG(ters, i + R1)){fc = 1.0; fcp = 0.0;}
    else if (d12 < LDG(ters, i + R2))
    {
        fc = 9.0/16.0 * cos(LDG(ters, i + PI_FACTOR) * (d12 - LDG(ters, i + R1)))
           - 1.0/16 * cos(LDG(ters, i + PI_FACTOR) * (d12 - LDG(ters, i + R1)) * 3.0)
           + 0.5;

        fcp = sin(LDG(ters, i + PI_FACTOR) * (d12 - LDG(ters, i + R1)) * 3.0) 
                * LDG(ters, i + PI_FACTOR) * 3.0/ 16.0
                - sin(LDG(ters, i + PI_FACTOR) * (d12 - LDG(ters, i + R1))) 
                * LDG(ters, i + PI_FACTOR) * 9.0 / 16.0;
    }
    else {fc  = 0.0; fcp = 0.0;}
}


static __device__ void find_fa
(int i, const double* __restrict__ ters, double d12, double &fa)
{
    fa = LDG(ters, i + B) * exp(- LDG(ters, i + MU) * d12);
    fa += LDG(ters, i + B2) * exp(- LDG(ters, i + MU2) * d12);
}


static __device__ void find_fc
(int i, const double* __restrict__ ters, double d12, double &fc)
{
    if (d12 < LDG(ters, i + R1)) {fc  = 1.0;}
    else if (d12 < LDG(ters, i + R2))
    {
        fc = 9.0/16.0 * cos(LDG(ters, i + PI_FACTOR) * (d12 - LDG(ters, i + R1)))
           - 1.0/16 * cos(LDG(ters, i + PI_FACTOR) * (d12 - LDG(ters, i + R1)) * 3.0)
           + 0.5;
    }
    else {fc  = 0.0;}
}


static __device__ void find_g_and_gp
(int i, const double* __restrict__ ters, double cos, double &g, double &gp)
{
    double temp = cos - LDG(ters, i + H);
    g  = temp * temp;
    gp = 2.0 * temp;
}


static __device__ void find_g
(int i, const double* __restrict__ ters, double cos, double &g)
{
    double temp = cos - LDG(ters, i + H);
    g  = temp * temp;
}


// step 1: pre-compute all the bond-order functions and their derivatives
static __global__ void find_force_tersoff_step1
(
    int number_of_particles, int *Na, int *Na_sum,
    const int* __restrict__ g_triclinic, 
    int num_types, int* g_neighbor_number, int* g_neighbor_list, int* g_type,
    const double* __restrict__ ters,
    const double* __restrict__ g_x,
    const double* __restrict__ g_y,
    const double* __restrict__ g_z,
    const double* __restrict__ g_box,
    double* g_b, double* g_bp
)
{
    int N1 = Na_sum[blockIdx.x];
    int N2 = N1 + Na[blockIdx.x];
    int n1 = N1 + threadIdx.x;
    if (n1 < N2)
    {
        int num_types2 = num_types * num_types;
        const double* __restrict__ h = g_box + 18 * blockIdx.x;
        int triclinic = LDG(g_triclinic, blockIdx.x);
        int neighbor_number = g_neighbor_number[n1];
        int type1 = g_type[n1];
        double x1 = LDG(g_x, n1); 
        double y1 = LDG(g_y, n1); 
        double z1 = LDG(g_z, n1);
        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {
            int n2 = g_neighbor_list[n1 + number_of_particles * i1];
            int type2 = g_type[n2];
            double x12  = LDG(g_x, n2) - x1;
            double y12  = LDG(g_y, n2) - y1;
            double z12  = LDG(g_z, n2) - z1;
            dev_apply_mic(triclinic, h, x12, y12, z12);
            double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            double zeta = 0.0;
            for (int i2 = 0; i2 < neighbor_number; ++i2)
            {
                int n3 = g_neighbor_list[n1 + number_of_particles * i2];
                if (n3 == n2) { continue; } // ensure that n3 != n2
                int type3 = g_type[n3];
                double x13 = LDG(g_x, n3) - x1;
                double y13 = LDG(g_y, n3) - y1;
                double z13 = LDG(g_z, n3) - z1;
                dev_apply_mic(triclinic, h, x13, y13, z13);
                double d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
                double cos123 = (x12 * x13 + y12 * y13 + z12 * z13) / (d12*d13);
                double fc_ijk_13, g_ijk;
                int ijk = type1 * num_types2 + type2 * num_types + type3;
                if (d13 > LDG(ters, ijk*NUM_PARAMS + R2)) {continue;}
                find_fc(ijk*NUM_PARAMS, ters, d13, fc_ijk_13);
                find_g(ijk*NUM_PARAMS, ters, cos123, g_ijk);
                zeta += fc_ijk_13 * g_ijk;
            }
            double bzn, b_ijj;
            int ijj = type1 * num_types2 + type2 * num_types + type2;
            bzn = pow(LDG(ters, ijj*NUM_PARAMS + BETA) *
                zeta, LDG(ters, ijj*NUM_PARAMS + EN));
            b_ijj = 
                pow(1.0 + bzn, LDG(ters, ijj*NUM_PARAMS + MINUS_HALF_OVER_N));
            if (zeta < 1.0e-16) // avoid division by 0
            {
                g_b[i1 * number_of_particles + n1]  = 1.0;
                g_bp[i1 * number_of_particles + n1] = 0.0;
            }
            else
            {
                g_b[i1 * number_of_particles + n1]  = b_ijj;
                g_bp[i1 * number_of_particles + n1]
                    = - b_ijj * bzn * 0.5 / ((1.0 + bzn) * zeta);
            }
        }
    }
}


// step 2: calculate all the partial forces dU_i/dr_ij
static __global__ void find_force_tersoff_step2
(
    int number_of_particles, int *Na, int *Na_sum,
    const int* __restrict__ g_triclinic, 
    int num_types, int *g_neighbor_number, int *g_neighbor_list, int *g_type,
    const double* __restrict__ ters,
    const double* __restrict__ g_b,
    const double* __restrict__ g_bp,
    const double* __restrict__ g_x,
    const double* __restrict__ g_y,
    const double* __restrict__ g_z,
    const double* __restrict__ g_box,
    double *g_potential, double *g_f12x, double *g_f12y, double *g_f12z
)
{
    int N1 = Na_sum[blockIdx.x];
    int N2 = N1 + Na[blockIdx.x];
    int n1 = N1 + threadIdx.x;
    if (n1 < N2)
    {
        int num_types2 = num_types * num_types;
        const double* __restrict__ h = g_box + 18 * blockIdx.x;
        int triclinic = LDG(g_triclinic, blockIdx.x);
        int neighbor_number = g_neighbor_number[n1];
        int type1 = g_type[n1];
        double x1 = LDG(g_x, n1); 
        double y1 = LDG(g_y, n1); 
        double z1 = LDG(g_z, n1);
        double pot_energy = 0.0;
        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {
            int index = i1 * number_of_particles + n1;
            int n2 = g_neighbor_list[index];
            int type2 = g_type[n2];

            double x12  = LDG(g_x, n2) - x1;
            double y12  = LDG(g_y, n2) - y1;
            double z12  = LDG(g_z, n2) - z1;
            dev_apply_mic(triclinic, h, x12, y12, z12);
            double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            double d12inv = 1.0 / d12;
            double fc_ijj_12, fcp_ijj_12;
            double fa_ijj_12, fap_ijj_12, fr_ijj_12, frp_ijj_12;
            int ijj = type1 * num_types2 + type2 * num_types + type2;
            find_fc_and_fcp(ijj*NUM_PARAMS, ters, d12, fc_ijj_12, fcp_ijj_12);
            find_fa_and_fap(ijj*NUM_PARAMS, ters, d12, fa_ijj_12, fap_ijj_12);
            find_fr_and_frp(ijj*NUM_PARAMS, ters, d12, fr_ijj_12, frp_ijj_12);

            // (i,j) part
            double b12 = LDG(g_b, index);
            double factor3=(fcp_ijj_12*(fr_ijj_12-b12*fa_ijj_12)+
                            fc_ijj_12*(frp_ijj_12-b12*fap_ijj_12))*d12inv;
            double f12x = x12 * factor3 * 0.5;
            double f12y = y12 * factor3 * 0.5;
            double f12z = z12 * factor3 * 0.5;

            // accumulate potential energy
            pot_energy += fc_ijj_12 * (fr_ijj_12 - b12 * fa_ijj_12) * 0.5;

            // (i,j,k) part
            double bp12 = LDG(g_bp, index);
            for (int i2 = 0; i2 < neighbor_number; ++i2)
            {
                int index_2 = n1 + number_of_particles * i2;
                int n3 = g_neighbor_list[index_2];
                if (n3 == n2) { continue; }
                int type3 = g_type[n3];
                double x13 = LDG(g_x, n3) - x1;
                double y13 = LDG(g_y, n3) - y1;
                double z13 = LDG(g_z, n3) - z1;
                dev_apply_mic(triclinic, h, x13, y13, z13);
                double d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
                double fc_ikk_13, fc_ijk_13, fa_ikk_13, fc_ikj_12, fcp_ikj_12;
                int ikj = type1 * num_types2 + type3 * num_types + type2;
                int ikk = type1 * num_types2 + type3 * num_types + type3;
                int ijk = type1 * num_types2 + type2 * num_types + type3;
                find_fc(ikk*NUM_PARAMS, ters, d13, fc_ikk_13);
                find_fc(ijk*NUM_PARAMS, ters, d13, fc_ijk_13);
                find_fa(ikk*NUM_PARAMS, ters, d13, fa_ikk_13);
                find_fc_and_fcp(ikj*NUM_PARAMS, ters, d12,
                                	fc_ikj_12, fcp_ikj_12);
                double bp13 = LDG(g_bp, index_2);
                double one_over_d12d13 = 1.0 / (d12 * d13);
                double cos123 = (x12*x13 + y12*y13 + z12*z13)*one_over_d12d13;
                double cos123_over_d12d12 = cos123*d12inv*d12inv;
                double g_ijk, gp_ijk;
                find_g_and_gp(ijk*NUM_PARAMS, ters, cos123, g_ijk, gp_ijk);

                double g_ikj, gp_ikj;
                find_g_and_gp(ikj*NUM_PARAMS, ters, cos123, g_ikj, gp_ikj);

                // derivatives with cosine
                double dc=-fc_ijj_12*bp12*fa_ijj_12*fc_ijk_13*gp_ijk+
                        -fc_ikj_12*bp13*fa_ikk_13*fc_ikk_13*gp_ikj;
                // derivatives with rij
                double dr=(-fc_ijj_12*bp12*fa_ijj_12*fc_ijk_13*g_ijk +
                  (-fcp_ikj_12*bp13*fa_ikk_13*g_ikj +
                  fc_ikj_12*bp13*fa_ikk_13*g_ikj)*fc_ikk_13)*d12inv;
                double cos_d = x13 * one_over_d12d13 - x12 * cos123_over_d12d12;
                f12x += (x12 * dr + dc * cos_d)*0.5;
                cos_d = y13 * one_over_d12d13 - y12 * cos123_over_d12d12;
                f12y += (y12 * dr + dc * cos_d)*0.5;
                cos_d = z13 * one_over_d12d13 - z12 * cos123_over_d12d12;
                f12z += (z12 * dr + dc * cos_d)*0.5;
            }
            g_f12x[index] = f12x; g_f12y[index] = f12y; g_f12z[index] = f12z;
        }
        // save potential
        g_potential[n1] = pot_energy;
    }
}


static __global__ void find_force_tersoff_step3
(
    int number_of_particles, int *Na, int *Na_sum,
    const int* __restrict__ g_triclinic,
    int *g_neighbor_number, int *g_neighbor_list,
    const double* __restrict__ g_f12x,
    const double* __restrict__ g_f12y,
    const double* __restrict__ g_f12z,
    const double* __restrict__ g_x,
    const double* __restrict__ g_y,
    const double* __restrict__ g_z,
    const double* __restrict__ g_box,
    double *g_fx, double *g_fy, double *g_fz,
    double *g_sx, double *g_sy, double *g_sz
)
{
    int N1 = Na_sum[blockIdx.x];
    int N2 = N1 + Na[blockIdx.x];
    int n1 = N1 + threadIdx.x;
    if (n1 < N2)
    {
        double s_fx = 0.0; // force_x
        double s_fy = 0.0; // force_y
        double s_fz = 0.0; // force_z
        double s_sx = 0.0; // virial_stress_x
        double s_sy = 0.0; // virial_stress_y
        double s_sz = 0.0; // virial_stress_z
        const double* __restrict__ h = g_box + 18 * blockIdx.x;
        int triclinic = LDG(g_triclinic, blockIdx.x);
        int neighbor_number = g_neighbor_number[n1];
        double x1 = LDG(g_x, n1); 
        double y1 = LDG(g_y, n1); 
        double z1 = LDG(g_z, n1);

        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {
            int index = i1 * number_of_particles + n1;
            int n2 = g_neighbor_list[index];
            int neighbor_number_2 = g_neighbor_number[n2];

            double x12  = LDG(g_x, n2) - x1;
            double y12  = LDG(g_y, n2) - y1;
            double z12  = LDG(g_z, n2) - z1;
            dev_apply_mic(triclinic, h, x12, y12, z12);

            double f12x = LDG(g_f12x, index);
            double f12y = LDG(g_f12y, index);
            double f12z = LDG(g_f12z, index);
            int offset = 0;
            for (int k = 0; k < neighbor_number_2; ++k)
            {
                if (n1 == g_neighbor_list[n2 + number_of_particles * k])
                { offset = k; break; }
            }
            index = offset * number_of_particles + n2;
            double f21x = LDG(g_f12x, index);
            double f21y = LDG(g_f12y, index);
            double f21z = LDG(g_f12z, index);

            // per atom force
            s_fx += f12x - f21x; 
            s_fy += f12y - f21y; 
            s_fz += f12z - f21z; 

            // per-atom virial
            s_sx -= x12 * (f12x - f21x) * 0.5;
            s_sy -= y12 * (f12y - f21y) * 0.5;
            s_sz -= z12 * (f12z - f21z) * 0.5;
        }
        // save force
        g_fx[n1] = s_fx;
        g_fy[n1] = s_fy;
        g_fz[n1] = s_fz;
        // save virial
        g_sx[n1] = s_sx;
        g_sy[n1] = s_sy;
        g_sz[n1] = s_sz;
    }
}


void Fitness::find_force(void)
{
    find_force_tersoff_step1<<<Nc, MAX_ATOM_NUMBER>>>
    (
        N, Na, Na_sum, box.triclinic, num_types,
        neighbor.NN, neighbor.NL, type, ters, x, y, z, box.h, b, bp
    );
    CUDA_CHECK_KERNEL
    find_force_tersoff_step2<<<Nc, MAX_ATOM_NUMBER>>>
    (
        N, Na, Na_sum, box.triclinic, num_types, neighbor.NN, neighbor.NL, type, 
        ters, b, bp, x, y, z, box.h, pe, f12x, f12y, f12z
    );
    CUDA_CHECK_KERNEL
    find_force_tersoff_step3<<<Nc, MAX_ATOM_NUMBER>>>
    (
        N, Na, Na_sum, box.triclinic, neighbor.NN, neighbor.NL, 
        f12x, f12y, f12z, x, y, z, box.h, fx, fy, fz, sxx, syy, szz
    );
    CUDA_CHECK_KERNEL
}


