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
#define D0     0
#define A      1
#define R0     2
#define S      3
#define BETA   4
#define EN     5 // special name for n to avoid conflict
#define H      6
#define R1     7
#define R2     8
#define PI_FACTOR 9
#define MINUS_HALF_OVER_N 10
#define C1     11
#define C2     12
#define C3     13



void Fitness::update_potential(double* potential_parameters)
{
    int n_entries = num_types * num_types * num_types;
    double d0 = potential_parameters[0];
    double a = potential_parameters[1];
    double r0 = potential_parameters[2];
    double s = potential_parameters[3];
    double n = potential_parameters[4];
    double h = potential_parameters[5];
    double c1 = potential_parameters[6];
    double c2 = potential_parameters[7];
    double c3 = potential_parameters[8];

    double r1 = 2.8;
    double r2 = 3.2;
    double beta = 1.0;

    for (int i = 0; i < n_entries; i++)
    {
        pot_para.ters[D0] = d0;
        pot_para.ters[A] = a;
        pot_para.ters[R0] = r0;
        pot_para.ters[S] = s;
        pot_para.ters[BETA] = beta;
        pot_para.ters[EN] = n;
        pot_para.ters[H] = h;
        pot_para.ters[C1] = c1;
        pot_para.ters[C2] = c2;
        pot_para.ters[C3] = c3;
        pot_para.ters[R1] = r1;
        pot_para.ters[R2] = r2;
        pot_para.ters[PI_FACTOR] = PI / (r2 - r1);
        pot_para.ters[MINUS_HALF_OVER_N] = - 0.5 / n;
    }
}


static __device__ void find_fr_and_frp
(Pot_Para pot_para, double d12, double &fr, double &frp)
{
    double d0 = pot_para.ters[D0];
    double a = pot_para.ters[A];
    double r0 = pot_para.ters[R0];
    double s = pot_para.ters[S];
    fr = d0 / (s-1) * exp(-sqrt(2.0*s)*a*(d12-r0));
    frp = -2.0*a*fr;
}


static __device__ void find_fa_and_fap
(Pot_Para pot_para, double d12, double &fa, double &fap)
{
    double d0 = pot_para.ters[D0];
    double a = pot_para.ters[A];
    double r0 = pot_para.ters[R0];
    double s = pot_para.ters[S];
    fa = s* d0 / (s-1) * exp(-sqrt(2.0/s)*a*(d12-r0));
    fap = -a*fa;
}


static __device__ void find_fc_and_fcp
(Pot_Para pot_para, double d12, double &fc, double &fcp)
{
    if (d12 < pot_para.ters[R1]){fc = 1.0; fcp = 0.0;}
    else if (d12 < pot_para.ters[R2])
    {
        fc = 0.5 * cos(pot_para.ters[PI_FACTOR] * (d12 - pot_para.ters[R1]))
           + 0.5;

        fcp = - sin(pot_para.ters[PI_FACTOR] * (d12 - pot_para.ters[R1])) 
                * pot_para.ters[PI_FACTOR] * 0.5;
    }
    else {fc  = 0.0; fcp = 0.0;}
}


static __device__ void find_fa
(Pot_Para pot_para, double d12, double &fa)
{
    double d0 = pot_para.ters[D0];
    double a = pot_para.ters[A];
    double r0 = pot_para.ters[R0];
    double s = pot_para.ters[S];
    fa = s* d0 / (s-1) * exp(-sqrt(2.0/s)*a*(d12-r0));
}


static __device__ void find_fc
(Pot_Para pot_para, double d12, double &fc)
{
    if (d12 < pot_para.ters[R1]) {fc  = 1.0;}
    else if (d12 < pot_para.ters[R2])
    {
        fc = 0.5 * cos(pot_para.ters[PI_FACTOR] * (d12 - pot_para.ters[R1]))
           + 0.5;
    }
    else {fc  = 0.0;}
}


static __device__ void find_g_and_gp
(Pot_Para pot_para, double cos, double &g, double &gp)
{
    double x = cos - pot_para.ters[H];
    g = pot_para.ters[C3] * x;
    g = (g + pot_para.ters[C2]) * x;
    g = (g + pot_para.ters[C1]) * x;
    gp = pot_para.ters[C3] * 3.0 * x;
    gp = (gp + pot_para.ters[C2] * 2.0) * x;
    gp = gp + pot_para.ters[C1];
}


static __device__ void find_g
(Pot_Para pot_para, double cos, double &g)
{
    double x = cos - pot_para.ters[H];
    g = pot_para.ters[C3] * x;
    g = (g + pot_para.ters[C2]) * x;
    g = (g + pot_para.ters[C1]) * x;
}


// step 1: pre-compute all the bond-order functions and their derivatives
static __global__ void find_force_tersoff_step1
(
    int number_of_particles, int *Na, int *Na_sum,
    const int* __restrict__ g_triclinic, 
    int num_types, int* g_neighbor_number, int* g_neighbor_list, int* g_type,
    Pot_Para pot_para,
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
        const double* __restrict__ h = g_box + 18 * blockIdx.x;
        int triclinic = LDG(g_triclinic, blockIdx.x);
        int neighbor_number = g_neighbor_number[n1];
        //int type1 = g_type[n1];
        double x1 = LDG(g_x, n1); 
        double y1 = LDG(g_y, n1); 
        double z1 = LDG(g_z, n1);
        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {
            int n2 = g_neighbor_list[n1 + number_of_particles * i1];
            //int type2 = g_type[n2];
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
                //int type3 = g_type[n3];
                double x13 = LDG(g_x, n3) - x1;
                double y13 = LDG(g_y, n3) - y1;
                double z13 = LDG(g_z, n3) - z1;
                dev_apply_mic(triclinic, h, x13, y13, z13);
                double d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
                double cos123 = (x12 * x13 + y12 * y13 + z12 * z13) / (d12*d13);
                double fc13, g123;

                find_fc(pot_para, d13, fc13);
                find_g(pot_para, cos123, g123);
                zeta += fc13 * g123;
            }
            double bzn, b_ijj;
            bzn = pow(zeta, pot_para.ters[EN]);
            b_ijj = pow(1.0 + bzn, pot_para.ters[MINUS_HALF_OVER_N]);
            g_b[i1 * number_of_particles + n1]  = b_ijj;
            g_bp[i1 * number_of_particles + n1]
                = - b_ijj * bzn * 0.5 / ((1.0 + bzn) * zeta);
        }
    }
}


// step 2: calculate all the partial forces dU_i/dr_ij
static __global__ void find_force_tersoff_step2
(
    int number_of_particles, int *Na, int *Na_sum,
    const int* __restrict__ g_triclinic, 
    int num_types, int *g_neighbor_number, int *g_neighbor_list, int *g_type,
    Pot_Para pot_para,
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
        const double* __restrict__ h = g_box + 18 * blockIdx.x;
        int triclinic = LDG(g_triclinic, blockIdx.x);
        int neighbor_number = g_neighbor_number[n1];
        //int type1 = g_type[n1];
        double x1 = LDG(g_x, n1); 
        double y1 = LDG(g_y, n1); 
        double z1 = LDG(g_z, n1);
        double pot_energy = 0.0;
        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {
            int index = i1 * number_of_particles + n1;
            int n2 = g_neighbor_list[index];
            //int type2 = g_type[n2];

            double x12  = LDG(g_x, n2) - x1;
            double y12  = LDG(g_y, n2) - y1;
            double z12  = LDG(g_z, n2) - z1;
            dev_apply_mic(triclinic, h, x12, y12, z12);
            double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            double d12inv = 1.0 / d12;
            double fc12, fcp12;
            double fa12, fap12, fr12, frp12;

            find_fc_and_fcp(pot_para, d12, fc12, fcp12);
            find_fa_and_fap(pot_para, d12, fa12, fap12);
            find_fr_and_frp(pot_para, d12, fr12, frp12);

            // (i,j) part
            double b12 = LDG(g_b, index);
            double factor3=(fcp12*(fr12-b12*fa12)+
                            fc12*(frp12-b12*fap12))*d12inv;
            double f12x = x12 * factor3 * 0.5;
            double f12y = y12 * factor3 * 0.5;
            double f12z = z12 * factor3 * 0.5;

            // accumulate potential energy
            pot_energy += fc12*(fr12-b12*fa12)*0.5;

            // (i,j,k) part
            double bp12 = LDG(g_bp, index);
            for (int i2 = 0; i2 < neighbor_number; ++i2)
            {
                int index_2 = n1 + number_of_particles * i2;
                int n3 = g_neighbor_list[index_2];
                if (n3 == n2) { continue; }
                //int type3 = g_type[n3];
                double x13 = LDG(g_x, n3) - x1;
                double y13 = LDG(g_y, n3) - y1;
                double z13 = LDG(g_z, n3) - z1;
                dev_apply_mic(triclinic, h, x13, y13, z13);
                double d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
                double fc13, fa13;
                find_fc(pot_para, d13, fc13);
                find_fa(pot_para, d13, fa13);
                double bp13 = LDG(g_bp, index_2);
                double one_over_d12d13 = 1.0 / (d12 * d13);
                double cos123 = (x12*x13 + y12*y13 + z12*z13)*one_over_d12d13;
                double cos123_over_d12d12 = cos123*d12inv*d12inv;
                double g123, gp123;
                find_g_and_gp(pot_para, cos123, g123, gp123);

                // derivatives with cosine
                double dc = -fc12 * bp12 * fa12 * fc13 * gp123
                            -fc12 * bp13 * fa13 * fc13 * gp123;
                // derivatives with rij
                double dr = -fcp12 * bp13 * fa13 * g123 * fc13 * d12inv;

                double cos_d = x13 * one_over_d12d13 - x12 * cos123_over_d12d12;
                f12x += (x12 * dr + dc * cos_d)*0.5;
                cos_d = y13 * one_over_d12d13 - y12 * cos123_over_d12d12;
                f12y += (y12 * dr + dc * cos_d)*0.5;
                cos_d = z13 * one_over_d12d13 - z12 * cos123_over_d12d12;
                f12z += (z12 * dr + dc * cos_d)*0.5;
            }
            g_f12x[index] = f12x; 
            g_f12y[index] = f12y; 
            g_f12z[index] = f12z;
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
        neighbor.NN, neighbor.NL, type, pot_para, x, y, z, box.h, b, bp
    );
    CUDA_CHECK_KERNEL
    find_force_tersoff_step2<<<Nc, MAX_ATOM_NUMBER>>>
    (
        N, Na, Na_sum, box.triclinic, num_types, neighbor.NN, neighbor.NL, type, 
        pot_para, b, bp, x, y, z, box.h, pe, f12x, f12y, f12z
    );
    CUDA_CHECK_KERNEL
    find_force_tersoff_step3<<<Nc, MAX_ATOM_NUMBER>>>
    (
        N, Na, Na_sum, box.triclinic, neighbor.NN, neighbor.NL, 
        f12x, f12y, f12z, x, y, z, box.h, fx, fy, fz, sxx, syy, szz
    );
    CUDA_CHECK_KERNEL
}


