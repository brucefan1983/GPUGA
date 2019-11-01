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
Calculate force, energy, and virial
------------------------------------------------------------------------------*/


#include "potential.cuh"
#include "box.cuh"
#include "neighbor.cuh"
#include "mic.cuh"
#include "error.cuh"

const float PI              = 3.141592653589793;

//Easy labels for indexing
const int D0                = 0;
const int A                 = 1;
const int R0                = 2;
const int S                 = 3;
const int EN                = 4;
const int BETA              = 5;
const int H                 = 6;
const int R1                = 7;
const int R2                = 8;
const int PI_FACTOR         = 9;
const int MINUS_HALF_OVER_N = 10;


void Potential::update_potential(float* potential_parameters, int num_types)
{
    float r1 = 2.8; // to be read in
    float r2 = 3.2; // to be read in
    if (num_types == 1)
    {
        pot_para.ters[D0]   = potential_parameters[0];
        pot_para.ters[A]    = potential_parameters[1];
        pot_para.ters[R0]   = potential_parameters[2];
        pot_para.ters[S]    = potential_parameters[3];
        pot_para.ters[EN]   = potential_parameters[4];
        pot_para.ters[BETA] = potential_parameters[5];
        pot_para.ters[H]    = potential_parameters[6];
        pot_para.ters[R1]   = r1;
        pot_para.ters[R2]   = r2;
        pot_para.ters[PI_FACTOR] = PI / (r2 - r1);
        pot_para.ters[MINUS_HALF_OVER_N] = - 0.5 / pot_para.ters[EN];
    }
    else
    {
        print_error("GPUGA only supports one atom type now.\n");
    }
}


static __device__ void find_fr_and_frp
(Pot_Para pot_para, float d12, float &fr, float &frp)
{
    float d0 = pot_para.ters[D0];
    float a = pot_para.ters[A];
    float r0 = pot_para.ters[R0];
    float s = pot_para.ters[S];
    fr = d0 / (s - 1) * exp(-sqrt(2.0 * s) * a * (d12 - r0));
    frp = -2.0 * a * fr;
}


static __device__ void find_fa_and_fap
(Pot_Para pot_para, float d12, float &fa, float &fap)
{
    float d0 = pot_para.ters[D0];
    float a = pot_para.ters[A];
    float r0 = pot_para.ters[R0];
    float s = pot_para.ters[S];
    fa = s * d0 / (s - 1) * exp(-sqrt(2.0 / s) * a * (d12 - r0));
    fap = -a * fa;
}


static __device__ void find_fc_and_fcp
(Pot_Para pot_para, float d12, float &fc, float &fcp)
{
    if (d12 < pot_para.ters[R1]) {fc = 1.0; fcp = 0.0;}
    else if (d12 < pot_para.ters[R2])
    {
        fc = 0.5 * cos(pot_para.ters[PI_FACTOR] * (d12 - pot_para.ters[R1])) + 0.5;
        fcp = - sin(pot_para.ters[PI_FACTOR] * (d12 - pot_para.ters[R1])) * pot_para.ters[PI_FACTOR] * 0.5;
    }
    else {fc  = 0.0; fcp = 0.0;}
}


static __device__ void find_fa
(Pot_Para pot_para, float d12, float &fa)
{
    float d0 = pot_para.ters[D0];
    float a = pot_para.ters[A];
    float r0 = pot_para.ters[R0];
    float s = pot_para.ters[S];
    fa = s * d0 / (s - 1) * exp(-sqrt(2.0 / s) * a * (d12 - r0));
}


static __device__ void find_fc
(Pot_Para pot_para, float d12, float &fc)
{
    if (d12 < pot_para.ters[R1]) {fc  = 1.0;}
    else if (d12 < pot_para.ters[R2])
    {
        fc = 0.5 * cos(pot_para.ters[PI_FACTOR] * (d12 - pot_para.ters[R1])) + 0.5;
    }
    else {fc  = 0.0;}
}


static __device__ void find_g_and_gp
(Pot_Para pot_para, float cos, float &g, float &gp)
{
    float x = cos - pot_para.ters[H];
    float beta = pot_para.ters[BETA];
    g = beta * x * x;
    gp = beta * 2.0 * x;
}


static __device__ void find_g
(Pot_Para pot_para, float cos, float &g)
{
    float x = cos - pot_para.ters[H];
    float beta = pot_para.ters[BETA];
    g = beta * x * x;
}


// step 1: pre-compute all the bond-order functions and their derivatives
static __global__ void find_force_tersoff_step1
(
    int number_of_particles, int *Na, int *Na_sum,
    const int* __restrict__ g_triclinic, 
    int num_types, int* g_neighbor_number, int* g_neighbor_list, int* g_type,
    Pot_Para pot_para,
    const float* __restrict__ g_x,
    const float* __restrict__ g_y,
    const float* __restrict__ g_z,
    const float* __restrict__ g_box,
    float* g_b, float* g_bp
)
{
    int N1 = Na_sum[blockIdx.x];
    int N2 = N1 + Na[blockIdx.x];
    int n1 = N1 + threadIdx.x;
    if (n1 < N2)
    {
        const float* __restrict__ h = g_box + 18 * blockIdx.x;
        int triclinic = LDG(g_triclinic, blockIdx.x);
        int neighbor_number = g_neighbor_number[n1];
        //int type1 = g_type[n1];
        float x1 = LDG(g_x, n1); 
        float y1 = LDG(g_y, n1); 
        float z1 = LDG(g_z, n1);
        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {
            int n2 = g_neighbor_list[n1 + number_of_particles * i1];
            //int type2 = g_type[n2];
            float x12  = LDG(g_x, n2) - x1;
            float y12  = LDG(g_y, n2) - y1;
            float z12  = LDG(g_z, n2) - z1;
            dev_apply_mic(triclinic, h, x12, y12, z12);
            float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            float zeta = 0.0;
            for (int i2 = 0; i2 < neighbor_number; ++i2)
            {
                int n3 = g_neighbor_list[n1 + number_of_particles * i2];
                if (n3 == n2) { continue; } // ensure that n3 != n2
                //int type3 = g_type[n3];
                float x13 = LDG(g_x, n3) - x1;
                float y13 = LDG(g_y, n3) - y1;
                float z13 = LDG(g_z, n3) - z1;
                dev_apply_mic(triclinic, h, x13, y13, z13);
                float d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
                float cos123 = (x12 * x13 + y12 * y13 + z12 * z13) / (d12*d13);
                float fc13, g123;

                find_fc(pot_para, d13, fc13);
                find_g(pot_para, cos123, g123);
                zeta += fc13 * g123;
            }
            float bzn, b_ijj;
            bzn = pow(zeta, pot_para.ters[EN]);
            b_ijj = pow(1.0 + bzn, pot_para.ters[MINUS_HALF_OVER_N]);
            g_b[i1 * number_of_particles + n1]  = b_ijj;
            g_bp[i1 * number_of_particles + n1] = - b_ijj * bzn * 0.5 / ((1.0 + bzn) * zeta);
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
    const float* __restrict__ g_b,
    const float* __restrict__ g_bp,
    const float* __restrict__ g_x,
    const float* __restrict__ g_y,
    const float* __restrict__ g_z,
    const float* __restrict__ g_box,
    float *g_potential, float *g_f12x, float *g_f12y, float *g_f12z
)
{
    int N1 = Na_sum[blockIdx.x];
    int N2 = N1 + Na[blockIdx.x];
    int n1 = N1 + threadIdx.x;
    if (n1 < N2)
    {
        const float* __restrict__ h = g_box + 18 * blockIdx.x;
        int triclinic = LDG(g_triclinic, blockIdx.x);
        int neighbor_number = g_neighbor_number[n1];
        //int type1 = g_type[n1];
        float x1 = LDG(g_x, n1); 
        float y1 = LDG(g_y, n1); 
        float z1 = LDG(g_z, n1);
        float pot_energy = 0.0;
        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {
            int index = i1 * number_of_particles + n1;
            int n2 = g_neighbor_list[index];
            //int type2 = g_type[n2];

            float x12  = LDG(g_x, n2) - x1;
            float y12  = LDG(g_y, n2) - y1;
            float z12  = LDG(g_z, n2) - z1;
            dev_apply_mic(triclinic, h, x12, y12, z12);
            float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            float d12inv = 1.0 / d12;
            float fc12, fcp12, fa12, fap12, fr12, frp12;

            find_fc_and_fcp(pot_para, d12, fc12, fcp12);
            find_fa_and_fap(pot_para, d12, fa12, fap12);
            find_fr_and_frp(pot_para, d12, fr12, frp12);

            // (i,j) part
            float b12 = LDG(g_b, index);
            float factor3 = (fcp12 * (fr12 - b12 * fa12) + fc12 * (frp12 - b12 * fap12)) * d12inv;
            float f12x = x12 * factor3 * 0.5;
            float f12y = y12 * factor3 * 0.5;
            float f12z = z12 * factor3 * 0.5;

            // accumulate potential energy
            pot_energy += fc12 * (fr12 - b12 * fa12) * 0.5;

            // (i,j,k) part
            float bp12 = LDG(g_bp, index);
            for (int i2 = 0; i2 < neighbor_number; ++i2)
            {
                int index_2 = n1 + number_of_particles * i2;
                int n3 = g_neighbor_list[index_2];
                if (n3 == n2) { continue; }
                //int type3 = g_type[n3];
                float x13 = LDG(g_x, n3) - x1;
                float y13 = LDG(g_y, n3) - y1;
                float z13 = LDG(g_z, n3) - z1;
                dev_apply_mic(triclinic, h, x13, y13, z13);
                float d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
                float fc13, fa13;
                find_fc(pot_para, d13, fc13);
                find_fa(pot_para, d13, fa13);
                float bp13 = LDG(g_bp, index_2);
                float one_over_d12d13 = 1.0 / (d12 * d13);
                float cos123 = (x12*x13 + y12*y13 + z12*z13) * one_over_d12d13;
                float cos123_over_d12d12 = cos123 * d12inv * d12inv;
                float g123, gp123;
                find_g_and_gp(pot_para, cos123, g123, gp123);

                // derivatives with cosine
                float dc = -fc12 * bp12 * fa12 * fc13 * gp123
                            -fc12 * bp13 * fa13 * fc13 * gp123;
                // derivatives with rij
                float dr = -fcp12 * bp13 * fa13 * g123 * fc13 * d12inv;

                float cos_d = x13 * one_over_d12d13 - x12 * cos123_over_d12d12;
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
    const float* __restrict__ g_f12x,
    const float* __restrict__ g_f12y,
    const float* __restrict__ g_f12z,
    const float* __restrict__ g_x,
    const float* __restrict__ g_y,
    const float* __restrict__ g_z,
    const float* __restrict__ g_box,
    float *g_fx, float *g_fy, float *g_fz,
    float *g_sx, float *g_sy, float *g_sz
)
{
    int N1 = Na_sum[blockIdx.x];
    int N2 = N1 + Na[blockIdx.x];
    int n1 = N1 + threadIdx.x;
    if (n1 < N2)
    {
        float s_fx = 0.0; // force_x
        float s_fy = 0.0; // force_y
        float s_fz = 0.0; // force_z
        float s_sx = 0.0; // virial_x
        float s_sy = 0.0; // virial_y
        float s_sz = 0.0; // virial_z
        const float* __restrict__ h = g_box + 18 * blockIdx.x;
        int triclinic = LDG(g_triclinic, blockIdx.x);
        int neighbor_number = g_neighbor_number[n1];
        float x1 = LDG(g_x, n1); 
        float y1 = LDG(g_y, n1); 
        float z1 = LDG(g_z, n1);

        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {
            int index = i1 * number_of_particles + n1;
            int n2 = g_neighbor_list[index];
            int neighbor_number_2 = g_neighbor_number[n2];

            float x12  = LDG(g_x, n2) - x1;
            float y12  = LDG(g_y, n2) - y1;
            float z12  = LDG(g_z, n2) - z1;
            dev_apply_mic(triclinic, h, x12, y12, z12);

            float f12x = LDG(g_f12x, index);
            float f12y = LDG(g_f12y, index);
            float f12z = LDG(g_f12z, index);
            int offset = 0;
            for (int k = 0; k < neighbor_number_2; ++k)
            {
                if (n1 == g_neighbor_list[n2 + number_of_particles * k])
                { offset = k; break; }
            }
            index = offset * number_of_particles + n2;
            float f21x = LDG(g_f12x, index);
            float f21y = LDG(g_f12y, index);
            float f21z = LDG(g_f12z, index);

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


void Potential::find_force
(
    int num_types, int Nc, int N, int *Na, int *Na_sum,
    int max_Na, int *type, Box *box, Neighbor *neighbor,
    float *x, float *y, float *z, float *fx, float *fy, float *fz, 
    float *sxx, float *syy, float *szz, float *pe, 
    float *f12x, float *f12y, float *f12z, float *b, float *bp
)
{
    find_force_tersoff_step1<<<Nc, max_Na>>>
    (
        N, Na, Na_sum, box->triclinic, num_types,
        neighbor->NN, neighbor->NL, type, pot_para, x, y, z, box->h, b, bp
    );
    CUDA_CHECK_KERNEL
    find_force_tersoff_step2<<<Nc, max_Na>>>
    (
        N, Na, Na_sum, box->triclinic, num_types, neighbor->NN, neighbor->NL, type, 
        pot_para, b, bp, x, y, z, box->h, pe, f12x, f12y, f12z
    );
    CUDA_CHECK_KERNEL
    find_force_tersoff_step3<<<Nc, max_Na>>>
    (
        N, Na, Na_sum, box->triclinic, neighbor->NN, neighbor->NL, 
        f12x, f12y, f12z, x, y, z, box->h, fx, fy, fz, sxx, syy, szz
    );
    CUDA_CHECK_KERNEL
}


