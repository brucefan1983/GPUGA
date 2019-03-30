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
find the neighbor list
------------------------------------------------------------------------------*/


#include "neighbor.cuh"
#include "box.cuh"
#include "mic.cuh"
#include "error.cuh"
#include "common.cuh"
#define BLOCK_SIZE 256


Neighbor::~Neighbor(void)
{
    CHECK(cudaFree(NN));
    CHECK(cudaFree(NL));
}


static __global__ void gpu_find_neighbor
(
    int triclinic, int pbc_x, int pbc_y, int pbc_z, 
    int N, int *Na, int *Na_sum,
    double cutoff_square, const double* __restrict__ box, 
    int *NN, int *NL, double *x, double *y, double *z
)
{
    int N1 = Na_sum[blockIdx.x];
    int N2 = N1 + Na[blockIdx.x];
    int n1 = N1 + threadIdx.x;
    if (n1 < N2)
    {
        const double* __restrict__ h = box + 18 * blockIdx.x;
        double x1 = x[n1];  
        double y1 = y[n1];  
        double z1 = z[n1];
        int count = 0;
        for (int n2 = N1; n2 < N2; ++n2)
        { 
            if (n2 == n1) { continue; }
            double x12 = x[n2]-x1; 
            double y12 = y[n2]-y1; 
            double z12 = z[n2]-z1;
            dev_apply_mic(triclinic, pbc_x, pbc_y, pbc_z, h, x12, y12, z12);
            double distance_square = x12 * x12 + y12 * y12 + z12 * z12;
            if (distance_square < cutoff_square) { NL[count++ * N + n1] = n2; }
        }
        NN[n1] = count;
    }
}


void Neighbor::compute
(int Nc, int N, int *Na, int *Na_sum, double *x, double *y, double *z, Box *box)
{
    int m1 = sizeof(int) * N;
    CHECK(cudaMalloc((void**)&NN, m1));
    CHECK(cudaMalloc((void**)&NL, m1 * MN));
    double rc2 = cutoff * cutoff;
    gpu_find_neighbor<<<Nc, MAX_ATOM_NUMBER>>>
    (
        box->triclinic, box->pbc_x, box->pbc_y, box->pbc_z,
        N, Na, Na_sum, rc2, box->h, NN, NL, x, y, z
    );
    CUDA_CHECK_KERNEL
}


