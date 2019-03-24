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
Get the fitness
------------------------------------------------------------------------------*/


#include "fitness.cuh"
#include "error.cuh"
#include "read_file.cuh"
#define BLOCK_SIZE 128


Fitness::Fitness(void)
{
    // nothing now
}


Fitness::~Fitness(void)
{
    // nothing now
}


void Fitness::compute
(
    int population_size, int number_of_variables, 
    double* population, double* fitness
)
{
    // a test function y = x1^2 + x2^2 + ... with solution x1 = x2 = ... = 0
    for (int n = 0; n < population_size; ++n)
    {
        double* individual = population + n * number_of_variables;
        double sum = 0.0;
        for (int m = 0; m < number_of_variables; ++m)
        {
            double tmp = (individual[m] * 2.0 - 1);
            sum += tmp * tmp;
        }
        fitness[n] = sum;
    }
}



void Fitness::get_fitness_population
(
    int population_size, int number_of_variables, 
    double* population, double* fitness
)
{
    for (int n = 0; n < population_size; ++n)
    {
        double* individual = population + n * number_of_variables;
        for (int m = 0; m < number_of_variables; ++m)
        {
            double a = potential_parameters_min[m];
            double b = potential_parameters_max[m] - a;
            potential_parameters[m] = a + b * individual[m];
        }
        find_force();
        fitness[n] = get_fitness_force();
    }
}


static __device__ void warp_reduce(volatile double* s, int t) 
{
    s[t] += s[t + 32]; s[t] += s[t + 16]; s[t] += s[t + 8];
    s[t] += s[t + 4];  s[t] += s[t + 2];  s[t] += s[t + 1];
}


static __global__ void gpu_sum_force_error
(
    int N, double *g_fx, double *g_fy, double *g_fz, 
    double *g_fx_ref, double *g_fy_ref, double *g_fz_ref, double *g_error
)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int number_of_patches = (N - 1) / 1024 + 1; 
    __shared__ double s_error[1024];
    s_error[tid] = 0.0;
    for (int patch = 0; patch < number_of_patches; ++patch)
    {
        int n = tid + patch * 1024;
        if (n < N) 
        {
            double dx = g_fx[n] - g_fx_ref[n];
            double dy = g_fy[n] - g_fy_ref[n];
            double dz = g_fz[n] - g_fz_ref[n];
            s_error[tid] += dx * dx + dy * dy + dz * dz;
        }
    }
    __syncthreads();
    if (tid < 512) s_error[tid] += s_error[tid + 512]; __syncthreads();
    if (tid < 256) s_error[tid] += s_error[tid + 256]; __syncthreads();
    if (tid < 128) s_error[tid] += s_error[tid + 128]; __syncthreads();
    if (tid <  64) s_error[tid] += s_error[tid + 64];  __syncthreads();
    if (tid <  32) warp_reduce(s_error, tid);
    if (tid ==  0) { g_error[bid] = s_error[0]; }
}


double Fitness::get_fitness_force(void)
{
    gpu_sum_force_error<<<1, 1024>>>(N, fx, fy, fz, 
        fx_ref, fy_ref, fz_ref, force_error_gpu);
    CHECK(cudaMemcpy(force_error_cpu, force_error_gpu, sizeof(double), 
        cudaMemcpyDeviceToHost));
    force_error_cpu[0] /= force_ref_square_sum;
    return force_error_cpu[0];
}


