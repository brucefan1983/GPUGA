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
#include "neighbor.cuh"
#include "error.cuh"
#include "read_file.cuh"
#include "common.cuh"


Fitness::Fitness(char* input_dir)
{
    read_xyz_in(input_dir);
    box.read_file(input_dir, Nc);
    neighbor.compute(Nc, N, Na, Na_sum, x, y, z, &box);
}


Fitness::~Fitness(void)
{
    MY_FREE(cpu_fx_ref);
    MY_FREE(cpu_fy_ref);
    MY_FREE(cpu_fz_ref);
    MY_FREE(cpu_fx);
    MY_FREE(cpu_fy);
    MY_FREE(cpu_fz);
    cudaFree(Na);
    cudaFree(Na_sum);
    cudaFree(type);
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    cudaFree(fx_ref);
    cudaFree(fy_ref);
    cudaFree(fz_ref);
    cudaFree(pe);
    cudaFree(sxx);
    cudaFree(syy);
    cudaFree(szz);
    cudaFree(fx);
    cudaFree(fy);
    cudaFree(fz);
    cudaFree(b);
    cudaFree(bp);
    cudaFree(f12x);
    cudaFree(f12y);
    cudaFree(f12z);
}


void Fitness::read_xyz_in(char* input_dir)
{
    print_line_1();
    printf("Started reading xyz.in.\n");
    print_line_2();
    char file_xyz[200];
    strcpy(file_xyz, input_dir);
    strcat(file_xyz, "/xyz.in");
    FILE *fid_xyz = my_fopen(file_xyz, "r");
    read_Nc(fid_xyz);
    read_Na(fid_xyz);
    read_xyz(fid_xyz);
    fclose(fid_xyz);
}


void Fitness::read_Nc(FILE* fid)
{
    int count = fscanf(fid, "%d", &Nc);
    if (count != 1) print_error("Reading error for xyz.in.\n");
    if (Nc < 1)
        print_error("Number of configurations should >= 1\n");
    else
        printf("Number of configurations = %d.\n", Nc);
}


void Fitness::read_Na(FILE* fid)
{ 
    CHECK(cudaMallocManaged((void**)&Na, sizeof(int) * Nc));
    CHECK(cudaMallocManaged((void**)&Na_sum, sizeof(int) * Nc));

    N = 0;

    for (int nc = 0; nc < Nc; ++nc)
    {
        Na_sum[nc] = 0;
    }

    for (int nc = 0; nc < Nc; ++nc)
    {
        int count = fscanf(fid, "%d", &Na[nc]);

        if (count != 1)
        {
            print_error("Reading error for xyz.in.\n");
        }

        N += Na[nc];

        if (Na[nc] < 1)
        {
            print_error("Number of atoms %d should >= 1\n");
        }
        else
        {
            printf("N[%d] = %d.\n", nc, Na[nc]);
        }
    }

    for (int nc = 1; nc < Nc; ++nc)
    {
        Na_sum[nc] = Na_sum[nc-1] + Na[nc-1];
    }
} 


void Fitness::read_xyz(FILE* fid)
{
    int m1 = sizeof(int) * N;
    int m2 = sizeof(double) * N;

    CHECK(cudaMallocManaged((void**)&type, m1));
    CHECK(cudaMallocManaged((void**)&x, m2));
    CHECK(cudaMallocManaged((void**)&y, m2));
    CHECK(cudaMallocManaged((void**)&z, m2));

    MY_MALLOC(cpu_fx_ref, double, N);
    MY_MALLOC(cpu_fy_ref, double, N);
    MY_MALLOC(cpu_fz_ref, double, N);
    MY_MALLOC(cpu_fx, double, N);
    MY_MALLOC(cpu_fy, double, N);
    MY_MALLOC(cpu_fz, double, N);
    num_types = 0;
    force_square_sum = 0.0;
    for (int n = 0; n < N; n++)
    {
        int count = fscanf(fid, "%d%lf%lf%lf%lf%lf%lf", 
            &(type[n]), &(x[n]), &(y[n]), &(z[n]),
            &(cpu_fx_ref[n]), &(cpu_fy_ref[n]), &(cpu_fz_ref[n]));
        if (count != 7) { print_error("reading error for xyz.in.\n"); }
        if (type[n] > num_types) { num_types = type[n]; }
        if (n < NC_FORCE * MAX_ATOM_NUMBER)
        {
            force_square_sum += cpu_fx_ref[n] * cpu_fx_ref[n]
                              + cpu_fy_ref[n] * cpu_fy_ref[n]
                              + cpu_fz_ref[n] * cpu_fz_ref[n];
        }
    }
    num_types++;

    allocate_memory_gpu();
    CHECK(cudaMemcpy(fx_ref, cpu_fx_ref, m2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(fy_ref, cpu_fy_ref, m2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(fz_ref, cpu_fz_ref, m2, cudaMemcpyHostToDevice));
}


void Fitness::allocate_memory_gpu(void)
{
    int m2 = sizeof(double) * N;
    // read from CPU
    CHECK(cudaMalloc((void**)&fx_ref, m2));
    CHECK(cudaMalloc((void**)&fy_ref, m2));
    CHECK(cudaMalloc((void**)&fz_ref, m2));
    // Calculated on the GPU
    CHECK(cudaMalloc((void**)&pe, m2));
    CHECK(cudaMalloc((void**)&sxx, m2));
    CHECK(cudaMalloc((void**)&syy, m2));
    CHECK(cudaMalloc((void**)&szz, m2));
    CHECK(cudaMalloc((void**)&fx, m2));
    CHECK(cudaMalloc((void**)&fy, m2));
    CHECK(cudaMalloc((void**)&fz, m2));
    CHECK(cudaMalloc((void**)&b, m2 * MAX_ATOM_NUMBER));
    CHECK(cudaMalloc((void**)&bp, m2 * MAX_ATOM_NUMBER));
    CHECK(cudaMalloc((void**)&f12x, m2 * MAX_ATOM_NUMBER));
    CHECK(cudaMalloc((void**)&f12y, m2 * MAX_ATOM_NUMBER));
    CHECK(cudaMalloc((void**)&f12z, m2 * MAX_ATOM_NUMBER));
}


void Fitness::compute
(
    int population_size, int number_of_variables, 
    double *parameters_min, double *parameters_max,
    double* population, double* fitness
)
{
    double *parameters;
    double *error_gpu, *error_cpu;
    MY_MALLOC(error_cpu, double, Nc);
    CHECK(cudaMalloc((void**)&error_gpu, sizeof(double) * Nc));
    MY_MALLOC(parameters, double, number_of_variables);
    for (int n = 0; n < population_size; ++n)
    {
        double* individual = population + n * number_of_variables;
        for (int m = 0; m < number_of_variables; ++m)
        {
            double a = parameters_min[m];
            double b = parameters_max[m] - a;
            parameters[m] = a + b * individual[m];
        }
        update_potential(parameters);
        find_force();
        fitness[n] = WEIGHT_ENERGY * get_fitness_energy(error_cpu, error_gpu);
        fitness[n] += WEIGHT_STRESS * get_fitness_stress(error_cpu, error_gpu);
        fitness[n] += WEIGHT_FORCE * get_fitness_force(error_cpu, error_gpu);
    }
    MY_FREE(parameters);
    MY_FREE(error_cpu);
    CHECK(cudaFree(error_gpu));
}


static void predict_energy_or_stress
(FILE* fid, double* cpu_data, double* data, double* ref, int N, int Nc)
{
    cudaMemcpy(cpu_data, data, sizeof(double)*N, cudaMemcpyDeviceToHost);
    for (int nc = NC_FORCE; nc < Nc; ++nc)
    {
        int offset = nc * MAX_ATOM_NUMBER;
        double cpu_data_nc = 0.0;
        for (int m = 0; m < MAX_ATOM_NUMBER; ++m)
        {
            cpu_data_nc += cpu_data[offset + m];
        }
        fprintf(fid, "%25.15e%25.15e\n", cpu_data_nc, ref[nc]);
    }
}


void Fitness::predict
(
    char* input_dir, int number_of_variables, double *parameters_min, 
    double *parameters_max, double* elite
)
{
    double *parameters;
    MY_MALLOC(parameters, double, number_of_variables);
    for (int m = 0; m < number_of_variables; ++m)
    {
        double a = parameters_min[m];
        double b = parameters_max[m] - a;
        parameters[m] = a + b * elite[m];
    }
    update_potential(parameters);
    find_force();
    MY_FREE(parameters);

    cudaMemcpy(cpu_fx, fx, sizeof(double)*N, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_fy, fy, sizeof(double)*N, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_fz, fz, sizeof(double)*N, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_fx_ref, fx_ref, sizeof(double)*N, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_fy_ref, fy_ref, sizeof(double)*N, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_fz_ref, fz_ref, sizeof(double)*N, cudaMemcpyDeviceToHost);

    char file_force[200];
    strcpy(file_force, input_dir);
    strcat(file_force, "/force.out");
    FILE* fid_force = my_fopen(file_force, "w");
    for (int n = 0; n < NC_FORCE*64; ++n)
    {
        fprintf(fid_force, "%25.15e%25.15e%25.15e%25.15e%25.15e%25.15e\n", 
            cpu_fx[n], cpu_fy[n], cpu_fz[n], 
            cpu_fx_ref[n], cpu_fy_ref[n], cpu_fz_ref[n]);
    }
    fclose(fid_force);

    char file[200];
    strcpy(file, input_dir);
    strcat(file, "/prediction.out");
    FILE* fid_prediction = my_fopen(file, "w");
    double *cpu_prediction; MY_MALLOC(cpu_prediction, double, N);
    // energy
    predict_energy_or_stress
    (fid_prediction, cpu_prediction, pe, box.cpu_pe_ref, N, Nc);
    // sxx
    predict_energy_or_stress
    (fid_prediction, cpu_prediction, sxx, box.cpu_sxx_ref, N, Nc);
    // syy
    predict_energy_or_stress
    (fid_prediction, cpu_prediction, syy, box.cpu_syy_ref, N, Nc);
    // szz
    predict_energy_or_stress
    (fid_prediction, cpu_prediction, szz, box.cpu_szz_ref, N, Nc);
    fclose(fid_prediction);
    MY_FREE(cpu_prediction);
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
    int number_of_patches = (N - 1) / 512 + 1; 
    __shared__ double s_error[512];
    s_error[tid] = 0.0;
    for (int patch = 0; patch < number_of_patches; ++patch)
    {
        int n = tid + patch * 512;
        if (n < N)
        {
            double dx = g_fx[n] - g_fx_ref[n];
            double dy = g_fy[n] - g_fy_ref[n];
            double dz = g_fz[n] - g_fz_ref[n];
            s_error[tid] += dx*dx + dy*dy + dz*dz;
        }
    }
    __syncthreads();
    if (tid < 256) s_error[tid] += s_error[tid + 256]; __syncthreads();
    if (tid < 128) s_error[tid] += s_error[tid + 128]; __syncthreads();
    if (tid <  64) s_error[tid] += s_error[tid + 64];  __syncthreads();
    if (tid <  32) warp_reduce(s_error, tid);
    if (tid ==  0) { g_error[0] = s_error[0]; }
}


double Fitness::get_fitness_force(double *error_cpu, double *error_gpu)
{
    int M = NC_FORCE * MAX_ATOM_NUMBER;
    gpu_sum_force_error<<<1, 512>>>(M, fx, fy, fz, 
        fx_ref, fy_ref, fz_ref, error_gpu);
    CHECK(cudaMemcpy(error_cpu, error_gpu, sizeof(double), 
        cudaMemcpyDeviceToHost));
    return sqrt(error_cpu[0] / force_square_sum);
}


static __global__ void gpu_sum_pe_error
(int *g_Na, int *g_Na_sum, double *g_pe, double *g_pe_ref, double *error_gpu)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int Na = g_Na[bid];
    int offset = g_Na_sum[bid];
    __shared__ double s_pe[64];
    s_pe[tid] = 0.0;
    if (tid < Na)
    {
        int n = offset + tid; // particle index
        s_pe[tid] += g_pe[n];
    }
    __syncthreads();
    if (tid < 32) { warp_reduce(s_pe, tid); }
    if (tid == 0)
    {
        double diff = s_pe[0] - g_pe_ref[bid];
        error_gpu[bid] = diff * diff;
    }
}


double Fitness::get_fitness_energy(double* error_cpu, double* error_gpu)
{
    gpu_sum_pe_error<<<Nc, 64>>>(Na, Na_sum, pe, box.pe_ref, error_gpu);
    int mem = sizeof(double) * Nc;
    CHECK(cudaMemcpy(error_cpu, error_gpu, mem, cudaMemcpyDeviceToHost));
    double error_ave = 0.0;
    for (int n = NC_FORCE; n < Nc; ++n)
    {
        error_ave += error_cpu[n];
    }
    return sqrt(error_ave / box.potential_square_sum);
}


double Fitness::get_fitness_stress(double* error_cpu, double* error_gpu)
{
    double error_ave = 0.0;
    int mem = sizeof(double) * Nc;

    gpu_sum_pe_error<<<Nc, 64>>>(Na, Na_sum, sxx, box.sxx_ref, error_gpu);
    CHECK(cudaMemcpy(error_cpu, error_gpu, mem, cudaMemcpyDeviceToHost));
    for (int n = NC_FORCE; n < Nc; ++n) {error_ave += error_cpu[n];}

    gpu_sum_pe_error<<<Nc, 64>>>(Na, Na_sum, syy, box.syy_ref, error_gpu);
    CHECK(cudaMemcpy(error_cpu, error_gpu, mem, cudaMemcpyDeviceToHost));
    for (int n = NC_FORCE; n < Nc; ++n) {error_ave += error_cpu[n];}

    gpu_sum_pe_error<<<Nc, 64>>>(Na, Na_sum, szz, box.szz_ref, error_gpu);
    CHECK(cudaMemcpy(error_cpu, error_gpu, mem, cudaMemcpyDeviceToHost));
    for (int n = NC_FORCE; n < Nc; ++n) {error_ave += error_cpu[n];}

    return sqrt(error_ave / box.virial_square_sum);
}


