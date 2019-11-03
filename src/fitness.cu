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


Fitness::Fitness(char* input_dir)
{
    read_potential(input_dir);
    read_weight(input_dir);
    read_xyz_in(input_dir);
    box.read_file(input_dir, Nc);
    neighbor.compute(Nc, N, Na, Na_sum, x, y, z, &box);
    potential.initialize(N, MAX_ATOM_NUMBER);
    MY_MALLOC(error_cpu, float, Nc);
    CHECK(cudaMalloc((void**)&error_gpu, sizeof(float) * Nc));
}


Fitness::~Fitness(void)
{
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
    CHECK(cudaFree(error_gpu));
    MY_FREE(error_cpu);
    MY_FREE(parameters_min);
    MY_FREE(parameters_max);
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


void Fitness::read_weight(char* input_dir)
{
    print_line_1();
    printf("Started reading weight.in.\n");
    print_line_2();

    char file_weight[200];
    strcpy(file_weight, input_dir);
    strcat(file_weight, "/weight.in");
    FILE *fid = my_fopen(file_weight, "r");

    int count = fscanf
    (
        fid, "%f%f%f", &weight.force, &weight.energy, &weight.stress
    );
    if (count != 3) print_error("Reading error for weight.in.\n");

    fclose(fid);

    if (weight.force < 0)
    {
        print_error("weight.force should >= 0\n");
    }
    else
    {
        printf("weight.force = %g.\n", weight.force);
    }

    if (weight.energy < 0)
    {
        print_error("weight.energy should >= 0\n");
    }
    else
    {
        printf("weight.energy = %g.\n", weight.energy);
    }

    if (weight.stress < 0)
    {
        print_error("weight.stress should >= 0\n");
    }
    else
    {
        printf("weight.stress = %g.\n", weight.stress);
    }
}


void Fitness::read_Nc(FILE* fid)
{
    int count = fscanf(fid, "%d%d", &Nc, &NC_FORCE);
    if (count != 2) print_error("Reading error for xyz.in.\n");

    if (Nc < 2)
    {
        print_error("Number of configurations should >= 2\n");
    }

    if (NC_FORCE < 1)
    {
        print_error("Number of force configurations should >= 1\n");
    }
    else if (NC_FORCE > Nc - 1)
    {
        print_error("Number of potential configurations should >= 1\n");
    }

    printf("Number of configurations is %d:\n", Nc);
    printf("    %d force configurations;\n", NC_FORCE);
    printf("    %d energy and virial configurations.\n", Nc - NC_FORCE);
}


void Fitness::read_Na(FILE* fid)
{ 
    CHECK(cudaMallocManaged((void**)&Na, sizeof(int) * Nc));
    CHECK(cudaMallocManaged((void**)&Na_sum, sizeof(int) * Nc));

    N = 0;
    MAX_ATOM_NUMBER = 0;

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
        if (Na[nc] > MAX_ATOM_NUMBER)
        {
            MAX_ATOM_NUMBER = Na[nc];
        }

        if (Na[nc] < 1)
        {
            print_error("Number of atoms %d should >= 1\n");
        }
    }

    for (int nc = 1; nc < Nc; ++nc)
    {
        Na_sum[nc] = Na_sum[nc-1] + Na[nc-1];
    }

    // get the total number of atoms in force configurations
    N_force = 0;
    for (int nc = 0; nc < NC_FORCE; ++nc)
    {
        N_force += Na[nc];
    }
    printf("Total number of atoms is %d:\n", N);
    printf("    %d in force configurations;\n", N_force);
    printf("    %d in energy and virial configurations.\n", N - N_force);
} 


void Fitness::read_xyz(FILE* fid)
{
    int m1 = sizeof(int) * N;
    int m2 = sizeof(float) * N;

    CHECK(cudaMallocManaged((void**)&type, m1));
    CHECK(cudaMallocManaged((void**)&x, m2));
    CHECK(cudaMallocManaged((void**)&y, m2));
    CHECK(cudaMallocManaged((void**)&z, m2));
    CHECK(cudaMallocManaged((void**)&fx, m2));
    CHECK(cudaMallocManaged((void**)&fy, m2));
    CHECK(cudaMallocManaged((void**)&fz, m2));
    CHECK(cudaMallocManaged((void**)&fx_ref, m2));
    CHECK(cudaMallocManaged((void**)&fy_ref, m2));
    CHECK(cudaMallocManaged((void**)&fz_ref, m2));
    CHECK(cudaMallocManaged((void**)&pe, m2));
    CHECK(cudaMallocManaged((void**)&sxx, m2));
    CHECK(cudaMallocManaged((void**)&syy, m2));
    CHECK(cudaMallocManaged((void**)&szz, m2));

    num_types = 0;
    force_square_sum = 0.0;
    for (int n = 0; n < N; n++)
    {
        int count = fscanf(fid, "%d%f%f%f%f%f%f", 
            &(type[n]), &(x[n]), &(y[n]), &(z[n]),
            &(fx_ref[n]), &(fy_ref[n]), &(fz_ref[n]));
        if (count != 7) { print_error("reading error for xyz.in.\n"); }
        if (type[n] > num_types) { num_types = type[n]; }
        if (n < NC_FORCE * MAX_ATOM_NUMBER)
        {
            force_square_sum += fx_ref[n] * fx_ref[n]
                              + fy_ref[n] * fy_ref[n]
                              + fz_ref[n] * fz_ref[n];
        }
    }
    num_types++;
}


void Fitness::read_potential(char* input_dir)
{
    print_line_1();
    printf("Started reading potential.in.\n");
    print_line_2();
    char file[200];
    strcpy(file, input_dir);
    strcat(file, "/potential.in");
    FILE* fid = my_fopen(file, "r");
    int count = fscanf(fid, "%d", &number_of_variables);
    if (count != 1) { print_error("reading error for potential.in."); }
    printf("number of variables = %d\n", number_of_variables);

    MY_MALLOC(parameters_min, float, number_of_variables);
    MY_MALLOC(parameters_max, float, number_of_variables);

    char name[20];

    count = fscanf(fid, "%s%f", name, &neighbor.cutoff);
    if (count != 2) { print_error("reading error for potential.in."); }
    printf("cutoff for neighbor list is %f.\n", neighbor.cutoff);

    for (int n = 0; n <  number_of_variables; ++n)
    {
        count = fscanf
        (fid, "%s%f%f", name, &parameters_min[n], &parameters_max[n]);
        if (count != 3) { print_error("reading error for potential.in."); }
        printf("%15s%15g%15g\n", name, parameters_min[n], parameters_max[n]);
    }
    fclose(fid);
}


void Fitness::compute(int population_size, float* population, float* fitness)
{
    float *parameters;
    MY_MALLOC(parameters, float, number_of_variables);

    for (int n = 0; n < population_size; ++n)
    {
        float* individual = population + n * number_of_variables;
        for (int m = 0; m < number_of_variables; ++m)
        {
            float a = parameters_min[m];
            float b = parameters_max[m] - a;
            parameters[m] = a + b * individual[m];
        }
        potential.update_potential(parameters, num_types);
        potential.find_force
        (
            num_types, Nc, N, Na, Na_sum, MAX_ATOM_NUMBER, type, &box, &neighbor,
            x, y, z, fx, fy, fz, sxx, syy, szz, pe
        );
        fitness[n] = weight.energy * get_fitness_energy();
        fitness[n] += weight.stress * get_fitness_stress();
        fitness[n] += weight.force * get_fitness_force();
    }

    MY_FREE(parameters);
}


void Fitness::predict_energy_or_stress(FILE* fid, float* data, float* ref)
{
    for (int nc = NC_FORCE; nc < Nc; ++nc)
    {
        int offset = Na_sum[nc];
        float data_nc = 0.0;
        for (int m = 0; m < Na[nc]; ++m)
        {
            data_nc += data[offset + m];
        }
        fprintf(fid, "%25.15e%25.15e\n", data_nc, ref[nc]);
    }
}


void Fitness::predict(char* input_dir, float* elite)
{
    float *parameters;
    MY_MALLOC(parameters, float, number_of_variables);
    for (int m = 0; m < number_of_variables; ++m)
    {
        float a = parameters_min[m];
        float b = parameters_max[m] - a;
        parameters[m] = a + b * elite[m];
    }
    potential.update_potential(parameters, num_types);
    potential.find_force
    (
        num_types, Nc, N, Na, Na_sum, MAX_ATOM_NUMBER, type, &box, &neighbor,
        x, y, z, fx, fy, fz, sxx, syy, szz, pe
    );
    MY_FREE(parameters);

    CHECK(cudaDeviceSynchronize()); // needed for CC < 6.0

    char file_force[200];
    strcpy(file_force, input_dir);
    strcat(file_force, "/force.out");
    FILE* fid_force = my_fopen(file_force, "w");
    for (int n = 0; n < N_force; ++n)
    {
        fprintf
        (
            fid_force, "%25.15e%25.15e%25.15e%25.15e%25.15e%25.15e\n", 
            fx[n], fy[n], fz[n], fx_ref[n], fy_ref[n], fz_ref[n]
        );
    }
    fclose(fid_force);

    char file[200];
    strcpy(file, input_dir);
    strcat(file, "/prediction.out");
    FILE* fid_prediction = my_fopen(file, "w");
    predict_energy_or_stress(fid_prediction, pe, box.pe_ref);
    predict_energy_or_stress(fid_prediction, sxx, box.sxx_ref);
    predict_energy_or_stress(fid_prediction, syy, box.syy_ref);
    predict_energy_or_stress(fid_prediction, szz, box.szz_ref);
    fclose(fid_prediction);
}


static __global__ void gpu_sum_force_error
(
    int N, float *g_fx, float *g_fy, float *g_fz, 
    float *g_fx_ref, float *g_fy_ref, float *g_fz_ref, float *g_error
)
{
    int tid = threadIdx.x;
    int number_of_rounds = (N - 1) / blockDim.x + 1; 
    extern __shared__ float s_error[];
    s_error[tid] = 0.0;
    for (int round = 0; round < number_of_rounds; ++round)
    {
        int n = tid + round * blockDim.x;
        if (n < N)
        {
            float dx = g_fx[n] - g_fx_ref[n];
            float dy = g_fy[n] - g_fy_ref[n];
            float dz = g_fz[n] - g_fz_ref[n];
            s_error[tid] += dx * dx + dy * dy + dz * dz;
        }
    }

    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 32; offset >>= 1)
    {
        if (tid < offset) { s_error[tid] += s_error[tid + offset]; }
        __syncthreads();
    }

    for (int offset = 32; offset > 0; offset >>= 1)
    {
        if (tid < offset) { s_error[tid] += s_error[tid + offset]; }
        __syncwarp();
    }

    if (tid ==  0) { g_error[0] = s_error[0]; }
}


float Fitness::get_fitness_force(void)
{
    gpu_sum_force_error<<<1, 512, sizeof(float) * 512>>>
    (N_force, fx, fy, fz, fx_ref, fy_ref, fz_ref, error_gpu);
    CHECK(cudaMemcpy(error_cpu, error_gpu, sizeof(float), 
        cudaMemcpyDeviceToHost));
    return sqrt(error_cpu[0] / force_square_sum);
}


static __global__ void gpu_sum_pe_error
(int *g_Na, int *g_Na_sum, float *g_pe, float *g_pe_ref, float *error_gpu)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int Na = g_Na[bid];
    int offset = g_Na_sum[bid];
    extern __shared__ float s_pe[];
    s_pe[tid] = 0.0;
    if (tid < Na)
    {
        int n = offset + tid; // particle index
        s_pe[tid] += g_pe[n];
    }
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 32; offset >>= 1)
    {
        if (tid < offset) { s_pe[tid] += s_pe[tid + offset]; }
        __syncthreads();
    }

    for (int offset = 32; offset > 0; offset >>= 1)
    {
        if (tid < offset) { s_pe[tid] += s_pe[tid + offset]; }
        __syncwarp();
    }

    if (tid == 0)
    {
        float diff = s_pe[0] - g_pe_ref[bid];
        error_gpu[bid] = diff * diff;
    }
}


static int get_block_size(int max_num_atom)
{
    int block_size = 64;
    for (int n = 64; n < 1024; n <<= 1 )
    {
        if (max_num_atom > n)
        {
            block_size = n << 1;
        }
    }
    return block_size;
}


float Fitness::get_fitness_energy(void)
{
    int block_size = get_block_size(MAX_ATOM_NUMBER);
    gpu_sum_pe_error<<<Nc, block_size, sizeof(float) * block_size>>>
    (Na, Na_sum, pe, box.pe_ref, error_gpu);
    int mem = sizeof(float) * Nc;
    CHECK(cudaMemcpy(error_cpu, error_gpu, mem, cudaMemcpyDeviceToHost));
    float error_ave = 0.0;
    for (int n = NC_FORCE; n < Nc; ++n)
    {
        error_ave += error_cpu[n];
    }
    return sqrt(error_ave / box.potential_square_sum);
}


float Fitness::get_fitness_stress(void)
{
    float error_ave = 0.0;
    int mem = sizeof(float) * Nc;
    int block_size = get_block_size(MAX_ATOM_NUMBER);

    gpu_sum_pe_error<<<Nc, block_size, sizeof(float) * block_size>>>
    (Na, Na_sum, sxx, box.sxx_ref, error_gpu);
    CHECK(cudaMemcpy(error_cpu, error_gpu, mem, cudaMemcpyDeviceToHost));
    for (int n = NC_FORCE; n < Nc; ++n) {error_ave += error_cpu[n];}

    gpu_sum_pe_error<<<Nc, block_size, sizeof(float) * block_size>>>
    (Na, Na_sum, syy, box.syy_ref, error_gpu);
    CHECK(cudaMemcpy(error_cpu, error_gpu, mem, cudaMemcpyDeviceToHost));
    for (int n = NC_FORCE; n < Nc; ++n) {error_ave += error_cpu[n];}

    gpu_sum_pe_error<<<Nc, block_size, sizeof(float) * block_size>>>
    (Na, Na_sum, szz, box.szz_ref, error_gpu);
    CHECK(cudaMemcpy(error_cpu, error_gpu, mem, cudaMemcpyDeviceToHost));
    for (int n = NC_FORCE; n < Nc; ++n) {error_ave += error_cpu[n];}

    return sqrt(error_ave / box.virial_square_sum);
}


