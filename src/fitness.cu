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
    read_train_in(input_dir);
    neighbor.compute(Nc, N, max_Na, Na, Na_sum, r, h);
    potential.initialize(N, max_Na);
    MY_MALLOC(error_cpu, float, Nc);
    CHECK(cudaMalloc((void**)&error_gpu, sizeof(float) * Nc));
}


Fitness::~Fitness(void)
{
    CHECK(cudaFree(h));
    CHECK(cudaFree(pe_ref)); 
    CHECK(cudaFree(sxx_ref));
    CHECK(cudaFree(syy_ref));
    CHECK(cudaFree(szz_ref));
    cudaFree(Na);
    cudaFree(Na_sum);
    cudaFree(type);
    cudaFree(r);
    cudaFree(force_ref);
    cudaFree(pe);
    cudaFree(virial);
    cudaFree(force);
    CHECK(cudaFree(error_gpu));
    MY_FREE(error_cpu);
    MY_FREE(parameters_min);
    MY_FREE(parameters_max);
}


static void get_inverse(float *cpu_h)
{
    cpu_h[9]  = cpu_h[4]*cpu_h[8] - cpu_h[5]*cpu_h[7];
    cpu_h[10] = cpu_h[2]*cpu_h[7] - cpu_h[1]*cpu_h[8];
    cpu_h[11] = cpu_h[1]*cpu_h[5] - cpu_h[2]*cpu_h[4];
    cpu_h[12] = cpu_h[5]*cpu_h[6] - cpu_h[3]*cpu_h[8];
    cpu_h[13] = cpu_h[0]*cpu_h[8] - cpu_h[2]*cpu_h[6];
    cpu_h[14] = cpu_h[2]*cpu_h[3] - cpu_h[0]*cpu_h[5];
    cpu_h[15] = cpu_h[3]*cpu_h[7] - cpu_h[4]*cpu_h[6];
    cpu_h[16] = cpu_h[1]*cpu_h[6] - cpu_h[0]*cpu_h[7];
    cpu_h[17] = cpu_h[0]*cpu_h[4] - cpu_h[1]*cpu_h[3];
    float volume = cpu_h[0] * (cpu_h[4]*cpu_h[8] - cpu_h[5]*cpu_h[7])
                 + cpu_h[1] * (cpu_h[5]*cpu_h[6] - cpu_h[3]*cpu_h[8])
                 + cpu_h[2] * (cpu_h[3]*cpu_h[7] - cpu_h[4]*cpu_h[6]);
    for (int n = 9; n < 18; n++) { cpu_h[n] /= volume; }
}


void Fitness::read_train_in(char* input_dir)
{
    print_line_1();
    printf("Started reading train.in.\n");
    print_line_2();

    char file_train[200];
    strcpy(file_train, input_dir);
    strcat(file_train, "/train.in");
    FILE *fid = my_fopen(file_train, "r");

    // get Nc and Nc_force
    read_Nc(fid);
    CHECK(cudaMallocManaged((void**)&h, sizeof(float) * Nc * 18));
    CHECK(cudaMallocManaged((void**)&pe_ref, sizeof(float) * Nc));
    CHECK(cudaMallocManaged((void**)&sxx_ref, sizeof(float) * Nc));
    CHECK(cudaMallocManaged((void**)&syy_ref, sizeof(float) * Nc));
    CHECK(cudaMallocManaged((void**)&szz_ref, sizeof(float) * Nc));
    CHECK(cudaMallocManaged((void**)&Na, sizeof(int) * Nc));
    CHECK(cudaMallocManaged((void**)&Na_sum, sizeof(int) * Nc));

    N = Nc * 64; // TODO: to be improved
    CHECK(cudaMallocManaged((void**)&type, sizeof(int) * N));
    CHECK(cudaMallocManaged((void**)&r, sizeof(float) * N * 3));
    CHECK(cudaMallocManaged((void**)&force, sizeof(float) * N * 3));
    CHECK(cudaMallocManaged((void**)&force_ref, sizeof(float) * N * 3));
    CHECK(cudaMallocManaged((void**)&pe, sizeof(float) * N));
    CHECK(cudaMallocManaged((void**)&virial, sizeof(float) * N * 6));

    float energy_minimum = 0.0;
    num_types = 0;
    potential_square_sum = 0.0;
    virial_square_sum = 0.0;
    force_square_sum = 0.0;
    for (int nc = 0; nc < Nc; ++nc)
    {
        Na_sum[nc] = 0;
    }

    for (int n = 0; n < Nc; ++n)
    {
        int count; 

        // Na, energy, virial
        float tmp; // TODO
        if (n < Nc_force)
        {
            count = fscanf(fid, "%d", &Na[n]);
            if (count != 1) { print_error("reading error for train.in.\n"); }
        }
        else
        {
            count = fscanf
            (
                fid, "%d%f%f%f%f%f%f%f", &Na[n], &pe_ref[n],
                &sxx_ref[n], &syy_ref[n], &szz_ref[n], 
                &tmp, &tmp, &tmp
            );
            if (count != 8) { print_error("reading error for train.in.\n"); }
            if (pe_ref[n] < energy_minimum) energy_minimum = pe_ref[n];
        }

        if (n > 0)
        {
            Na_sum[n] = Na_sum[n-1] + Na[n-1];
        }

        // box
        for (int k = 0; k < 9; ++k)
        {
            count = fscanf(fid, "%f", &h[k + 18 * n]);
            if (count != 1) { print_error("reading error for train.in.\n"); }
        }
        get_inverse(h + 18 * n);

        // type, position, force
        for (int k = 0; k < Na[n]; ++k)
        {
            if (n < Nc_force)
            {
                count = fscanf
                (
                    fid, "%d%f%f%f%f%f%f", 
                    &type[Na_sum[n] + k], 
                    &r[Na_sum[n] + k], 
                    &r[Na_sum[n] + k + N],
                    &r[Na_sum[n] + k + N * 2],
                    &force_ref[Na_sum[n] + k], 
                    &force_ref[Na_sum[n] + k + N],
                    &force_ref[Na_sum[n] + k + N * 2]
                );
                if (count != 7) { print_error("reading error for train.in.\n"); }

                force_square_sum += force_ref[Na_sum[n] + k] * force_ref[Na_sum[n] + k]
                    + force_ref[Na_sum[n] + k + N] * force_ref[Na_sum[n] + k + N]
                    + force_ref[Na_sum[n] + k + N * 2] * force_ref[Na_sum[n] + k + N * 2];
            }
            else
            {
                count = fscanf
                (
                    fid, "%d%f%f%f", 
                    &type[Na_sum[n] + k], 
                    &r[Na_sum[n] + k], 
                    &r[Na_sum[n] + k + N],
                    &r[Na_sum[n] + k + N * 2]
                );
                if (count != 4) { print_error("reading error for train.in.\n"); }
            }

            if (type[Na_sum[n] + k] > num_types) { num_types = type[Na_sum[n] + k]; }
        }
    }

    fclose(fid);

    ++num_types;

    for (int n = Nc_force; n < Nc; ++n)
    {
        float energy = pe_ref[n] - energy_minimum;
        potential_square_sum += energy * energy;
        virial_square_sum += sxx_ref[n] * sxx_ref[n]
                               + syy_ref[n] * syy_ref[n]
                               + szz_ref[n] * szz_ref[n];
    }

    // N, max_N
    N = 0;
    max_Na = 0;

    for (int nc = 0; nc < Nc; ++nc)
    {
        N += Na[nc];
        if (Na[nc] > max_Na)
        {
            max_Na = Na[nc];
        }
        if (Na[nc] < 1)
        {
            print_error("Number of atoms %d should >= 1\n");
        }
    }

    // get the total number of atoms in force configurations
    N_force = 0;
    for (int nc = 0; nc < Nc_force; ++nc)
    {
        N_force += Na[nc];
    }
    printf("Total number of atoms is %d:\n", N);
    printf("    %d atoms in the largest configuration;\n", max_Na);
    printf("    %d atoms in force configurations;\n", N_force);
    printf("    %d atoms in energy-virial configurations.\n", N - N_force);
}


void Fitness::read_Nc(FILE* fid)
{
    int count = fscanf(fid, "%d%d", &Nc, &Nc_force);
    if (count != 2) print_error("Reading error for xyz.in.\n");

    if (Nc < 2)
    {
        print_error("Number of configurations should >= 2\n");
    }

    if (Nc_force < 1)
    {
        print_error("Number of force configurations should >= 1\n");
    }
    else if (Nc_force > Nc - 1)
    {
        print_error("Number of potential configurations should >= 1\n");
    }

    printf("Number of configurations is %d:\n", Nc);
    printf("    %d force configurations;\n", Nc_force);
    printf("    %d energy-virial configurations.\n", Nc - Nc_force);
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

    char name[20];

    int count = fscanf(fid, "%s%d", name, &potential_type);
    if (count != 2) { print_error("reading error for potential.in."); }
    if (potential_type == 1)
    {
        number_of_variables = 9;
        printf
        (
            "Use one-element mini-Tersoff potential with %d parameters.\n",
            number_of_variables
        );
    }
    else if (potential_type == 2)
    {
        number_of_variables = 37;
        printf
        (
            "Use two-element mini-Tersoff potential with %d parameters.\n",
            number_of_variables
        );
    }
    else
    {
        print_error("unsupported potential type.\n");
    }

    MY_MALLOC(parameters_min, float, number_of_variables);
    MY_MALLOC(parameters_max, float, number_of_variables);

    count = fscanf(fid, "%s%f", name, &neighbor.cutoff);
    if (count != 2) { print_error("reading error for potential.in."); }
    printf("cutoff for neighbor list is %g A.\n", neighbor.cutoff);

    count = fscanf(fid, "%s%f", name, &weight.force);
    if (count != 2) { print_error("reading error for potential.in."); }
    if (weight.force < 0)
    {
        print_error("weight for force should >= 0\n");
    }
    printf("weight for force is %g.\n", weight.force);

    count = fscanf(fid, "%s%f", name, &weight.energy);
    if (count != 2) { print_error("reading error for potential.in."); }
    if (weight.energy < 0)
    {
        print_error("weight for energy should >= 0\n");
    }
    printf("weight for energy is %g.\n", weight.energy);

    count = fscanf(fid, "%s%f", name, &weight.stress);
    if (count != 2) { print_error("reading error for potential.in."); }
    if (weight.stress < 0)
    {
        print_error("weight for stress should >= 0\n");
    }
    printf("weight for stress is %g.\n", weight.stress);

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
            num_types, Nc, N, Na, Na_sum, max_Na, type, h, &neighbor,
            r, force, virial, pe
        );
        fitness[n] = weight.energy * get_fitness_energy();
        fitness[n] += weight.stress * get_fitness_stress();
        fitness[n] += weight.force * get_fitness_force();
    }

    MY_FREE(parameters);
}


void Fitness::predict_energy_or_stress(FILE* fid, float* data, float* ref)
{
    for (int nc = Nc_force; nc < Nc; ++nc)
    {
        int offset = Na_sum[nc];
        float data_nc = 0.0;
        for (int m = 0; m < Na[nc]; ++m)
        {
            data_nc += data[offset + m];
        }
        fprintf(fid, "%g %g\n", data_nc, ref[nc]);
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
        num_types, Nc, N, Na, Na_sum, max_Na, type, h, &neighbor,
        r, force, virial, pe
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
            fid_force, "%g %g %g %g %g %g\n", 
            force[n], force[n+N], force[n+N*2], 
            force_ref[n], force_ref[n+N], force_ref[n+N*2]
        );
    }
    fclose(fid_force);

    char file[200];
    strcpy(file, input_dir);
    strcat(file, "/prediction.out");
    FILE* fid_prediction = my_fopen(file, "w");
    predict_energy_or_stress(fid_prediction, pe, pe_ref);
    predict_energy_or_stress(fid_prediction, virial, sxx_ref);
    predict_energy_or_stress(fid_prediction, virial+N, syy_ref);
    predict_energy_or_stress(fid_prediction, virial+N*2, szz_ref);
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
    s_error[tid] = 0.0f;
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
    (
        N_force, force, force+N, force+N*2, 
        force_ref, force_ref+N, force_ref+N*2, error_gpu
    );
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
    s_pe[tid] = 0.0f;
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
    int block_size = get_block_size(max_Na);
    gpu_sum_pe_error<<<Nc, block_size, sizeof(float) * block_size>>>
    (Na, Na_sum, pe, pe_ref, error_gpu);
    int mem = sizeof(float) * Nc;
    CHECK(cudaMemcpy(error_cpu, error_gpu, mem, cudaMemcpyDeviceToHost));
    float error_ave = 0.0;
    for (int n = Nc_force; n < Nc; ++n)
    {
        error_ave += error_cpu[n];
    }
    return sqrt(error_ave / potential_square_sum);
}


float Fitness::get_fitness_stress(void)
{
    float error_ave = 0.0;
    int mem = sizeof(float) * Nc;
    int block_size = get_block_size(max_Na);

    gpu_sum_pe_error<<<Nc, block_size, sizeof(float) * block_size>>>
    (Na, Na_sum, virial, sxx_ref, error_gpu);
    CHECK(cudaMemcpy(error_cpu, error_gpu, mem, cudaMemcpyDeviceToHost));
    for (int n = Nc_force; n < Nc; ++n) {error_ave += error_cpu[n];}

    gpu_sum_pe_error<<<Nc, block_size, sizeof(float) * block_size>>>
    (Na, Na_sum, virial+N, syy_ref, error_gpu);
    CHECK(cudaMemcpy(error_cpu, error_gpu, mem, cudaMemcpyDeviceToHost));
    for (int n = Nc_force; n < Nc; ++n) {error_ave += error_cpu[n];}

    gpu_sum_pe_error<<<Nc, block_size, sizeof(float) * block_size>>>
    (Na, Na_sum, virial+N*2, szz_ref, error_gpu);
    CHECK(cudaMemcpy(error_cpu, error_gpu, mem, cudaMemcpyDeviceToHost));
    for (int n = Nc_force; n < Nc; ++n) {error_ave += error_cpu[n];}

    return sqrt(error_ave / virial_square_sum);
}


