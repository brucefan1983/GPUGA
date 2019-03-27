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
#include "mic.cuh"
#include "error.cuh"
#include "read_file.cuh"
#define BLOCK_SIZE 128
#define NUM_PARAMS 19 


Fitness::Fitness(char* input_dir)
{
    read_xyz_in(input_dir);
    read_box(input_dir);
    find_neighbor();

// test the force
find_force();
double *cpu_fx;
MY_MALLOC(cpu_fx, double, N);
cudaMemcpy(cpu_fx, fx, sizeof(double)*N, cudaMemcpyDeviceToHost);
FILE* fid = my_fopen("f.out", "w");
for (int n = 0; n < N; ++n)
{
    fprintf(fid, "%20.10f\n", cpu_fx[n]);
}
cudaMemcpy(cpu_fx, fx_ref, sizeof(double)*N, cudaMemcpyDeviceToHost);
for (int n = 0; n < N; ++n)
{
    fprintf(fid, "%20.10f\n", cpu_fx[n]);
}
fclose(fid);
// test the force over
}


Fitness::~Fitness(void)
{
    MY_FREE(cpu_ters);
    cudaFree(ters);
    cudaFree(Na);
    cudaFree(Na_sum);
    cudaFree(type);
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    cudaFree(fx_ref);
    cudaFree(fy_ref);
    cudaFree(fz_ref);
    cudaFree(NN);
    cudaFree(NL);
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


void Fitness::read_box(char* input_dir)
{
    print_line_1();
    printf("Started reading box.in.\n");
    print_line_2();
    char file_box[200];
    strcpy(file_box, input_dir);
    strcat(file_box, "/box.in");
    FILE *fid_box = my_fopen(file_box, "r");

    int count = fscanf(fid_box, "%d", &num_boxes);
    if (count != 1) print_error("Reading error for box.in.\n");
    if (num_boxes < 1)
        print_error("Number of boxes should >= 1\n");
    else
        printf("Number of boxes = %d.\n", num_boxes);

    count = fscanf(fid_box, "%d", &box.triclinic);
    if (count != 1) print_error("Reading error for box.in.\n");
    if (box.triclinic == 0)
    {
        printf("orthogonal\n");
        box.memory = sizeof(double) * 3;
    }
    else if (box.triclinic == 1)
    {
        printf("triclinic\n");
        box.memory = sizeof(double) * 9;
    }
    else
        print_error("Invalid box type.\n");

    if (box.triclinic == 1)
    {
        MY_MALLOC(box.cpu_h, double, 18);
        double ax, ay, az, bx, by, bz, cx, cy, cz;
        int count = fscanf(fid_box, "%d%d%d%lf%lf%lf%lf%lf%lf%lf%lf%lf",
            &box.pbc_x, &box.pbc_y, &box.pbc_z, &ax, &ay, &az, &bx, &by, &bz,
            &cx, &cy, &cz);
        if (count != 12) print_error("reading error for xyz.in.\n");
        box.cpu_h[0] = ax; box.cpu_h[3] = ay; box.cpu_h[6] = az;
        box.cpu_h[1] = bx; box.cpu_h[4] = by; box.cpu_h[7] = bz;
        box.cpu_h[2] = cx; box.cpu_h[5] = cy; box.cpu_h[8] = cz;
        box.get_inverse();
        printf("%d %d %d ", box.pbc_x, box.pbc_y, box.pbc_z);
        for (int k = 0; k < 9; ++k) printf("%g ", box.cpu_h[k]);
        printf("\n");
    }
    else
    {
        MY_MALLOC(box.cpu_h, double, 6);
        double lx, ly, lz;
        int count = fscanf(fid_box, "%d%d%d%lf%lf%lf",
            &box.pbc_x, &box.pbc_y, &box.pbc_z, &lx, &ly, &lz);
        if (count != 6) print_error("reading error for line 2 of xyz.in.\n");
        box.cpu_h[0] = lx; box.cpu_h[1] = ly; box.cpu_h[2] = lz;
        box.cpu_h[3] = lx*0.5; box.cpu_h[4] = ly*0.5; box.cpu_h[5] = lz*0.5;
        printf("%d %d %d %g %g %g\n", 
            box.pbc_x, box.pbc_y, box.pbc_z, lx, ly, lz);
    }
    fclose(fid_box);
    CHECK(cudaMalloc((void**)&box.h, box.memory * 2));
    CHECK(cudaMemcpy(box.h, box.cpu_h, box.memory*2, cudaMemcpyHostToDevice));
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
    int *cpu_Na;
    int *cpu_Na_sum; 
    MY_MALLOC(cpu_Na, int, Nc);
    MY_MALLOC(cpu_Na_sum, int, Nc);
    N = 0;
    for (int nc = 0; nc < Nc; ++nc) { cpu_Na_sum[nc] = 0; }
    for (int nc = 0; nc < Nc; ++nc)
    {
        int count = fscanf(fid, "%d", &cpu_Na[nc]);
        if (count != 1) print_error("Reading error for xyz.in.\n");
        N += cpu_Na[nc];
        if (cpu_Na[nc] < 1)
            print_error("Number of atoms %d should >= 1\n");
        else
            printf("N[%d] = %d.\n", nc, cpu_Na[nc]);
    }
    for (int nc = 1; nc < Nc; ++nc) 
        cpu_Na_sum[nc] = cpu_Na_sum[nc-1] + cpu_Na[nc-1];
    int mem = sizeof(int) * Nc;
    CHECK(cudaMalloc((void**)&Na, mem));
    CHECK(cudaMalloc((void**)&Na_sum, mem));
    CHECK(cudaMemcpy(Na, cpu_Na, mem, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(Na_sum, cpu_Na_sum, mem, cudaMemcpyHostToDevice));
    MY_FREE(cpu_Na);
    MY_FREE(cpu_Na_sum);
} 


void Fitness::read_xyz(FILE* fid)
{
    int *cpu_type;
    double *cpu_x, *cpu_y, *cpu_z, *cpu_fx_ref, *cpu_fy_ref, *cpu_fz_ref;
    MY_MALLOC(cpu_type, int, N);
    MY_MALLOC(cpu_x, double, N);
    MY_MALLOC(cpu_y, double, N);
    MY_MALLOC(cpu_z, double, N);
    MY_MALLOC(cpu_fx_ref, double, N);
    MY_MALLOC(cpu_fy_ref, double, N);
    MY_MALLOC(cpu_fz_ref, double, N);
    num_types = 0;
    for (int n = 0; n < N; n++)
    {
        int count = fscanf(fid, "%d%lf%lf%lf%lf%lf%lf", 
            &(cpu_type[n]), &(cpu_x[n]), &(cpu_y[n]), &(cpu_z[n]),
            &(cpu_fx_ref[n]), &(cpu_fy_ref[n]), &(cpu_fz_ref[n]));
        if (count != 7) { print_error("reading error for xyz.in.\n"); }
        if (cpu_type[n] > num_types) { num_types = cpu_type[n]; }
    }
    num_types++;
    int m1 = sizeof(int) * N;
    int m2 = sizeof(double) * N;
    allocate_memory_gpu();
    CHECK(cudaMemcpy(type, cpu_type, m1, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(x, cpu_x, m2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(y, cpu_y, m2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(z, cpu_z, m2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(fx_ref, cpu_fx_ref, m2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(fy_ref, cpu_fy_ref, m2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(fz_ref, cpu_fz_ref, m2, cudaMemcpyHostToDevice));
    MY_FREE(cpu_type);
    MY_FREE(cpu_x);
    MY_FREE(cpu_y);
    MY_FREE(cpu_z);
    MY_FREE(cpu_fx_ref);
    MY_FREE(cpu_fy_ref);
    MY_FREE(cpu_fz_ref);

    int n_entries = num_types * num_types * num_types;
    MY_MALLOC(cpu_ters, double, n_entries * NUM_PARAMS);
    CHECK(cudaMalloc((void**)&ters, sizeof(double) * n_entries * NUM_PARAMS));
}


void Fitness::allocate_memory_gpu(void)
{
    int m1 = sizeof(int) * N;
    int m2 = sizeof(double) * N;
    // read from CPU
    CHECK(cudaMalloc((void**)&type, m1));
    CHECK(cudaMalloc((void**)&x, m2));
    CHECK(cudaMalloc((void**)&y, m2));
    CHECK(cudaMalloc((void**)&z, m2));
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
    CHECK(cudaMalloc((void**)&NN, m1));
    CHECK(cudaMalloc((void**)&NL, m1 * 20));
    CHECK(cudaMalloc((void**)&b, m2 * 20));
    CHECK(cudaMalloc((void**)&bp, m2 * 20));
    CHECK(cudaMalloc((void**)&f12x, m2 * 20));
    CHECK(cudaMalloc((void**)&f12y, m2 * 20));
    CHECK(cudaMalloc((void**)&f12z, m2 * 20));
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


