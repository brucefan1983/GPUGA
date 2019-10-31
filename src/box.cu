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
The class defining the simulation box.
------------------------------------------------------------------------------*/


#include "box.cuh"
#include "error.cuh"
#include "common.cuh"


void Box::read_file(char* input_dir, int Nc)
{
    print_line_1();
    printf("Started reading box.in.\n");
    print_line_2();
    char file_box[200];
    strcpy(file_box, input_dir);
    strcat(file_box, "/box.in");
    FILE *fid_box = my_fopen(file_box, "r");

    CHECK(cudaMallocManaged((void**)&triclinic, sizeof(int) * Nc));
    CHECK(cudaMallocManaged((void**)&h, sizeof(float) * Nc * 18));
    MY_MALLOC(cpu_pe_ref, float, Nc);
    MY_MALLOC(cpu_sxx_ref, float, Nc);
    MY_MALLOC(cpu_syy_ref, float, Nc);
    MY_MALLOC(cpu_szz_ref, float, Nc);
    
    potential_square_sum = 0.0;
    virial_square_sum = 0.0;
    for (int n = 0; n < Nc; ++n)
    {
        float *h_local = h + n * 18; // define a local pointer

        int count = fscanf(fid_box, "%d%f%f%f%f", 
            &triclinic[n], &cpu_pe_ref[n], &cpu_sxx_ref[n],
            &cpu_syy_ref[n], &cpu_szz_ref[n]);

        if (n >= NC_FORCE)
        {
            float energy = cpu_pe_ref[n] + 4.63 * MAX_ATOM_NUMBER;
            potential_square_sum += energy * energy;
            virial_square_sum += cpu_sxx_ref[n] * cpu_sxx_ref[n]
                               + cpu_syy_ref[n] * cpu_syy_ref[n]
                               + cpu_szz_ref[n] * cpu_szz_ref[n];
        }

        if (count != 5) print_error("Reading error for box.in.\n");
        if (triclinic[n] == 0) printf("orthogonal %g %g %g %g\n", 
            cpu_pe_ref[n], cpu_sxx_ref[n], cpu_syy_ref[n], cpu_szz_ref[n]);
        else if (triclinic[n] == 1) printf("triclinic %g %g %g %g\n", 
            cpu_pe_ref[n], cpu_sxx_ref[n], cpu_syy_ref[n], cpu_szz_ref[n]);
        else print_error("Invalid box type.\n");

        if (triclinic[n] == 1)
        {
            float ax, ay, az, bx, by, bz, cx, cy, cz;
            int count = fscanf(fid_box, "%f%f%f%f%f%f%f%f%f",
                &ax, &ay, &az, &bx, &by, &bz, &cx, &cy, &cz);
            if (count != 9) print_error("reading error for box.in.\n");
            h_local[0] = ax; h_local[3] = ay; h_local[6] = az;
            h_local[1] = bx; h_local[4] = by; h_local[7] = bz;
            h_local[2] = cx; h_local[5] = cy; h_local[8] = cz;
            get_inverse(triclinic[n], h_local);
            for (int k = 0; k < 9; ++k) printf("%g ", h_local[k]);
            printf("\n");
        }
        else
        {
            float lx, ly, lz;
            int count = fscanf(fid_box, "%f%f%f", &lx, &ly, &lz);
            if (count != 3) print_error("reading error for box.in.\n");
            h_local[0] = lx; h_local[1] = ly; h_local[2] = lz;
            printf("%g %g %g\n", lx, ly, lz);
        }
    }
    fclose(fid_box);

    int memory = sizeof(float) * Nc;
    CHECK(cudaMalloc((void**)&pe_ref, memory));
    CHECK(cudaMalloc((void**)&sxx_ref, memory));
    CHECK(cudaMalloc((void**)&syy_ref, memory));
    CHECK(cudaMalloc((void**)&szz_ref, memory));
    CHECK(cudaMemcpy(pe_ref, cpu_pe_ref, memory, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(sxx_ref, cpu_sxx_ref, memory, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(syy_ref, cpu_syy_ref, memory, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(szz_ref, cpu_szz_ref, memory, cudaMemcpyHostToDevice));
}  


Box::~Box(void)
{
    CHECK(cudaFree(triclinic));
    CHECK(cudaFree(h));
    MY_FREE(cpu_pe_ref);
    MY_FREE(cpu_sxx_ref);
    MY_FREE(cpu_syy_ref);
    MY_FREE(cpu_szz_ref);
    CHECK(cudaFree(pe_ref)); 
    CHECK(cudaFree(sxx_ref));
    CHECK(cudaFree(syy_ref));
    CHECK(cudaFree(szz_ref));
}


float Box::get_volume(int triclinic, float *cpu_h)
{
    float volume;
    if (triclinic)
    {
        volume = cpu_h[0] * (cpu_h[4]*cpu_h[8] - cpu_h[5]*cpu_h[7])
               + cpu_h[1] * (cpu_h[5]*cpu_h[6] - cpu_h[3]*cpu_h[8])
               + cpu_h[2] * (cpu_h[3]*cpu_h[7] - cpu_h[4]*cpu_h[6]);
    }
    else
    {
        volume = cpu_h[0] * cpu_h[1] * cpu_h[2];
    }
    return volume;
}


void Box::get_inverse(int triclinic, float *cpu_h)
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
    float volume = get_volume(triclinic, cpu_h);
    for (int n = 9; n < 18; n++)
    {
        cpu_h[n] /= volume;
    }
}


