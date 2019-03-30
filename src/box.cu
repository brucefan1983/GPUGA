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


void Box::read_file(char* input_dir, int Nc)
{
    print_line_1();
    printf("Started reading box.in.\n");
    print_line_2();
    char file_box[200];
    strcpy(file_box, input_dir);
    strcat(file_box, "/box.in");
    FILE *fid_box = my_fopen(file_box, "r");

    MY_MALLOC(cpu_pe_ref, double, Nc);
    MY_MALLOC(cpu_h, double, 18 * Nc);
    pe_ref_square_sum = 0.0;
    for (int n = 0; n < Nc; ++n)
    {
        double *h_local = cpu_h + n * 18; // define a local pointer

        int count = fscanf(fid_box, "%d%lf", &triclinic, &cpu_pe_ref[n]);
        if (count != 2) print_error("Reading error for box.in.\n");
        if (triclinic == 0) printf("orthogonal %g\n", cpu_pe_ref[n]);
        else if (triclinic == 1) printf("triclinic %g\n", cpu_pe_ref[n]);
        else print_error("Invalid box type.\n");

        pe_ref_square_sum += cpu_pe_ref[n] * cpu_pe_ref[n];

        if (triclinic == 1)
        {
            double ax, ay, az, bx, by, bz, cx, cy, cz;
            int count = fscanf(fid_box, "%lf%lf%lf%lf%lf%lf%lf%lf%lf",
                &ax, &ay, &az, &bx, &by, &bz, &cx, &cy, &cz);
            if (count != 9) print_error("reading error for box.in.\n");
            h_local[0] = ax; h_local[3] = ay; h_local[6] = az;
            h_local[1] = bx; h_local[4] = by; h_local[7] = bz;
            h_local[2] = cx; h_local[5] = cy; h_local[8] = cz;
            get_inverse(h_local);
            for (int k = 0; k < 9; ++k) printf("%g ", h_local[k]);
            printf("\n");
        }
        else
        {
            double lx, ly, lz;
            int count = fscanf(fid_box, "%lf%lf%lf", &lx, &ly, &lz);
            if (count != 3) print_error("reading error for box.in.\n");
            h_local[0] = lx; h_local[1] = ly; h_local[2] = lz;
            printf("%g %g %g\n", lx, ly, lz);
        }
    }
    fclose(fid_box);

    int memory = sizeof(double) * Nc * 18;
    CHECK(cudaMalloc((void**)&h, memory));
    CHECK(cudaMemcpy(h, cpu_h, memory, cudaMemcpyHostToDevice));
    memory = sizeof(double) * Nc;
    CHECK(cudaMalloc((void**)&pe_ref, memory));
    CHECK(cudaMemcpy(pe_ref, cpu_pe_ref, memory, cudaMemcpyHostToDevice));
}  


Box::~Box(void)
{
    MY_FREE(cpu_h);
    CHECK(cudaFree(h));
    MY_FREE(cpu_pe_ref);
    CHECK(cudaFree(pe_ref)); 
}


double Box::get_volume(double *cpu_h)
{
    double volume;
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


void Box::get_inverse(double *cpu_h)
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
    double volume = get_volume(cpu_h);
    for (int n = 9; n < 18; n++)
    {
        cpu_h[n] /= volume;
    }
}


