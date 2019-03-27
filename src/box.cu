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


void Box::read_file(char* input_dir)
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

    MY_MALLOC(cpu_h, double, 18 * num_boxes);
    for (int n = 0; n < num_boxes; ++n)
    {
        count = fscanf(fid_box, "%d", &triclinic);
        if (count != 1) print_error("Reading error for box.in.\n");
        if (triclinic == 0) printf("orthogonal\n");
        else if (triclinic == 1) printf("triclinic\n");
        else print_error("Invalid box type.\n");

        if (triclinic == 1)
        {
            double ax, ay, az, bx, by, bz, cx, cy, cz;
            int count = fscanf(fid_box, "%d%d%d%lf%lf%lf%lf%lf%lf%lf%lf%lf",
                &pbc_x, &pbc_y, &pbc_z, &ax, &ay, &az, &bx, &by, &bz,
                &cx, &cy, &cz);
            if (count != 12) print_error("reading error for box.in.\n");
            cpu_h[0] = ax; cpu_h[3] = ay; cpu_h[6] = az;
            cpu_h[1] = bx; cpu_h[4] = by; cpu_h[7] = bz;
            cpu_h[2] = cx; cpu_h[5] = cy; cpu_h[8] = cz;
            get_inverse();
            printf("%d %d %d ", pbc_x, pbc_y, pbc_z);
            for (int k = 0; k < 9; ++k) printf("%g ", cpu_h[k]);
            printf("\n");
        }
        else
        {
            double lx, ly, lz;
            int count = fscanf(fid_box, "%d%d%d%lf%lf%lf",
                &pbc_x, &pbc_y, &pbc_z, &lx, &ly, &lz);
            if (count != 6) print_error("reading error for box.in.\n");
            cpu_h[0] = lx; cpu_h[1] = ly; cpu_h[2] = lz;
            printf("%d %d %d %g %g %g\n", pbc_x, pbc_y, pbc_z, lx, ly, lz);
        }
        fclose(fid_box);
        int memory = sizeof(double) * num_boxes * 18;
        CHECK(cudaMalloc((void**)&h, memory));
        CHECK(cudaMemcpy(h, cpu_h, memory, cudaMemcpyHostToDevice));
    }
}  


Box::~Box(void)
{
    MY_FREE(cpu_h);
    CHECK(cudaFree(h));
}


double Box::get_volume(void)
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


void Box::get_inverse(void)
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
    double volume = get_volume();
    for (int n = 9; n < 18; n++)
    {
        cpu_h[n] /= volume;
    }
}


