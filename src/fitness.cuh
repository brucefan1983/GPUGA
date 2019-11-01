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


#pragma once
#include "box.cuh"
#include "neighbor.cuh"
#include "potential.cuh"
#include <stdio.h>


class Fitness
{
public:

    Fitness(char*);
    ~Fitness(void);
    void compute(int, int, float*, float*, float*, float*);
    void predict(char*, int, float*, float*, float*);

protected:

    // functions related to initialization
    void read_xyz_in(char*);
    void read_Nc(FILE*);
    void read_Na(FILE*);
    void read_xyz(FILE*);
    void read_box(char*);
    void allocate_memory_gpu(void);

    // functions related to fitness evaluation
    void predict_energy_or_stress(FILE*, float*, float*, float*, int, int);
    float get_fitness_force(float*, float*);
    float get_fitness_energy(float*, float*);
    float get_fitness_stress(float*, float*);

    // integer variables
    int Nc;      // number of configurations
    int N;       // total number of atoms (sum of Na)
    int num_types;
    int MAX_ATOM_NUMBER; 
    int *Na;     // number of atoms in each configuration
    int *Na_sum; // prefix sum of Na
    int *type;

    const int NC_FORCE           = 5;


    // real variables
    float *b, *bp, *f12x, *f12y, *f12z;
    float *x, *y, *z, *fx, *fy, *fz, *pe, *sxx, *syy, *szz;
    float *fx_ref, *fy_ref, *fz_ref; 
    float *cpu_fx, *cpu_fy, *cpu_fz;
    float *cpu_fx_ref, *cpu_fy_ref, *cpu_fz_ref;
    float force_square_sum;

    // other classes
    Box box;
    Neighbor neighbor;
    Potential potential;
};


