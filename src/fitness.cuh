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


struct Weight
{
    float force;
    float energy;
    float stress;
};


class Fitness
{
public:

    Fitness(char*);
    ~Fitness(void);
    void compute(int, int, float*, float*, float*, float*);
    void predict(char*, int, float*, float*, float*);

protected:

    // functions related to initialization
    void read_weight(char*);
    void read_xyz_in(char*);
    void read_Nc(FILE*);
    void read_Na(FILE*);
    void read_xyz(FILE*);
    void read_box(char*);

    // functions related to fitness evaluation
    void predict_energy_or_stress(FILE*, float*, float*, int, int);
    float get_fitness_force(void);
    float get_fitness_energy(void);
    float get_fitness_stress(void);

    int Nc;              // number of configurations
    int NC_FORCE;        // number of force configurations
    int N;               // total number of atoms (sum of Na[])
    int num_types;       // number of atom types
    int MAX_ATOM_NUMBER; // number of atoms in the largest configuration
    int *Na;             // number of atoms in each configuration
    int *Na_sum;         // prefix sum of Na
    int *type;           // atom type

    float *x, *y, *z;                // position
    float *fx, *fy, *fz;             // force
    float *pe;                       // potential energy
    float *sxx, *syy, *szz;          // virial
    float *fx_ref, *fy_ref, *fz_ref; // reference force
    float *error_cpu, *error_gpu;    // error in energy, virial, or force
    float force_square_sum;          // sum of force square

    // other classes
    Box box;
    Neighbor neighbor;
    Potential potential;
    Weight weight;
};


