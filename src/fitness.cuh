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
    void compute(int, float*, float*);
    void predict(char*, float*);
    int number_of_variables; // number of variables in the potential
    float *parameters_min;   // lower limit of the potential parameters
    float *parameters_max;   // upper limit of the potential parameters

protected:

    // functions related to initialization
    void read_Nc(FILE*);
    void read_Na(FILE*);
    void read_potential(char*);
    void read_train_in(char*);

    // functions related to fitness evaluation
    void predict_energy_or_stress(FILE*, float*, float*);
    float get_fitness_force(void);
    float get_fitness_energy(void);
    float get_fitness_stress(void);

    int potential_type;  // 1=tersoff_mini_1 and 2=tersoff_mini_2
    int Nc;              // number of configurations
    int Nc_force;        // number of force configurations
    int N;               // total number of atoms (sum of Na[])
    int N_force;         // total number of atoms in force configurations
    int max_Na;          // number of atoms in the largest configuration
    int *Na;             // number of atoms in each configuration
    int *Na_sum;         // prefix sum of Na
    int *type;           // atom type

    float *r;                        // position
    float *force;                    // force
    float *pe;                       // potential energy
    float *virial;                   // per-atom virial tensor
    float *h;                        // box and inverse box
    float *pe_ref;                   // reference energy for the whole box
    float *virial_ref;               // reference virial for the whole box
    float *force_ref;                // reference force
    float *error_cpu, *error_gpu;    // error in energy, virial, or force
    float force_square_sum;          // sum of force square
    float potential_square_sum;      // sum of potential square
    float virial_square_sum;         // sum of virial square

    // other classes
    Neighbor neighbor;
    Potential potential;
    Weight weight;
};


