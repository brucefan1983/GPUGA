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
#include <random>
class Fitness;


class GA
{
public:
    GA(char*, Fitness*);
    ~GA(void);

protected:
    // parameters
    std::mt19937 rng;
    int maximum_generation = 1000;
    int number_of_variables = 10;
    int population_size = 200;
    int parent_number = 100;
    int child_number = 100;
    float mutation_rate = 0.2;
    // data
    int* index;
    float* fitness;
    float* cumulative_probabilities;
    float* population;
    float* population_copy;
    float* parameters_min;
    float* parameters_max;
    // for evolution
    void compute(char*, Fitness*);
    void sort_population(int);
    void output(int, FILE*);
    void crossover(void);
    int get_a_parent(void);
    void mutation(int);
    // for initialization
    void copy_potential(Fitness*);
    void read_parameters(char*);
    void parse(char**, int);
    void parse_maximum_generation(char**, int);
    void parse_population_size(char**, int);
    void parse_parent_number(char**, int);
    void parse_mutation_rate(char**, int);
};


