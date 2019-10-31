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
    int maximum_generation = 1000000;
    int number_of_variables = 10;
    int population_size = 100;
    int parent_number = 50;
    int child_number = 50;
    double mutation_rate = 0.1;
    double minimum_cost = 1.0e-6;
    // data
    int* index;
    double* fitness;
    double* cumulative_probabilities;
    double* population;
    double* population_copy;
    double* parameters_min;
    double* parameters_max;
    // for evolution
    void compute(char*, Fitness*);
    void sort_population(int);
    void output(int, FILE*);
    void crossover(void);
    int get_a_parent(void);
    void mutation(int);
    // for initialization
    void read_parameters(char*);
    void read_potential(char*);
    void parse(char**, int);
    void parse_maximum_generation(char**, int);
    void parse_number_of_variables(char**, int);
    void parse_population_size(char**, int);
    void parse_parent_number(char**, int);
    void parse_mutation_rate(char**, int);
    void parse_minimum_cost(char**, int);
};


