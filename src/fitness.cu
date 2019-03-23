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
#include "error.cuh"
#include "read_file.cuh"
#define BLOCK_SIZE 128


Fitness::Fitness(void)
{
    // nothing now
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


Fitness::~Fitness(void)
{
    // nothing now
}


