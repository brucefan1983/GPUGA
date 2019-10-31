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


class Box
{
public:
    void read_file(char*, int);
    ~Box(void);
    int *triclinic;
    double* h; // GPU box data
    // energy for the whole box 
    double *pe_ref, *sxx_ref, *syy_ref, *szz_ref;
    double *cpu_pe_ref, *cpu_sxx_ref, *cpu_syy_ref, *cpu_szz_ref;
    double potential_square_sum;
    double virial_square_sum;
    // functions
    double get_volume(int, double*); // get the volume of the box
    void get_inverse(int, double*);  // get the inverse box matrix
};


