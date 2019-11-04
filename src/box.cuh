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
    void read_file(char*, int, int);
    ~Box(void);
    int *triclinic;                     // 1 for triclinic and 0 for orthogonal
    float* h;                           // box and inverse box
    float *pe_ref;                      // energy for the whole box
    float *sxx_ref, *syy_ref, *szz_ref; // virial for the whole box
    float potential_square_sum;         // sum of potential square
    float virial_square_sum;            // sum of virial square
private:
    float get_volume(int, float*);      // get the volume of the box
    void get_inverse(int, float*);      // get the inverse box matrix
};


