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
class Box;
class Neighbor;


struct Pot_Para
{
    float ters[13];
};

class Potential
{
public:

    void update_potential(float*, int);
    void find_force(int num_types, int Nc, int N, int *Na, int *Na_sum,
        int max_Na, int *type, Box *box, Neighbor *neighbor,
        float *x, float *y, float *z, float *fx, float *fy, float *fz, 
        float *sxx, float *syy, float *szz, float *pe, 
        float *f12x, float *f12y, float *f12z, float *b, float *bp);

private:
    
    Pot_Para pot_para;
};



