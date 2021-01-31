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
#include "gpu_vector.cuh"
#include "potential.cuh"
class Neighbor;

struct RI_Para {
  float a[3], b[3], c[3], qq[3], v_rc, dv_rc, cutoff;
};

class RI : public Potential
{
public:
  void initialize(int, int);
  void update_potential(const std::vector<float>&);
  void find_force(
    int Nc,
    int N,
    int* Na,
    int* Na_sum,
    int max_Na,
    int* type,
    float* h,
    Neighbor* neighbor,
    float* r,
    GPU_Vector<float>& f,
    GPU_Vector<float>& virial,
    GPU_Vector<float>& pe);

private:
  RI_Para ri_para;
};
