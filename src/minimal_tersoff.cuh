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

struct Pot_Para {
  float D0[2], A[2], R0[2], S[2], EN, BETA, H, R1, R2, PI_FACTOR, MINUS_HALF_OVER_N;
};

class Minimal_Tersoff : public Potential
{
public:
  Minimal_Tersoff(int num_types);
  void initialize(int, int);
  void update_potential(const std::vector<float>&);
  void find_force(
    int,
    int,
    int*,
    int*,
    int,
    int*,
    float*,
    Neighbor*,
    float*,
    GPU_Vector<float>&,
    GPU_Vector<float>&,
    GPU_Vector<float>&);

private:
  int num_types_;
  GPU_Vector<float> b, bp, f12x, f12y, f12z;
  Pot_Para pot_para;
};
