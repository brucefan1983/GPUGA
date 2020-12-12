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
#include <vector>

class Neighbor;

struct Pot_Para {
  float ters[12];
};

const float PI = 3.141592653589793;

// Easy labels for indexing
const int D0 = 0;
const int A = 1;
const int R0 = 2;
const int S = 3;
const int EN = 4;
const int BETA = 5;
const int H = 6;
const int R1 = 7;
const int R2 = 8;
const int GAMMA = 9;
const int PI_FACTOR = 10;
const int MINUS_HALF_OVER_N = 11;

class Potential
{
public:
  virtual ~Potential() = default;
  virtual void initialize(int, int) = 0;
  virtual void update_potential(const std::vector<float>&) = 0;
  virtual void find_force(
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
    GPU_Vector<float>&) = 0;
};

void update_minimal_tersoff_parameters(
  const std::vector<float>& potential_parameters, Pot_Para& pot_para);

void find_force_tersoff(
  const Pot_Para& pot_para,
  int Nc,
  int N,
  int* Na,
  int* Na_sum,
  int max_Na,
  int* type,
  float* h,
  int* NN,
  int* NL,
  GPU_Vector<float>& b,
  GPU_Vector<float>& bp,
  GPU_Vector<float>& f12x,
  GPU_Vector<float>& f12y,
  GPU_Vector<float>& f12z,
  float* r,
  GPU_Vector<float>& f,
  GPU_Vector<float>& virial,
  GPU_Vector<float>& pe);
