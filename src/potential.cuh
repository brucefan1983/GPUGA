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

class Neighbor;

struct Pot_Para {
  float ters[11];
  float lj[2];
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
const int PI_FACTOR = 9;
const int MINUS_HALF_OVER_N = 10;

class Potential
{
public:
  virtual ~Potential() = default;
  virtual void initialize(int, int) = 0;
  virtual void update_potential(const float*) = 0;
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

void update_minimal_tersoff_parameters(const float* potential_parameters, Pot_Para& pot_para);

__global__ void find_force_tersoff_step1(
  int number_of_particles,
  int* Na,
  int* Na_sum,
  int* g_neighbor_number,
  int* g_neighbor_list,
  int* g_type,
  Pot_Para pot_para,
  const float* __restrict__ g_x,
  const float* __restrict__ g_y,
  const float* __restrict__ g_z,
  const float* __restrict__ g_box,
  float* g_b,
  float* g_bp);

__global__ void find_force_tersoff_step2(
  int number_of_particles,
  int* Na,
  int* Na_sum,
  int* g_neighbor_number,
  int* g_neighbor_list,
  int* g_type,
  Pot_Para pot_para,
  const float* __restrict__ g_b,
  const float* __restrict__ g_bp,
  const float* __restrict__ g_x,
  const float* __restrict__ g_y,
  const float* __restrict__ g_z,
  const float* __restrict__ g_box,
  float* g_potential,
  float* g_f12x,
  float* g_f12y,
  float* g_f12z);

__global__ void find_force_tersoff_step3(
  int number_of_particles,
  int* Na,
  int* Na_sum,
  int* g_neighbor_number,
  int* g_neighbor_list,
  const float* __restrict__ g_f12x,
  const float* __restrict__ g_f12y,
  const float* __restrict__ g_f12z,
  const float* __restrict__ g_x,
  const float* __restrict__ g_y,
  const float* __restrict__ g_z,
  const float* __restrict__ g_box,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial);
