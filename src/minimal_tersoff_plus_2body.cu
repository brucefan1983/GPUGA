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
Calculate force, energy, and virial for minimal-Tersoff + two-body
------------------------------------------------------------------------------*/

#include "error.cuh"
#include "gpu_vector.cuh"
#include "mic.cuh"
#include "minimal_tersoff_plus_2body.cuh"
#include "neighbor.cuh"

const int EPSILON = 0;
const int SIGMA = 1;

void Minimal_Tersoff_Plus_2body::initialize(int N, int MAX_ATOM_NUMBER)
{
  b.resize(N * MAX_ATOM_NUMBER);
  bp.resize(N * MAX_ATOM_NUMBER);
  f12x.resize(N * MAX_ATOM_NUMBER);
  f12y.resize(N * MAX_ATOM_NUMBER);
  f12z.resize(N * MAX_ATOM_NUMBER);
  NN_tersoff.resize(N);
  NL_tersoff.resize(N * MAX_ATOM_NUMBER);
}

void Minimal_Tersoff_Plus_2body::update_potential(const float* potential_parameters)
{
  // potential_parameters[0-8] for minimal Tersoff
  update_minimal_tersoff_parameters(potential_parameters, pot_para);
  // potential_parameters[9-10] for LJ
  // pot_para.lj[EPSILON] = potential_parameters[9];
  // pot_para.lj[SIGMA] = potential_parameters[10];
}

static __global__ void find_force_2body(
  int number_of_particles,
  int* Na,
  int* Na_sum,
  int* g_neighbor_number,
  int* g_neighbor_list,
  int* g_neighbor_number_tersoff,
  int* g_neighbor_list_tersoff,
  int* g_type,
  Pot_Para pot_para,
  const float* __restrict__ g_x,
  const float* __restrict__ g_y,
  const float* __restrict__ g_z,
  const float* __restrict__ g_box)
{
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  int n1 = N1 + threadIdx.x;
  if (n1 < N2) {
    const float* __restrict__ h = g_box + 18 * blockIdx.x;
    int neighbor_number = g_neighbor_number[n1];

    float x1 = g_x[n1];
    float y1 = g_y[n1];
    float z1 = g_z[n1];

    int count = 0;
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int n2 = g_neighbor_list[n1 + number_of_particles * i1];

      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      dev_apply_mic(h, x12, y12, z12);
      float distance_square = x12 * x12 + y12 * y12 + z12 * z12;

      if (distance_square < 3.0 * 3.0) { // TODO
        g_neighbor_list_tersoff[count++ * number_of_particles + n1] = n2;
      }
    }
    g_neighbor_number_tersoff[n1] = count;
  }
}

void Minimal_Tersoff_Plus_2body::find_force(
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
  GPU_Vector<float>& pe)
{
  // 2body
  find_force_2body<<<Nc, max_Na>>>(
    N, Na, Na_sum, neighbor->NN, neighbor->NL, NN_tersoff.data(), NL_tersoff.data(), type, pot_para,
    r, r + N, r + N * 2, h);
  CUDA_CHECK_KERNEL

  // tersoff
  find_force_tersoff_step1<<<Nc, max_Na>>>(
    N, Na, Na_sum, NN_tersoff.data(), NL_tersoff.data(), type, pot_para, r, r + N, r + N * 2, h,
    b.data(), bp.data());
  CUDA_CHECK_KERNEL
  find_force_tersoff_step2<<<Nc, max_Na>>>(
    N, Na, Na_sum, NN_tersoff.data(), NL_tersoff.data(), type, pot_para, b.data(), bp.data(), r,
    r + N, r + N * 2, h, pe.data(), f12x.data(), f12y.data(), f12z.data());
  CUDA_CHECK_KERNEL
  find_force_tersoff_step3<<<Nc, max_Na>>>(
    N, Na, Na_sum, NN_tersoff.data(), NL_tersoff.data(), f12x.data(), f12y.data(), f12z.data(), r,
    r + N, r + N * 2, h, f.data(), f.data() + N, f.data() + N * 2, virial.data());
  CUDA_CHECK_KERNEL
}
