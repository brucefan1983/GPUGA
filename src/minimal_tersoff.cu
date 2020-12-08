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
Calculate force, energy, and virial
------------------------------------------------------------------------------*/

#include "error.cuh"
#include "gpu_vector.cuh"
#include "mic.cuh"
#include "minimal_tersoff.cuh"
#include "neighbor.cuh"

void Minimal_Tersoff::initialize(int N, int MAX_ATOM_NUMBER)
{
  b.resize(N * MAX_ATOM_NUMBER);
  bp.resize(N * MAX_ATOM_NUMBER);
  f12x.resize(N * MAX_ATOM_NUMBER);
  f12y.resize(N * MAX_ATOM_NUMBER);
  f12z.resize(N * MAX_ATOM_NUMBER);
}

void Minimal_Tersoff::update_potential(float* potential_parameters)
{
  pot_para.ters[D0] = potential_parameters[0];
  pot_para.ters[A] = potential_parameters[1];
  pot_para.ters[R0] = potential_parameters[2];
  pot_para.ters[S] = potential_parameters[3];
  pot_para.ters[EN] = potential_parameters[4];
  pot_para.ters[BETA] = potential_parameters[5];
  pot_para.ters[H] = potential_parameters[6];
  pot_para.ters[R1] = potential_parameters[7];
  pot_para.ters[R2] = potential_parameters[8];
  pot_para.ters[PI_FACTOR] = PI / (pot_para.ters[R2] - pot_para.ters[R1]);
  pot_para.ters[MINUS_HALF_OVER_N] = -0.5 / pot_para.ters[EN];
}

void Minimal_Tersoff::find_force(
  int Nc,
  int N,
  int* Na,
  int* Na_sum,
  int max_Na,
  int* type,
  float* h,
  Neighbor* neighbor,
  float* r,
  float* f,
  float* virial,
  float* pe)
{
  find_force_tersoff_step1<<<Nc, max_Na>>>(
    N, Na, Na_sum, neighbor->NN, neighbor->NL, type, pot_para, r, r + N, r + N * 2, h, b.data(),
    bp.data());
  CUDA_CHECK_KERNEL
  find_force_tersoff_step2<<<Nc, max_Na>>>(
    N, Na, Na_sum, neighbor->NN, neighbor->NL, type, pot_para, b.data(), bp.data(), r, r + N,
    r + N * 2, h, pe, f12x.data(), f12y.data(), f12z.data());
  CUDA_CHECK_KERNEL
  find_force_tersoff_step3<<<Nc, max_Na>>>(
    N, Na, Na_sum, neighbor->NN, neighbor->NL, f12x.data(), f12y.data(), f12z.data(), r, r + N,
    r + N * 2, h, f, f + N, f + N * 2, virial);
  CUDA_CHECK_KERNEL
}
