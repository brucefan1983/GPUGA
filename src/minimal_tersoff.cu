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

void Minimal_Tersoff::update_potential(const float* potential_parameters)
{
  update_minimal_tersoff_parameters(potential_parameters, pot_para);
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
  GPU_Vector<float>& f,
  GPU_Vector<float>& virial,
  GPU_Vector<float>& pe)
{
  f.fill(0.0);
  virial.fill(0.0);
  pe.fill(0.0);

  find_force_tersoff(
    pot_para, Nc, N, Na, Na_sum, max_Na, type, h, neighbor->NN, neighbor->NL, b, bp, f12x, f12y,
    f12z, r, f, virial, pe);
}
