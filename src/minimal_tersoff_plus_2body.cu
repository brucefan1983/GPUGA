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

void Minimal_Tersoff_Plus_2body::update_potential(const std::vector<float>& potential_parameters)
{
  update_minimal_tersoff_parameters(potential_parameters, pot_para);
}

static __device__ void
find_fr_and_frp(float d0, float a, float r0, float s, float d12, float& fr, float& frp)
{

  fr = d0 / (s - 1.0f) * exp(-sqrt(2.0f * s) * a * (d12 - r0));
  frp = -sqrt(2.0f * s) * a * fr;
}

static __device__ void
find_fa_and_fap(float d0, float a, float r0, float s, float d12, float& fa, float& fap)
{
  fa = s * d0 / (s - 1.0f) * exp(-sqrt(2.0f / s) * a * (d12 - r0));
  fap = -sqrt(2.0f / s) * a * fa;
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
  const float* __restrict__ g_box,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial,
  float* g_pe)
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
    float pe = 0.0f;
    float fx = 0.0f;
    float fy = 0.0f;
    float fz = 0.0f;
    float virial_xx = 0.0f;
    float virial_yy = 0.0f;
    float virial_zz = 0.0f;
    float virial_xy = 0.0f;
    float virial_yz = 0.0f;
    float virial_zx = 0.0f;

    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int n2 = g_neighbor_list[n1 + number_of_particles * i1];

      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      dev_apply_mic(h, x12, y12, z12);
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);

      if (d12 < pot_para.ters[R1]) {
        g_neighbor_list_tersoff[count++ * number_of_particles + n1] = n2;
      } else {
        float d0 = pot_para.ters[D0];
        float a = pot_para.ters[A];
        float r0 = pot_para.ters[R0];
        float s = pot_para.ters[S];
        float fr, frp, fa, fap;
        find_fr_and_frp(d0, a, r0, s, d12, fr, frp);
        find_fa_and_fap(d0, a, r0, s, d12, fa, fap);
        float f12 = (frp - fap) / d12;
        fx += x12 * f12;
        fy += y12 * f12;
        fz += z12 * f12;
        virial_xx -= x12 * x12 * f12 * 0.5f;
        virial_yy -= y12 * y12 * f12 * 0.5f;
        virial_zz -= z12 * z12 * f12 * 0.5f;
        virial_xy -= x12 * y12 * f12 * 0.5f;
        virial_yz -= y12 * z12 * f12 * 0.5f;
        virial_zx -= z12 * x12 * f12 * 0.5f;
        pe += (fr - fa) * 0.5f;
      }
    }
    g_neighbor_number_tersoff[n1] = count;

    g_fx[n1] = fx * pot_para.ters[GAMMA];
    g_fy[n1] = fy * pot_para.ters[GAMMA];
    g_fz[n1] = fz * pot_para.ters[GAMMA];
    g_virial[n1 + number_of_particles * 0] = virial_xx;
    g_virial[n1 + number_of_particles * 1] = virial_yy;
    g_virial[n1 + number_of_particles * 2] = virial_zz;
    g_virial[n1 + number_of_particles * 3] = virial_xy;
    g_virial[n1 + number_of_particles * 4] = virial_yz;
    g_virial[n1 + number_of_particles * 5] = virial_zx;
    g_pe[n1] = pe;
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
  find_force_2body<<<Nc, max_Na>>>(
    N, Na, Na_sum, neighbor->NN, neighbor->NL, NN_tersoff.data(), NL_tersoff.data(), type, pot_para,
    r, r + N, r + N * 2, h, f.data(), f.data() + N, f.data() + N * 2, virial.data(), pe.data());
  CUDA_CHECK_KERNEL

  find_force_tersoff(
    pot_para, Nc, N, Na, Na_sum, max_Na, type, h, NN_tersoff.data(), NL_tersoff.data(), b, bp, f12x,
    f12y, f12z, r, f, virial, pe);
}
