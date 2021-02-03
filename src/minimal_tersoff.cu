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

const float PI = 3.141592653589793;

Minimal_Tersoff::Minimal_Tersoff(int num_types) { num_types_ = num_types; }

void Minimal_Tersoff::initialize(int N, int MAX_ATOM_NUMBER)
{
  b.resize(N * MAX_ATOM_NUMBER);
  bp.resize(N * MAX_ATOM_NUMBER);
  f12x.resize(N * MAX_ATOM_NUMBER);
  f12y.resize(N * MAX_ATOM_NUMBER);
  f12z.resize(N * MAX_ATOM_NUMBER);
}

void Minimal_Tersoff::update_potential(const std::vector<float>& potential_parameters)
{
  pot_para.D0[0] = potential_parameters[0];
  pot_para.A[0] = potential_parameters[1];
  pot_para.R0[0] = potential_parameters[2];
  pot_para.S[0] = potential_parameters[3];
  int offset = 0;
  if (num_types_ == 2) {
    offset = 4;
    pot_para.D0[1] = potential_parameters[4];
    pot_para.A[1] = potential_parameters[5];
    pot_para.R0[1] = potential_parameters[6];
    pot_para.S[1] = potential_parameters[7];
  }
  pot_para.EN = potential_parameters[4 + offset];
  pot_para.BETA = potential_parameters[5 + offset];
  pot_para.H = potential_parameters[6 + offset];
  pot_para.R1 = potential_parameters[7 + offset];
  pot_para.R2 = potential_parameters[8 + offset];
  pot_para.PI_FACTOR = PI / (pot_para.R2 - pot_para.R1);
  pot_para.MINUS_HALF_OVER_N = -0.5 / pot_para.EN;
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

static __device__ void
find_fc_and_fcp(float r1, float r2, float pi_factor, float d12, float& fc, float& fcp)
{
  if (d12 < r1) {
    fc = 1.0f;
    fcp = 0.0f;
  } else if (d12 < r2) {
    fc = 0.5f * cos(pi_factor * (d12 - r1)) + 0.5f;
    fcp = -sin(pi_factor * (d12 - r1)) * pi_factor * 0.5f;
  } else {
    fc = 0.0f;
    fcp = 0.0f;
  }
}

static __device__ void find_fa(float d0, float a, float r0, float s, float d12, float& fa)
{
  fa = s * d0 / (s - 1.0f) * exp(-sqrt(2.0f / s) * a * (d12 - r0));
}

static __device__ void find_fc(float r1, float r2, float pi_factor, float d12, float& fc)
{
  if (d12 < r1) {
    fc = 1.0f;
  } else if (d12 < r2) {
    fc = 0.5f * cos(pi_factor * (d12 - r1)) + 0.5f;
  } else {
    fc = 0.0f;
  }
}

static __device__ void find_g_and_gp(float beta, float h, float cos, float& g, float& gp)
{
  float x = cos - h;
  g = beta * x * x;
  gp = beta * 2.0f * x;
}

static __device__ void find_g(float beta, float h, float cos, float& g)
{
  float x = cos - h;
  g = beta * x * x;
}

// step 1: pre-compute all the bond-order functions and their derivatives
static __global__ void find_force_tersoff_step1(
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
  float* g_bp)
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
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int n2 = g_neighbor_list[n1 + number_of_particles * i1];

      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      dev_apply_mic(h, x12, y12, z12);
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float zeta = 0.0f;
      for (int i2 = 0; i2 < neighbor_number; ++i2) {
        int n3 = g_neighbor_list[n1 + number_of_particles * i2];
        if (n3 == n2) {
          continue;
        } // ensure that n3 != n2

        float x13 = g_x[n3] - x1;
        float y13 = g_y[n3] - y1;
        float z13 = g_z[n3] - z1;
        dev_apply_mic(h, x13, y13, z13);
        float d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
        float cos123 = (x12 * x13 + y12 * y13 + z12 * z13) / (d12 * d13);
        float fc13, g123;

        find_fc(pot_para.R1, pot_para.R2, pot_para.PI_FACTOR, d13, fc13);
        find_g(pot_para.BETA, pot_para.H, cos123, g123);
        zeta += fc13 * g123;
      }
      float bzn, b_ijj;
      bzn = pow(zeta, pot_para.EN);
      b_ijj = pow(1.0f + bzn, pot_para.MINUS_HALF_OVER_N);

      if (zeta < 1.0e-16f) // avoid division by 0
      {
        g_b[i1 * number_of_particles + n1] = 1.0;
        g_bp[i1 * number_of_particles + n1] = 0.0;
      } else {
        g_b[i1 * number_of_particles + n1] = b_ijj;
        g_bp[i1 * number_of_particles + n1] = -b_ijj * bzn * 0.5f / ((1.0f + bzn) * zeta);
      }
    }
  }
}

// step 2: calculate all the partial forces dU_i/dr_ij
static __global__ void find_force_tersoff_step2(
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
  float* g_f12z)
{
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  int n1 = N1 + threadIdx.x;
  if (n1 < N2) {
    const float* __restrict__ h = g_box + 18 * blockIdx.x;
    int neighbor_number = g_neighbor_number[n1];
    int type1 = g_type[n1];

    float x1 = g_x[n1];
    float y1 = g_y[n1];
    float z1 = g_z[n1];
    float pot_energy = 0.0f;
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * number_of_particles + n1;
      int n2 = g_neighbor_list[index];
      int type12 = type1 + g_type[n2];

      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      dev_apply_mic(h, x12, y12, z12);
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float d12inv = 1.0f / d12;
      float fc12, fcp12, fa12, fap12, fr12, frp12;

      float d0 = pot_para.D0[type12];
      float a = pot_para.A[type12];
      float r0 = pot_para.R0[type12];
      float s = pot_para.S[type12];

      find_fc_and_fcp(pot_para.R1, pot_para.R2, pot_para.PI_FACTOR, d12, fc12, fcp12);
      find_fa_and_fap(d0, a, r0, s, d12, fa12, fap12);
      find_fr_and_frp(d0, a, r0, s, d12, fr12, frp12);

      // (i,j) part
      float b12 = g_b[index];
      float factor3 = (fcp12 * (fr12 - b12 * fa12) + fc12 * (frp12 - b12 * fap12)) * d12inv;
      float f12x = x12 * factor3 * 0.5f;
      float f12y = y12 * factor3 * 0.5f;
      float f12z = z12 * factor3 * 0.5f;

      // accumulate potential energy
      pot_energy += fc12 * (fr12 - b12 * fa12) * 0.5f;

      // (i,j,k) part
      float bp12 = g_bp[index];
      for (int i2 = 0; i2 < neighbor_number; ++i2) {
        int index_2 = n1 + number_of_particles * i2;
        int n3 = g_neighbor_list[index_2];
        if (n3 == n2) {
          continue;
        }

        float x13 = g_x[n3] - x1;
        float y13 = g_y[n3] - y1;
        float z13 = g_z[n3] - z1;
        dev_apply_mic(h, x13, y13, z13);
        float d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
        float fc13, fa13;
        find_fc(pot_para.R1, pot_para.R2, pot_para.PI_FACTOR, d13, fc13);
        find_fa(d0, a, r0, s, d13, fa13);
        float bp13 = g_bp[index_2];
        float one_over_d12d13 = 1.0f / (d12 * d13);
        float cos123 = (x12 * x13 + y12 * y13 + z12 * z13) * one_over_d12d13;
        float cos123_over_d12d12 = cos123 * d12inv * d12inv;
        float g123, gp123;
        find_g_and_gp(pot_para.BETA, pot_para.H, cos123, g123, gp123);

        // derivatives with cosine
        float dc = -fc12 * bp12 * fa12 * fc13 * gp123 - fc12 * bp13 * fa13 * fc13 * gp123;
        // derivatives with rij
        float dr = -fcp12 * bp13 * fa13 * g123 * fc13 * d12inv;

        float cos_d = x13 * one_over_d12d13 - x12 * cos123_over_d12d12;
        f12x += (x12 * dr + dc * cos_d) * 0.5f;
        cos_d = y13 * one_over_d12d13 - y12 * cos123_over_d12d12;
        f12y += (y12 * dr + dc * cos_d) * 0.5f;
        cos_d = z13 * one_over_d12d13 - z12 * cos123_over_d12d12;
        f12z += (z12 * dr + dc * cos_d) * 0.5f;
      }
      g_f12x[index] = f12x;
      g_f12y[index] = f12y;
      g_f12z[index] = f12z;
    }
    // save potential
    g_potential[n1] = pot_energy;
  }
}

static __global__ void find_force_tersoff_step3(
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
  float* g_virial)
{
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  int n1 = N1 + threadIdx.x;
  if (n1 < N2) {
    float s_fx = 0.0f; // force_x
    float s_fy = 0.0f; // force_y
    float s_fz = 0.0f; // force_z
    float s_virial_xx = 0.0f;
    float s_virial_yy = 0.0f;
    float s_virial_zz = 0.0f;
    float s_virial_xy = 0.0f;
    float s_virial_yz = 0.0f;
    float s_virial_zx = 0.0f;
    const float* __restrict__ h = g_box + 18 * blockIdx.x;
    int neighbor_number = g_neighbor_number[n1];
    float x1 = g_x[n1];
    float y1 = g_y[n1];
    float z1 = g_z[n1];

    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * number_of_particles + n1;
      int n2 = g_neighbor_list[index];
      int neighbor_number_2 = g_neighbor_number[n2];

      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      dev_apply_mic(h, x12, y12, z12);

      float f12x = g_f12x[index];
      float f12y = g_f12y[index];
      float f12z = g_f12z[index];
      int offset = 0;
      for (int k = 0; k < neighbor_number_2; ++k) {
        if (n1 == g_neighbor_list[n2 + number_of_particles * k]) {
          offset = k;
          break;
        }
      }
      index = offset * number_of_particles + n2;
      float f21x = g_f12x[index];
      float f21y = g_f12y[index];
      float f21z = g_f12z[index];

      // per atom force
      s_fx += f12x - f21x;
      s_fy += f12y - f21y;
      s_fz += f12z - f21z;

      // per-atom virial
      s_virial_xx -= x12 * (f12x - f21x) * 0.5f;
      s_virial_yy -= y12 * (f12y - f21y) * 0.5f;
      s_virial_zz -= z12 * (f12z - f21z) * 0.5f;
      s_virial_xy -= x12 * (f12y - f21y) * 0.5f;
      s_virial_yz -= y12 * (f12z - f21z) * 0.5f;
      s_virial_zx -= z12 * (f12x - f21x) * 0.5f;
    }
    // save force
    g_fx[n1] = s_fx;
    g_fy[n1] = s_fy;
    g_fz[n1] = s_fz;
    // save virial
    g_virial[n1] = s_virial_xx;
    g_virial[n1 + number_of_particles] = s_virial_yy;
    g_virial[n1 + number_of_particles * 2] = s_virial_zz;
    g_virial[n1 + number_of_particles * 3] = s_virial_xy;
    g_virial[n1 + number_of_particles * 4] = s_virial_yz;
    g_virial[n1 + number_of_particles * 5] = s_virial_zx;
  }
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
  find_force_tersoff_step1<<<Nc, max_Na>>>(
    N, Na, Na_sum, neighbor->NN, neighbor->NL, type, pot_para, r, r + N, r + N * 2, h, b.data(),
    bp.data());
  CUDA_CHECK_KERNEL
  find_force_tersoff_step2<<<Nc, max_Na>>>(
    N, Na, Na_sum, neighbor->NN, neighbor->NL, type, pot_para, b.data(), bp.data(), r, r + N,
    r + N * 2, h, pe.data(), f12x.data(), f12y.data(), f12z.data());
  CUDA_CHECK_KERNEL
  find_force_tersoff_step3<<<Nc, max_Na>>>(
    N, Na, Na_sum, neighbor->NN, neighbor->NL, f12x.data(), f12y.data(), f12z.data(), r, r + N,
    r + N * 2, h, f.data(), f.data() + N, f.data() + N * 2, virial.data());
  CUDA_CHECK_KERNEL
}
