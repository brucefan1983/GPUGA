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
Calculate force, energy, and virial for RI (rigid-ion) potential
------------------------------------------------------------------------------*/

#include "error.cuh"
#include "gpu_vector.cuh"
#include "mic.cuh"
#include "neighbor.cuh"
#include "ri.cuh"

// Easy labels for indexing
const int Q0 = 0;
const int A00 = 1;
const int B00 = 2;
const int C00 = 3;
const int A11 = 4;
const int B11 = 5;
const int C11 = 6;
const int A01 = 7;
const int B01 = 8;
const int C01 = 9;

#define RI_ALPHA 0.2f
#define RI_ALPHA_SQ 0.04f
#define RI_PI_FACTOR 0.225675833419103f // ALPHA * 2 / SQRT(PI)
#define K_C 14.399645f                  // 1/(4*PI*epsilon_0)

void RI::initialize(int N, int MAX_ATOM_NUMBER)
{
  // nothing
}

void RI::update_potential(const std::vector<float>& potential_parameters)
{
  float q0 = potential_parameters[Q0];
  ri_para.cutoff = 10.0f;
  ri_para.qq[0] = q0 * q0 * K_C;
  ri_para.qq[1] = -2.0f * q0 * q0 * K_C;
  ri_para.qq[2] = 4.0f * q0 * q0 * K_C;
  ri_para.a[0] = potential_parameters[A00];
  ri_para.a[1] = potential_parameters[A01];
  ri_para.a[2] = potential_parameters[A11];
  ri_para.b[0] = 1.0f / potential_parameters[B00]; // from rho to b
  ri_para.b[1] = 1.0f / potential_parameters[B01]; // from rho to b
  ri_para.b[2] = 1.0f / potential_parameters[B11]; // from rho to b
  ri_para.c[0] = potential_parameters[C00];
  ri_para.c[1] = potential_parameters[C01];
  ri_para.c[2] = potential_parameters[C11];
  ri_para.v_rc = erfc(RI_ALPHA * ri_para.cutoff) / ri_para.cutoff;
  ri_para.dv_rc = -erfc(RI_ALPHA * ri_para.cutoff) / (ri_para.cutoff * ri_para.cutoff);
  ri_para.dv_rc -=
    RI_PI_FACTOR * exp(-RI_ALPHA_SQ * ri_para.cutoff * ri_para.cutoff) / ri_para.cutoff;
}

// get U_ij and (d U_ij / d r_ij) / r_ij (the RI potential)
static __device__ void find_p2_and_f2(int type12, RI_Para ri_para, float d12, float& p2, float& f2)
{
  float d12sq = d12 * d12;
  float d12inv = 1.0f / d12;
  float d12inv3 = d12inv * d12inv * d12inv;
  float d12inv6 = d12inv3 * d12inv3;
  float d12inv7 = d12inv6 * d12inv;
  float exponential = exp(-d12 * ri_para.b[type12]); // b = 1/rho
  float erfc_r = erfc(RI_ALPHA * d12) * d12inv;
  p2 = ri_para.a[type12] * exponential - ri_para.c[type12] * d12inv3 * d12inv3;
  p2 += ri_para.qq[type12] * (erfc_r - ri_para.v_rc - ri_para.dv_rc * (d12 - ri_para.cutoff));
  f2 = 6.0f * ri_para.c[type12] * d12inv7 - ri_para.a[type12] * exponential * ri_para.b[type12];
  f2 -= ri_para.qq[type12] *
        (erfc_r * d12inv + RI_PI_FACTOR * d12inv * exp(-RI_ALPHA_SQ * d12sq) + ri_para.dv_rc);
  f2 *= d12inv;
}

static __global__ void find_force_2body(
  int number_of_particles,
  int* Na,
  int* Na_sum,
  int* g_neighbor_number,
  int* g_neighbor_list,
  int* g_type,
  RI_Para ri_para,
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
    int type1 = g_type[n1];

    float x1 = g_x[n1];
    float y1 = g_y[n1];
    float z1 = g_z[n1];

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
      int type12 = type1 + g_type[n2];

      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      dev_apply_mic(h, x12, y12, z12);
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);

      float p2, f2;
      find_p2_and_f2(type12, ri_para, d12, p2, f2);

      fx += x12 * f2;
      fy += y12 * f2;
      fz += z12 * f2;
      virial_xx -= x12 * x12 * f2 * 0.5f;
      virial_yy -= y12 * y12 * f2 * 0.5f;
      virial_zz -= z12 * z12 * f2 * 0.5f;
      virial_xy -= x12 * y12 * f2 * 0.5f;
      virial_yz -= y12 * z12 * f2 * 0.5f;
      virial_zx -= z12 * x12 * f2 * 0.5f;
      pe += p2 * 0.5f;
    }

    g_fx[n1] = fx;
    g_fy[n1] = fy;
    g_fz[n1] = fz;
    g_virial[n1 + number_of_particles * 0] = virial_xx;
    g_virial[n1 + number_of_particles * 1] = virial_yy;
    g_virial[n1 + number_of_particles * 2] = virial_zz;
    g_virial[n1 + number_of_particles * 3] = virial_xy;
    g_virial[n1 + number_of_particles * 4] = virial_yz;
    g_virial[n1 + number_of_particles * 5] = virial_zx;
    g_pe[n1] = pe;
  }
}

void RI::find_force(
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
    N, Na, Na_sum, neighbor->NN, neighbor->NL, type, ri_para, r, r + N, r + N * 2, h, f.data(),
    f.data() + N, f.data() + N * 2, virial.data(), pe.data());
  CUDA_CHECK_KERNEL
}
