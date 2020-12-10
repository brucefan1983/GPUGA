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
#include <stdio.h>


#define MY_MALLOC(p, t, n) p = (t *) malloc(sizeof(t) * (n));                  \
                           if(p == NULL)                                       \
                           {                                                   \
                               printf("Failed to allocate!\n");                \
                               exit(EXIT_FAILURE);                             \
                           }


#define MY_FREE(p) if(p != NULL)                                               \
                   {                                                           \
                       free(p);                                                \
                       p = NULL;                                               \
                   }                                                           \
                   else                                                        \
                   {                                                           \
                       printf("Try to free NULL!\n");                          \
                       exit(EXIT_FAILURE);                                     \
                   }


#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error_code = call;                                       \
    if (error_code != cudaSuccess)                                             \
    {                                                                          \
        fprintf(stderr, "CUDA Error:\n");                                      \
        fprintf(stderr, "    File:       %s\n", __FILE__);                     \
        fprintf(stderr, "    Line:       %d\n", __LINE__);                     \
        fprintf(stderr, "    Error code: %d\n", error_code);                   \
        fprintf(stderr, "    Error text: %s\n",                                \
            cudaGetErrorString(error_code));                                   \
        exit(1);                                                               \
    }                                                                          \
}


#define CUDA_CHECK_KERNEL                                                      \
{                                                                              \
    CHECK(cudaGetLastError())                                                  \
}


void print_error (const char *str);
void print_line_1(void);
void print_line_2(void);
FILE* my_fopen(const char *filename, const char *mode);


