#ifndef HIP_BLASLT_COMPAT_H
#define HIP_BLASLT_COMPAT_H
/*
From hipblaslt headers

Copyright (C) 2022 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#include <hip/hip_fp16.h>
#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>
#include "hip/amd_detail/amd_hip_bfloat16.h"

typedef struct {
  uint64_t data[4];
} hipblasLtMatrixLayoutOpaque_t;

typedef enum {
  HIPBLASLT_R_16F = 150,     /**< 16 bit floating point, real */
  HIPBLASLT_R_32F = 151,     /**< 32 bit floating point, real */
  HIPBLASLT_R_64F = 152,     /**< 64 bit floating point, real */
  HIPBLASLT_C_16F = 153,     /**< 16 bit floating point, complex */
  HIPBLASLT_C_32F = 154,     /**< 32 bit floating point, complex */
  HIPBLASLT_C_64F = 155,     /**< 64 bit floating point, complex */
  HIPBLASLT_R_8I = 160,      /**<  8 bit signed integer, real */
  HIPBLASLT_R_8U = 161,      /**<  8 bit unsigned integer, real */
  HIPBLASLT_R_32I = 162,     /**< 32 bit signed integer, real */
  HIPBLASLT_R_32U = 163,     /**< 32 bit unsigned integer, real */
  HIPBLASLT_C_8I = 164,      /**<  8 bit signed integer, complex */
  HIPBLASLT_C_8U = 165,      /**<  8 bit unsigned integer, complex */
  HIPBLASLT_C_32I = 166,     /**< 32 bit signed integer, complex */
  HIPBLASLT_C_32U = 167,     /**< 32 bit unsigned integer, complex */
  HIPBLASLT_R_16B = 168,     /**< 16 bit bfloat, real */
  HIPBLASLT_C_16B = 169,     /**< 16 bit bfloat, complex */
  HIPBLASLT_R_8F_E4M3 = 170, /**< 8 bit floating point in E4M3 format, real */
  HIPBLASLT_R_8F_E5M2 = 171, /**< 8 bit floating point in E5M2 format, real */
  HIPBLASLT_DATATYPE_INVALID = 255, /**< Invalid datatype value, do not use */
} hipblasltDatatype_t;

#define hipblasLtMatrixLayout_t hipblasLtMatrixLayoutOpaque_t

hipblasStatus_t hipblasLtMatrixLayoutCreate(hipblasLtMatrixLayout_t *matLayout,
                                            hipblasltDatatype_t type,
                                            uint64_t rows, uint64_t cols,
                                            int64_t ld);

#define hipblasLtHandle_t void *

hipblasStatus_t hipblasLtCreate(hipblasLtHandle_t *handle);

#endif