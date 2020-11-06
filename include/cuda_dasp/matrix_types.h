#pragma once

#include <cuda_runtime.h>


struct Cov3
{
  float xx, xy, xz,
    yy, yz,
    zz;
};

struct Mat33
{
  float3 rows[3];
};

struct Transform3
{
    Mat33 R;
    float3 t;
};
