// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

//------------------------------------------------------------------------
// Constants and helpers.

#define IP_FWD_MAX_KERNEL_BLOCK_WIDTH   8
#define IP_FWD_MAX_KERNEL_BLOCK_HEIGHT  8
#define IP_GRAD_MAX_KERNEL_BLOCK_WIDTH  8
#define IP_GRAD_MAX_KERNEL_BLOCK_HEIGHT 8
#define IP_MAX_DIFF_ATTRS               32

//------------------------------------------------------------------------
// CUDA kernel params.

template <typename floatT>
struct InterpolateKernelParamsT
{
    const int*      tri;                            // Incoming triangle buffer.
    const floatT*    attr;                          // Incoming attribute buffer.
    const float*    rast;                           // Incoming rasterizer output buffer.
    const float*    rastDB;                         // Incoming rasterizer output buffer for bary derivatives.
    const floatT*    dy;                            // Incoming attribute gradients.
    const floatT*    dda;                           // Incoming attr diff gradients.
    floatT*          out;                           // Outgoing interpolated attributes.
    floatT*          outDA;                         // Outgoing texcoord major axis lengths.
    floatT*          gradAttr;                      // Outgoing attribute gradients.
    floatT*          gradRaster;                    // Outgoing rasterizer gradients.
    floatT*          gradRasterDB;                  // Outgoing rasterizer bary diff gradients.
    int             numTriangles;                   // Number of triangles.
    int             numVertices;                    // Number of vertices.
    int             numAttr;                        // Number of total vertex attributes.
    int             numDiffAttr;                    // Number of attributes to differentiate.
    int             width;                          // Image width.
    int             height;                         // Image height.
    int             depth;                          // Minibatch size.
    int             attrBC;                         // 0=normal, 1=attr is broadcast.
    int             instance_mode;                  // 0=normal, 1=instance mode.
    int             diff_attrs_all;                 // 0=normal, 1=produce pixel differentials for all attributes.
    int             diffAttrs[IP_MAX_DIFF_ATTRS];   // List of attributes to differentiate.
};
using InterpolateKernelParams = InterpolateKernelParamsT<float>;
using InterpolateKernelParams64 = InterpolateKernelParamsT<double>;
//------------------------------------------------------------------------
