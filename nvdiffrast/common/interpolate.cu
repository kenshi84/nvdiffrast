// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "common.h"
#include "interpolate.h"

//------------------------------------------------------------------------
// Forward kernel.

template <bool ENABLE_DA, typename floatT>
static __forceinline__ __device__ void InterpolateFwdKernelTemplate(const InterpolateKernelParamsT<floatT> p)
{
    using floatT2 = typename float_trait<floatT>::vec2;
    using floatT3 = typename float_trait<floatT>::vec3;
    using floatT4 = typename float_trait<floatT>::vec4;
    auto make_floatT2 = float_trait<floatT>::make_vec2;
    auto make_floatT3 = float_trait<floatT>::make_vec3;
    auto make_floatT4 = float_trait<floatT>::make_vec4;

    // Calculate pixel position.
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= p.width || py >= p.height || pz >= p.depth)
        return;

    // Pixel index.
    int pidx = px + p.width * (py + p.height * pz);

    // Output ptrs.
    floatT* out = p.out + pidx * p.numAttr;
    floatT2* outDA = ENABLE_DA ? (((floatT2*)p.outDA) + pidx * p.numDiffAttr) : 0;

    // Fetch rasterizer output.
    float4 r = ((float4*)p.rast)[pidx];
    int triIdx = float_to_triidx(r.w) - 1;
    bool triValid = (triIdx >= 0 && triIdx < p.numTriangles);

    // If no geometry in entire warp, zero the output and exit.
    // Otherwise force barys to zero and output with live threads.
    if (__all_sync(0xffffffffu, !triValid))
    {
        for (int i=0; i < p.numAttr; i++)
            out[i] = 0.f;
        if (ENABLE_DA)
            for (int i=0; i < p.numDiffAttr; i++)
                outDA[i] = make_floatT2(0.f, 0.f);
        return;
    }

    // Fetch vertex indices.
    int vi0 = triValid ? p.tri[triIdx * 3 + 0] : 0;
    int vi1 = triValid ? p.tri[triIdx * 3 + 1] : 0;
    int vi2 = triValid ? p.tri[triIdx * 3 + 2] : 0;

    // Bail out if corrupt indices.
    if (vi0 < 0 || vi0 >= p.numVertices ||
        vi1 < 0 || vi1 >= p.numVertices ||
        vi2 < 0 || vi2 >= p.numVertices)
        return;

    // In instance mode, adjust vertex indices by minibatch index unless broadcasting.
    if (p.instance_mode && !p.attrBC)
    {
        vi0 += pz * p.numVertices;
        vi1 += pz * p.numVertices;
        vi2 += pz * p.numVertices;
    }

    // Pointers to attributes.
    const floatT* a0 = p.attr + vi0 * p.numAttr;
    const floatT* a1 = p.attr + vi1 * p.numAttr;
    const floatT* a2 = p.attr + vi2 * p.numAttr;

    // Barys. If no triangle, force all to zero -> output is zero.
    float b0 = triValid ? r.x : 0.f;
    float b1 = triValid ? r.y : 0.f;
    float b2 = triValid ? (1.f - r.x - r.y) : 0.f;

    // Interpolate and write attributes.
    for (int i=0; i < p.numAttr; i++)
        out[i] = b0*a0[i] + b1*a1[i] + b2*a2[i];

    // No diff attrs? Exit.
    if (!ENABLE_DA)
        return;

    // Read bary pixel differentials if we have a triangle.
    float4 db = make_float4(0.f, 0.f, 0.f, 0.f);
    if (triValid)
        db = ((float4*)p.rastDB)[pidx];

    // Unpack a bit.
    float dudx = db.x;
    float dudy = db.y;
    float dvdx = db.z;
    float dvdy = db.w;

    // Calculate the pixel differentials of chosen attributes.    
    for (int i=0; i < p.numDiffAttr; i++)
    {   
        // Input attribute index.
        int j = p.diff_attrs_all ? i : p.diffAttrs[i];
        if (j < 0)
            j += p.numAttr; // Python-style negative indices.

        // Zero output if invalid index.
        floatT dsdx = 0.f;
        floatT dsdy = 0.f;
        if (j >= 0 && j < p.numAttr)
        {
            floatT s0 = a0[j];
            floatT s1 = a1[j];
            floatT s2 = a2[j];
            floatT dsdu = s0 - s2;
            floatT dsdv = s1 - s2;
            dsdx = dudx*dsdu + dvdx*dsdv;
            dsdy = dudy*dsdu + dvdy*dsdv;
        }

        // Write.
        outDA[i] = make_floatT2(dsdx, dsdy);
    }
}

// Template specializations.
__global__ void InterpolateFwdKernel  (const InterpolateKernelParams p) { InterpolateFwdKernelTemplate<false, float>(p); }
__global__ void InterpolateFwdKernelDa(const InterpolateKernelParams p) { InterpolateFwdKernelTemplate<true, float>(p); }
__global__ void InterpolateFwdKernel64  (const InterpolateKernelParams64 p) { InterpolateFwdKernelTemplate<false, double>(p); }
__global__ void InterpolateFwdKernelDa64(const InterpolateKernelParams64 p) { InterpolateFwdKernelTemplate<true, double>(p); }

//------------------------------------------------------------------------
// Gradient kernel.

template <bool ENABLE_DA, typename floatT>
static __forceinline__ __device__ void InterpolateGradKernelTemplate(const InterpolateKernelParamsT<floatT> p)
{
    using floatT2 = typename float_trait<floatT>::vec2;
    using floatT3 = typename float_trait<floatT>::vec3;
    using floatT4 = typename float_trait<floatT>::vec4;
    auto make_floatT2 = float_trait<floatT>::make_vec2;
    auto make_floatT3 = float_trait<floatT>::make_vec3;
    auto make_floatT4 = float_trait<floatT>::make_vec4;

    // Temporary space for coalesced atomics.
    CA_DECLARE_TEMP(IP_GRAD_MAX_KERNEL_BLOCK_WIDTH * IP_GRAD_MAX_KERNEL_BLOCK_HEIGHT);

    // Calculate pixel position.
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= p.width || py >= p.height || pz >= p.depth)
        return;

    // Pixel index.
    int pidx = px + p.width * (py + p.height * pz);

    // Fetch triangle ID. If none, output zero bary/db gradients and exit.
    float4 r = ((float4*)p.rast)[pidx];
    int triIdx = float_to_triidx(r.w) - 1;
    if (triIdx < 0 || triIdx >= p.numTriangles)
    {
        ((floatT4*)p.gradRaster)[pidx] = make_floatT4(0.f, 0.f, 0.f, 0.f);
        if (ENABLE_DA)
            ((floatT4*)p.gradRasterDB)[pidx] = make_floatT4(0.f, 0.f, 0.f, 0.f);
        return;
    }

    // Fetch vertex indices.
    int vi0 = p.tri[triIdx * 3 + 0];
    int vi1 = p.tri[triIdx * 3 + 1];
    int vi2 = p.tri[triIdx * 3 + 2];

    // Bail out if corrupt indices.
    if (vi0 < 0 || vi0 >= p.numVertices ||
        vi1 < 0 || vi1 >= p.numVertices ||
        vi2 < 0 || vi2 >= p.numVertices)
        return;

    // In instance mode, adjust vertex indices by minibatch index unless broadcasting.
    if (p.instance_mode && !p.attrBC)
    {
        vi0 += pz * p.numVertices;
        vi1 += pz * p.numVertices;
        vi2 += pz * p.numVertices;
    }

    // Initialize coalesced atomics.
    CA_SET_GROUP(triIdx);

    // Pointers to inputs.
    const floatT* a0 = p.attr + vi0 * p.numAttr;
    const floatT* a1 = p.attr + vi1 * p.numAttr;
    const floatT* a2 = p.attr + vi2 * p.numAttr;
    const floatT* pdy = p.dy + pidx * p.numAttr;

    // Pointers to outputs.
    floatT* ga0 = p.gradAttr + vi0 * p.numAttr;
    floatT* ga1 = p.gradAttr + vi1 * p.numAttr;
    floatT* ga2 = p.gradAttr + vi2 * p.numAttr;

    // Barys and bary gradient accumulators.
    float b0 = r.x;
    float b1 = r.y;
    float b2 = 1.f - r.x - r.y;
    floatT gb0 = 0.f;
    floatT gb1 = 0.f;

    // Loop over attributes and accumulate attribute gradients.
    for (int i=0; i < p.numAttr; i++)
    {
        floatT y = pdy[i];
        floatT s0 = a0[i];
        floatT s1 = a1[i];
        floatT s2 = a2[i];
        gb0 += y * (s0 - s2);
        gb1 += y * (s1 - s2);
        caAtomicAdd(ga0 + i, b0 * y);
        caAtomicAdd(ga1 + i, b1 * y);
        caAtomicAdd(ga2 + i, b2 * y);
    }

    // Write the bary gradients.
    ((floatT4*)p.gradRaster)[pidx] = make_floatT4(gb0, gb1, 0.f, 0.f);

    // If pixel differentials disabled, we're done.
    if (!ENABLE_DA)
        return;

    // Calculate gradients based on attribute pixel differentials.
    const floatT2* dda = ((floatT2*)p.dda) + pidx * p.numDiffAttr;
    floatT gdudx = 0.f;
    floatT gdudy = 0.f;
    floatT gdvdx = 0.f;
    floatT gdvdy = 0.f;

    // Read bary pixel differentials.
    float4 db = ((float4*)p.rastDB)[pidx];
    float dudx = db.x;
    float dudy = db.y;
    float dvdx = db.z;
    float dvdy = db.w;

    for (int i=0; i < p.numDiffAttr; i++)
    {
        // Input attribute index.
        int j = p.diff_attrs_all ? i : p.diffAttrs[i];
        if (j < 0)
            j += p.numAttr; // Python-style negative indices.

        // Check that index is valid.
        if (j >= 0 && j < p.numAttr)
        {
            floatT2 dsdxy = dda[i];
            floatT dsdx = dsdxy.x;
            floatT dsdy = dsdxy.y;

            floatT s0 = a0[j];
            floatT s1 = a1[j];
            floatT s2 = a2[j];

            // Gradients of db.
            floatT dsdu = s0 - s2;
            floatT dsdv = s1 - s2;
            gdudx += dsdu * dsdx;
            gdudy += dsdu * dsdy;
            gdvdx += dsdv * dsdx;
            gdvdy += dsdv * dsdy;

            // Gradients of attributes.
            floatT du = dsdx*dudx + dsdy*dudy;
            floatT dv = dsdx*dvdx + dsdy*dvdy;
            caAtomicAdd(ga0 + j, du);
            caAtomicAdd(ga1 + j, dv);
            caAtomicAdd(ga2 + j, -du - dv);
        }
    }

    // Write.
    ((floatT4*)p.gradRasterDB)[pidx] = make_floatT4(gdudx, gdudy, gdvdx, gdvdy);
}

// Template specializations.
__global__ void InterpolateGradKernel  (const InterpolateKernelParams p) { InterpolateGradKernelTemplate<false, float>(p); }
__global__ void InterpolateGradKernelDa(const InterpolateKernelParams p) { InterpolateGradKernelTemplate<true, float>(p); }
__global__ void InterpolateGradKernel64  (const InterpolateKernelParams64 p) { InterpolateGradKernelTemplate<false, double>(p); }
__global__ void InterpolateGradKernelDa64(const InterpolateKernelParams64 p) { InterpolateGradKernelTemplate<true, double>(p); }

//------------------------------------------------------------------------
