// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "torch_common.inl"
#include "../common/common.h"
#include "../common/interpolate.h"

//------------------------------------------------------------------------
// Kernel prototypes.

void InterpolateFwdKernel   (const InterpolateKernelParams p);
void InterpolateFwdKernelDa (const InterpolateKernelParams p);
void InterpolateFwdKernel64   (const InterpolateKernelParams64 p);
void InterpolateFwdKernelDa64 (const InterpolateKernelParams64 p);
void InterpolateGradKernel  (const InterpolateKernelParams p);
void InterpolateGradKernelDa(const InterpolateKernelParams p);
void InterpolateGradKernel64  (const InterpolateKernelParams64 p);
void InterpolateGradKernelDa64(const InterpolateKernelParams64 p);

//------------------------------------------------------------------------
// Helper

template <typename floatT>
static void set_diff_attrs(InterpolateKernelParamsT<floatT>& p, bool diff_attrs_all, std::vector<int>& diff_attrs_vec)
{
    if (diff_attrs_all)
    {
        p.numDiffAttr = p.numAttr;
        p.diff_attrs_all = 1;
    }
    else
    {
        NVDR_CHECK(diff_attrs_vec.size() <= IP_MAX_DIFF_ATTRS, "too many entries in diff_attrs list (increase IP_MAX_DIFF_ATTRS)");
        p.numDiffAttr = diff_attrs_vec.size();
        memcpy(p.diffAttrs, &diff_attrs_vec[0], diff_attrs_vec.size()*sizeof(int));
    }
}

//------------------------------------------------------------------------
// Forward op.

std::tuple<torch::Tensor, torch::Tensor> interpolate_fwd_da(torch::Tensor attr, torch::Tensor rast, torch::Tensor tri, torch::Tensor rast_db, bool diff_attrs_all, std::vector<int>& diff_attrs_vec)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(attr));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    InterpolateKernelParams p = {}; // Initialize all fields to zero.
    InterpolateKernelParams64 p64 = {};
    bool enable_da = (rast_db.defined()) && (diff_attrs_all || !diff_attrs_vec.empty());
    bool use_double = attr.scalar_type() == torch::kFloat64;
    p.instance_mode = p64.instance_mode = (attr.sizes().size() > 2) ? 1 : 0;

    // Check inputs.
    if (enable_da)
    {
        NVDR_CHECK_DEVICE(attr, rast, tri, rast_db);
        NVDR_CHECK_CONTIGUOUS(attr, rast, tri, rast_db);
        NVDR_CHECK_F32(rast, rast_db);
        NVDR_CHECK_I32(tri);
    }
    else
    {
        NVDR_CHECK_DEVICE(attr, rast, tri);
        NVDR_CHECK_CONTIGUOUS(attr, rast, tri);
        NVDR_CHECK_F32(rast);
        NVDR_CHECK_I32(tri);
    }
    if (use_double)
        NVDR_CHECK_F64(attr);
    else
        NVDR_CHECK_F32(attr);

    // Sanity checks.
    NVDR_CHECK(rast.sizes().size() == 4 && rast.size(0) > 0 && rast.size(1) > 0 && rast.size(2) > 0 && rast.size(3) == 4, "rast must have shape[>0, >0, >0, 4]");
    NVDR_CHECK( tri.sizes().size() == 2 && tri.size(0) > 0 && tri.size(1) == 3, "tri must have shape [>0, 3]");
    NVDR_CHECK((attr.sizes().size() == 2 || attr.sizes().size() == 3) && attr.size(0) > 0 && attr.size(1) > 0 && (attr.sizes().size() == 2 || attr.size(2) > 0), "attr must have shape [>0, >0, >0] or [>0, >0]");
    if (p.instance_mode)
        NVDR_CHECK(attr.size(0) == rast.size(0) || attr.size(0) == 1, "minibatch size mismatch between inputs rast, attr");
    if (enable_da)
    {
        NVDR_CHECK(rast_db.sizes().size() == 4 && rast_db.size(0) > 0 && rast_db.size(1) > 0 && rast_db.size(2) > 0 && rast_db.size(3) == 4, "rast_db must have shape[>0, >0, >0, 4]");
        NVDR_CHECK(rast_db.size(1) == rast.size(1) && rast_db.size(2) == rast.size(2), "spatial size mismatch between inputs rast and rast_db");
        NVDR_CHECK(rast_db.size(0) == rast.size(0), "minibatch size mismatch between inputs rast, rast_db");
    }

    // Extract input dimensions.
    p.numVertices  = p64.numVertices  = attr.size(p.instance_mode ? 1 : 0);
    p.numAttr      = p64.numAttr      = attr.size(p.instance_mode ? 2 : 1);
    p.numTriangles = p64.numTriangles = tri.size(0);
    p.height       = p64.height       = rast.size(1);
    p.width        = p64.width        = rast.size(2);
    p.depth        = p64.depth        = rast.size(0);

    // Set attribute pixel differential info if enabled, otherwise leave as zero.
    if (enable_da)
    {
        if (use_double)
            set_diff_attrs(p64, diff_attrs_all, diff_attrs_vec);
        else
            set_diff_attrs(p, diff_attrs_all, diff_attrs_vec);
    }
    else
        p.numDiffAttr = p64.numDiffAttr = 0;

    // Get input pointers.
    p.attr = !use_double ? attr.data_ptr<float>() : NULL;
    p64.attr = use_double ? attr.data_ptr<double>() : NULL;
    p.rast = p64.rast = rast.data_ptr<float>();
    p.tri = p64.tri = tri.data_ptr<int>();
    p.rastDB = p64.rastDB = enable_da ? rast_db.data_ptr<float>() : NULL;
    p.attrBC = p64.attrBC = (p.instance_mode && attr.size(0) == 1) ? 1 : 0;

    // Allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(use_double ? torch::kFloat64 : torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out = torch::empty({p.depth, p.height, p.width, p.numAttr}, opts);
    torch::Tensor out_da = torch::empty({p.depth, p.height, p.width, p.numDiffAttr * 2}, opts);

    p.out = !use_double ? out.data_ptr<float>() : NULL;
    p64.out = use_double ? out.data_ptr<double>() : NULL;
    p.outDA = enable_da && !use_double ? out_da.data_ptr<float>() : NULL;
    p64.outDA = enable_da && use_double ? out_da.data_ptr<double>() : NULL;

    // Verify that buffers are aligned to allow float2/float4 operations.
    NVDR_CHECK(!((uintptr_t)p.rast   & 15), "rast input tensor not aligned to float4");
    NVDR_CHECK(!((uintptr_t)p.rastDB & 15), "rast_db input tensor not aligned to float4");
    NVDR_CHECK(!((uintptr_t)p.outDA  &  7), "out_da output tensor not aligned to float2");
    NVDR_CHECK(!((uintptr_t)p64.outDA & 15), "out_da output tensor not aligned to double2");

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(IP_FWD_MAX_KERNEL_BLOCK_WIDTH, IP_FWD_MAX_KERNEL_BLOCK_HEIGHT, p.width, p.height);
    dim3 gridSize  = getLaunchGridSize(blockSize, p.width, p.height, p.depth);

    // Launch CUDA kernel.
    void* args[] = {use_double ? (void*)&p64 : (void*)&p};
    void* func;
    if (use_double)
        func = enable_da ? (void*)InterpolateFwdKernelDa64 : (void*)InterpolateFwdKernel64;
    else
        func = enable_da ? (void*)InterpolateFwdKernelDa : (void*)InterpolateFwdKernel;
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel(func, gridSize, blockSize, args, 0, stream));

    // Return results.
    return std::tuple<torch::Tensor, torch::Tensor>(out, out_da);
}

// Version without derivatives.
std::tuple<torch::Tensor, torch::Tensor> interpolate_fwd(torch::Tensor attr, torch::Tensor rast, torch::Tensor tri)
{
    std::vector<int> empty_vec;
    torch::Tensor empty_tensor;
    return interpolate_fwd_da(attr, rast, tri, empty_tensor, false, empty_vec);
}

//------------------------------------------------------------------------
// Gradient op.

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> interpolate_grad_da(torch::Tensor attr, torch::Tensor rast, torch::Tensor tri, torch::Tensor dy, torch::Tensor rast_db, torch::Tensor dda, bool diff_attrs_all, std::vector<int>& diff_attrs_vec)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(attr));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    InterpolateKernelParams p = {}; // Initialize all fields to zero.
    InterpolateKernelParams64 p64 = {};
    bool enable_da = (rast_db.defined()) && (diff_attrs_all || !diff_attrs_vec.empty());
    bool use_double = attr.scalar_type() == torch::kFloat64;
    p.instance_mode = p64.instance_mode = (attr.sizes().size() > 2) ? 1 : 0;

    // Check inputs.
    if (enable_da)
    {
        NVDR_CHECK_DEVICE(attr, rast, tri, dy, rast_db, dda);
        NVDR_CHECK_CONTIGUOUS(attr, rast, tri, rast_db);
        NVDR_CHECK_F32(rast, rast_db);
        NVDR_CHECK_I32(tri);
        if (use_double)
            NVDR_CHECK_F64(dda);
        else
            NVDR_CHECK_F32(dda);
    }
    else
    {
        NVDR_CHECK_DEVICE(attr, rast, tri, dy);
        NVDR_CHECK_CONTIGUOUS(attr, rast, tri);
        NVDR_CHECK_F32(rast);
        NVDR_CHECK_I32(tri);
    }
    if (use_double)
        NVDR_CHECK_F64(attr, dy);
    else
        NVDR_CHECK_F32(attr, dy);

    // Depth of attributes.
    int attr_depth = p.instance_mode ? (attr.sizes().size() > 1 ? attr.size(0) : 0) : 1;

    // Sanity checks.
    NVDR_CHECK(rast.sizes().size() == 4 && rast.size(0) > 0 && rast.size(1) > 0 && rast.size(2) > 0 && rast.size(3) == 4, "rast must have shape[>0, >0, >0, 4]");
    NVDR_CHECK(tri.sizes().size() == 2 && tri.size(0) > 0 && tri.size(1) == 3, "tri must have shape [>0, 3]");
    NVDR_CHECK((attr.sizes().size() == 2 || attr.sizes().size() == 3) && attr.size(0) > 0 && attr.size(1) > 0 && (attr.sizes().size() == 2 || attr.size(2) > 0), "attr must have shape [>0, >0, >0] or [>0, >0]");
    NVDR_CHECK(dy.sizes().size() == 4 && dy.size(0) > 0 && dy.size(1) == rast.size(1) && dy.size(2) == rast.size(2) && dy.size(3) > 0, "dy must have shape [>0, height, width, >0]");
    NVDR_CHECK(dy.size(3) == attr.size(attr.sizes().size() - 1), "argument count mismatch between inputs dy, attr");
    NVDR_CHECK((attr_depth == rast.size(0) || attr_depth == 1) && dy.size(0) == rast.size(0), "minibatch size mismatch between inputs rast, dy, attr");
    if (enable_da)
    {
        NVDR_CHECK(dda.sizes().size() == 4 && dda.size(0) > 0 && dda.size(1) == rast.size(1) && dda.size(2) == rast.size(2), "dda must have shape [>0, height, width, ?]");
        NVDR_CHECK(dda.size(0) == rast.size(0), "minibatch size mismatch between rast, dda");
        NVDR_CHECK(rast_db.sizes().size() == 4 && rast_db.size(0) > 0 && rast_db.size(1) > 0 && rast_db.size(2) > 0 && rast_db.size(3) == 4, "rast_db must have shape[>0, >0, >0, 4]");
        NVDR_CHECK(rast_db.size(1) == rast.size(1) && rast_db.size(2) == rast.size(2), "spatial size mismatch between inputs rast and rast_db");
        NVDR_CHECK(rast_db.size(0) == rast.size(0), "minibatch size mismatch between inputs rast, rast_db");
    }

    // Extract input dimensions.
    p.numVertices  = p64.numVertices  = attr.size(p.instance_mode ? 1 : 0);
    p.numAttr      = p64.numAttr      = attr.size(p.instance_mode ? 2 : 1);
    p.numTriangles = p64.numTriangles = tri.size(0);
    p.height       = p64.height       = rast.size(1);
    p.width        = p64.width        = rast.size(2);
    p.depth        = p64.depth        = rast.size(0);

    // Ensure gradients are contiguous.
    torch::Tensor dy_ = dy.contiguous();
    torch::Tensor dda_;
    if (enable_da)
        dda_ = dda.contiguous();

    // Set attribute pixel differential info if enabled, otherwise leave as zero.
    if (enable_da)
    {
        if (use_double)
            set_diff_attrs(p64, diff_attrs_all, diff_attrs_vec);
        else
            set_diff_attrs(p, diff_attrs_all, diff_attrs_vec);
    }
    else
        p.numDiffAttr = p64.numDiffAttr = 0;

    // Get input pointers.
    p.attr = !use_double ? attr.data_ptr<float>() : NULL;
    p64.attr = use_double ? attr.data_ptr<double>() : NULL;
    p.rast = p64.rast = rast.data_ptr<float>();
    p.tri = p64.tri = tri.data_ptr<int>();
    p.dy = !use_double ? dy_.data_ptr<float>() : NULL;
    p64.dy = use_double ? dy_.data_ptr<double>() : NULL;
    p.rastDB = p64.rastDB = enable_da ? rast_db.data_ptr<float>() : NULL;
    p.dda = enable_da && !use_double ? dda_.data_ptr<float>() : NULL;
    p64.dda = enable_da && use_double ? dda_.data_ptr<double>() : NULL;
    p.attrBC = p64.attrBC = (p.instance_mode && attr_depth < p.depth) ? 1 : 0;

    // Allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(use_double ? torch::kFloat64 : torch::kFloat32).device(torch::kCUDA);
    torch::Tensor gradAttr = torch::zeros_like(attr);
    torch::Tensor gradRaster = torch::empty(rast.sizes(), opts);
    torch::Tensor gradRasterDB;
    if (enable_da)
        gradRasterDB = torch::empty(rast_db.sizes(), opts);

    p.gradAttr = !use_double ? gradAttr.data_ptr<float>() : NULL;
    p64.gradAttr = use_double ? gradAttr.data_ptr<double>() : NULL;
    p.gradRaster = !use_double ? gradRaster.data_ptr<float>() : NULL;
    p64.gradRaster = use_double ? gradRaster.data_ptr<double>() : NULL;
    p.gradRasterDB = enable_da && !use_double ? gradRasterDB.data_ptr<float>() : NULL;
    p64.gradRasterDB = enable_da && use_double ? gradRasterDB.data_ptr<double>() : NULL;

    // Verify that buffers are aligned to allow float2/float4 operations.
    NVDR_CHECK(!((uintptr_t)p.rast         & 15), "rast input tensor not aligned to float4");
    NVDR_CHECK(!((uintptr_t)p.rastDB       & 15), "rast_db input tensor not aligned to float4");
    NVDR_CHECK(!((uintptr_t)p.dda          &  7), "dda input tensor not aligned to float2");
    NVDR_CHECK(!((uintptr_t)p64.dda        & 15), "dda input tensor not aligned to double2");
    NVDR_CHECK(!((uintptr_t)p.gradRaster   & 15), "grad_rast output tensor not aligned to float4");
    NVDR_CHECK(!((uintptr_t)p64.gradRaster & 31), "grad_rast output tensor not aligned to double4");
    NVDR_CHECK(!((uintptr_t)p.gradRasterDB & 15), "grad_rast_db output tensor not aligned to float4");
    NVDR_CHECK(!((uintptr_t)p64.gradRasterDB & 31), "grad_rast_db output tensor not aligned to double4");

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(IP_GRAD_MAX_KERNEL_BLOCK_WIDTH, IP_GRAD_MAX_KERNEL_BLOCK_HEIGHT, p.width, p.height);
    dim3 gridSize  = getLaunchGridSize(blockSize, p.width, p.height, p.depth);

    // Launch CUDA kernel.
    void* args[] = {use_double ? (void*)&p64 : (void*)&p};
    void* func;
    if (use_double)
        func = enable_da ? (void*)InterpolateGradKernelDa64 : (void*)InterpolateGradKernel64;
    else
        func = enable_da ? (void*)InterpolateGradKernelDa : (void*)InterpolateGradKernel;
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel(func, gridSize, blockSize, args, 0, stream));

    // Return results.
    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>(gradAttr, gradRaster, gradRasterDB);
}

// Version without derivatives.
std::tuple<torch::Tensor, torch::Tensor> interpolate_grad(torch::Tensor attr, torch::Tensor rast, torch::Tensor tri, torch::Tensor dy)
{
    std::vector<int> empty_vec;
    torch::Tensor empty_tensor;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> result = interpolate_grad_da(attr, rast, tri, dy, empty_tensor, empty_tensor, false, empty_vec);
    return std::tuple<torch::Tensor, torch::Tensor>(std::get<0>(result), std::get<1>(result));
}

//------------------------------------------------------------------------
