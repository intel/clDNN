/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "pooling_kernel_ref.h"
#include "kernel_selector_utils.h" 

namespace KernelSelector 
{
    ParamsKey PoolingKernelRef::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableInputLayout(DataLayout::bfyx);
        k.EnableOutputLayout(DataLayout::bfyx);
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBatching();
        k.EnablePoolType(PoolType::MAX);
        k.EnablePoolType(PoolType::AVG);
        k.EnablePoolRemainder(PoolRemainder::FLOOR);
        k.EnablePoolRemainder(PoolRemainder::CEIL);
        k.EnablePoolKernelDividerMode(KernelDividerMode::FIXED);
        k.EnablePoolKernelDividerMode(KernelDividerMode::DYNAMIC);
        return k;
    }

    KernelsData PoolingKernelRef::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        assert(params.GetType() == KernelType::POOLING);

        KernelData kd = KernelData::Default<PoolingParams>(params);

        PoolingParams& newParams = *static_cast<PoolingParams*>(kd.params.get());
        const auto& pp = newParams.poolParams;
        
        const std::string kernel_id = GetEntryPoint(kernelName, params.layerID, options);
        std::stringstream jit;
        jit << GetBaseJit(newParams, kernel_id)
            << "#define POOL_SIZE_X (" << pp.poolSize.x << ")\n"
            << "#define POOL_SIZE_Y (" << pp.poolSize.y << ")\n"
            << "#define POOL_PAD_X (" << pp.poolPad.x << ")\n"
            << "#define POOL_PAD_Y (" << pp.poolPad.y << ")\n"
            << "#define POOL_STRIDE_X (" << pp.poolStride.x << ")\n"
            << "#define POOL_STRIDE_Y (" << pp.poolStride.y << ")\n";

        jit << "#define " << toString(pp.poolType) << "_POOLING\n";
        jit << "#define " << toString(pp.divMode) << "_KERNEL_DIVIDER\n";

        const auto& out = newParams.output;
        auto& kernel = kd.kernels[0];
        kernel.workGroups.global = { out.X().v, out.Y().v, out.Feature().v*out.Batch().v };
        kernel.workGroups.local = GetOptimalLocalWorkGroupSizes(kernel.workGroups.global);
        kernel.kernelString = GetKernelString(kernelName, jit.str(), kernel_id);
        kernel.arguments = GetArgumentDesc(1, false, false);

        kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}