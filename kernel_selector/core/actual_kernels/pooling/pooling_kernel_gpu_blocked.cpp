/*
// Copyright (c) 2018-2019 Intel Corporation
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

#include "pooling_kernel_gpu_blocked.h"

namespace kernel_selector
{
    static const size_t sub_group_size = 16;
    static const size_t feature_slice_size = 16;
    static const size_t x_block_size = 8;

    ParamsKey PoolingKernelBlocked::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableInputLayout(DataLayout::bfyx_f16);
        k.EnableOutputLayout(DataLayout::bfyx_f16);
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBatching();
        k.EnablePoolType(PoolType::MAX);
        k.EnablePoolType(PoolType::AVG);
        k.EnablePoolRemainder(PoolRemainder::FLOOR);
        k.EnablePoolRemainder(PoolRemainder::CEIL);
        k.EnablePoolKernelDividerMode(KernelDividerMode::FIXED);
        k.EnablePoolKernelDividerMode(KernelDividerMode::DYNAMIC);
        k.EnablePoolKernelDividerMode(KernelDividerMode::DYNAMIC_WITH_PADDING);
        k.EnableDifferentTypes();
        return k;
    }

    PoolingKernelBase::DispatchData PoolingKernelBlocked::SetDefault(const pooling_params& params) const
    {
        DispatchData kd = PoolingKernelBase::SetDefault(params);

        const auto& out = params.output;

        auto x = out.X().v;
        auto y = out.Y().v;
        auto f = out.Feature().v;
        auto b = out.Batch().v;

        kd.gws0 = CeilDiv(x, x_block_size) * y;
        kd.gws1 = Align(f, sub_group_size);
        kd.gws2 = b;

        kd.lws0 = 1;
        kd.lws1 = sub_group_size;
        kd.lws2 = 1;

        kd.effiency = FORCE_PRIORITY_2;

        return kd;
    }

    JitConstants PoolingKernelBlocked::GetJitConstants(const pooling_params& params, DispatchData runInfo) const
    {
        auto input = params.inputs[0];
        auto output = params.output;
        auto jit = PoolingKernelBase::GetJitConstants(params, runInfo);

        size_t input_line_size = params.poolStride.x * (x_block_size - 1) + params.poolSize.x;

        jit.AddConstant(MakeJitConstant("IC_BLOCK", feature_slice_size));
        jit.AddConstant(MakeJitConstant("X_BLOCK_SIZE", x_block_size));
        jit.AddConstant(MakeJitConstant("INPUT_LINE_SIZE", input_line_size));
        jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));
        jit.AddConstant(MakeJitConstant("X_BLOCKS", CeilDiv(output.X().v, x_block_size)));

        return jit;
    }

    bool PoolingKernelBlocked::Validate(const Params& p, const optional_params& o) const
    {
        if (!PoolingKernelBase::Validate(p, o))
        {
            return false;
        }

        const auto& pp = static_cast<const pooling_params&>(p);

        // TODO Enable handling of not full feature slice in kernel
        if (pp.output.Feature().v % feature_slice_size != 0)
            return false;

        return true;
    }

    KernelsData PoolingKernelBlocked::GetKernelsData(const Params& params, const optional_params& options) const
    {
        const auto& pooling_p = static_cast<const pooling_params&>(params);
        if(pooling_p.output.Batch().v == 1)
            return GetCommonKernelsData(params, options, FORCE_PRIORITY_1);
        else
            return GetCommonKernelsData(params, options, FORCE_PRIORITY_7);
    }
}
