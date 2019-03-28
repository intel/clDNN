/*
// Copyright (c) 2016-2019 Intel Corporation
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

#include <iostream>
#include "convolution_kernel_bfyx_f16_1x1.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
    static const size_t sub_group_size = 16;
    static const size_t feature_slice_size = 16;
    static const size_t x_block_size = 8;

    ParamsKey ConvolutionKernel_bfyx_f16_1x1::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableInputWeightsType(WeightsType::F16);
        k.EnableInputLayout(DataLayout::bfyx_f16);
        k.EnableOutputLayout(DataLayout::bfyx_f16);
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableDilation();
        k.EnableBiasPerFeature();
        k.EnableNonBiasTerm();
        k.EnableBatching();
        k.EnableSubGroup();
        k.EnableSubGroupShort();
        return k;
    }

    ConvolutionKernelBase::DispatchData ConvolutionKernel_bfyx_f16_1x1::SetDefault(const convolution_params& params, int) const
    {
        DispatchData kd = ConvolutionKernelBase::SetDefault(params);

        const auto& out = params.output;
        auto x = out.X().v;
        auto y = out.Y().v;
        auto f = out.Feature().v;
        auto b = out.Batch().v;

        kd.gws0 = CeilDiv(x * y, x_block_size);
        kd.gws1 = Align(f, feature_slice_size);
        kd.gws2 = b;

        kd.lws0 = 1;
        kd.lws1 = sub_group_size;
        kd.lws2 = 1;

        if (b == 1)
            kd.effiency = FORCE_PRIORITY_1;
        else
            kd.effiency = FORCE_PRIORITY_7;

        return kd;
    }

    bool ConvolutionKernel_bfyx_f16_1x1::Validate(const Params& p, const optional_params& o) const
    {
        if (!ConvolutionKernelBase::Validate(p, o))
        {
            return false;
        }

        const auto& params = static_cast<const convolution_params&>(p);

        const auto &input = params.inputs[0];
        const auto &output = params.output;

        // TODO Add support for output padding to kernel
        const bool bOutputPad = output.X().pad.Total() != 0 || output.Y().pad.Total() != 0;

        const bool bOutputSizes = output.X().v != input.X().v || output.Y().v != input.Y().v || output.Feature().v % 16 != 0;
        const bool bFilterSize = params.filterSize.x != 1 || params.filterSize.y != 1;
        const bool bStride = params.stride.x != 1 || params.stride.y != 1;

        if(bOutputSizes || bFilterSize || bStride || bOutputPad)
        {
            return false;
        }

        return true;
    }

    JitConstants ConvolutionKernel_bfyx_f16_1x1::GetJitConstants(const convolution_params& params, const DispatchData& runInfo) const
    {
        auto jit = Parent::GetJitConstants(params, runInfo);

        jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));
        jit.AddConstant(MakeJitConstant("PADDED_INPUT", params.inputs[0].X().pad.Total() != 0));
        jit.AddConstant(MakeJitConstant("PADDED_OUTPUT", params.output.X().pad.Total() != 0));

        jit.AddConstant(MakeJitConstant("X_BLOCK_SIZE", x_block_size));
        jit.AddConstant(MakeJitConstant("X_BLOCKS", CeilDiv(params.output.X().v, x_block_size)));
        jit.AddConstant(MakeJitConstant("IC_BLOCKS", CeilDiv(params.inputs[0].Feature().v, feature_slice_size)));

        return jit;
    }

    KernelsData ConvolutionKernel_bfyx_f16_1x1::GetKernelsData(const Params& params, const optional_params& options) const
    {
        // TODO Add execution mode to tuning parameters
        return GetCommonKernelsData(params, options, DEFAULT);
    }
}
