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

#include <iostream>
#include "convolution_kernel_bfyx_f16_depthwise.h"
#include "kernel_selector_utils.h"
 
namespace kernel_selector 
{
    static const size_t sub_group_size = 16;
    static const size_t feature_slice_size = 16;

    ParamsKey ConvolutionKernel_bfyx_f16_depthwise::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputWeightsType(WeightsType::F16);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableInputLayout(DataLayout::bfyx_f16);
        k.EnableOutputLayout(DataLayout::bfyx_f16);
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBiasPerFeature();
        k.EnableNonBiasTerm();
        k.EnableBatching();
        k.EnableSplitSupport();
        k.EnableSubGroup();
        k.EnableSubGroupShort();
        k.EnableDepthwiseSeparableOpt();
        return k;
    }

    bool ConvolutionKernel_bfyx_f16_depthwise::Validate(const Params& p, const optional_params&) const
    {
       const convolution_params& cp = static_cast<const convolution_params&>(p);
       if (!cp.depthwise_separable_opt || cp.inputs[0].Feature().v != cp.split || cp.filterSize.x != 3 || cp.filterSize.y != 3 || cp.inputs[0].Batch().v != 1)
           return false;

       if (cp.stride.x != 1 && cp.stride.x != 2)
           return false;

        return true;
    }

    ConvolutionKernelBase::DispatchData ConvolutionKernel_bfyx_f16_depthwise::SetDefault(const convolution_params& params, int) const
    {
        DispatchData runInfo = Parent::SetDefault(params);
        const auto& out = params.output;

        runInfo.gws0 = CeilDiv(out.X().v, 8) * out.Y().v;
        runInfo.gws1 = Align(out.Feature().v, feature_slice_size);
        runInfo.gws2 = out.Batch().v;
        runInfo.lws0 = 1;
        runInfo.lws1 = sub_group_size;
        runInfo.lws2 = 1;

        if (out.Batch().v == 1)
            runInfo.effiency = FORCE_PRIORITY_1;
        else
            runInfo.effiency = FORCE_PRIORITY_7;

        return runInfo;
    }

    JitConstants ConvolutionKernel_bfyx_f16_depthwise::GetJitConstants(const convolution_params& params, const DispatchData& kd) const
    {
        auto mem_consts = ConvolutionKernelBase::GetJitConstants(params, kd);

        mem_consts.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));
        mem_consts.AddConstant(MakeJitConstant("XY_BLOCKS",  CeilDiv(params.output.X().v, 8)));
        mem_consts.AddConstant(MakeJitConstant("IC_BLOCK", feature_slice_size));
        return mem_consts;
    }

    KernelsData ConvolutionKernel_bfyx_f16_depthwise::GetKernelsData(const Params& params, const optional_params& options) const
    {
        return GetCommonKernelsData(params, options);
    }
}
