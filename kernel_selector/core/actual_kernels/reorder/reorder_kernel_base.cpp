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

#include "kernel_selector_common.h"
#include "reorder_kernel_base.h"
#include "common_tools.h"
#include "kernel_selector_utils.h" 

namespace KernelSelector 
{
    inline uint32_t SubGroupSize(WeightsLayout l)
    {
        switch (l)
        {
        case WeightsLayout::os_iyx_osv16:
        case WeightsLayout::os_iyx_osv16_rotate_180:
        case WeightsLayout::os_i_osv16:
        case WeightsLayout::os_i_osv16__ai8:
        case WeightsLayout::i_yxs_os_yxsv2_osv16:
        case WeightsLayout::iy_xs_os_xsv2_osv16__ao32:
            return 16;
        case WeightsLayout::os_i_osv8__ai8:
        case WeightsLayout::iy_xs_os_xsv2_osv8__ao32:
            return 8;
        default:
            return 1;
        }
    }

    inline uint32_t SubGroupSize(DataLayout l)
    {
        switch (l)
        {
        case DataLayout::bs_f_bsv16__af8:
            return 16;
        case DataLayout::bs_f_bsv8__af8:
            return 8;
        default:
            return 1;
        }
    }

    JitConstants ReorderKernelBase::GetJitConstants(const ReorderWeightsParams& params) const
    {
        JitConstants mem_consts = MakeReorderWeightsJitConstants(params);
       
        mem_consts.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", SubGroupSize(params.reorderParams.output.GetLayout())));

        return mem_consts;
    }

    JitConstants ReorderKernelBase::GetJitConstants(const ReorderParams& params) const
    {
        JitConstants mem_consts = MakeReorderJitConstants(params);

        mem_consts.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", SubGroupSize(params.output.GetLayout())));

        return mem_consts;
    }

    ReorderKernelBase::DispatchData ReorderKernelBase::SetDefault(const ReorderWeightsParams& params) const
    {
        const auto& out = params.reorderParams.output;

        DispatchData kd;

        std::vector<size_t> global(3);

        global = { out.OFM().v, out.IFM().v, out.X().v*out.Y().v };
        auto local = GetOptimalLocalWorkGroupSizes(global);

        kd.gws0 = global[0];
        kd.gws1 = global[1];
        kd.gws2 = global[2];

        kd.lws0 = local[0];
        kd.lws1 = local[1];
        kd.lws2 = local[2];

        return kd;
    }

    ReorderKernelBase::DispatchData ReorderKernelBase::SetDefault(const ReorderParams& params) const
    {
        DispatchData kd;

        auto global = GetTensorFriendlyWorkGroups(params.inputs[0]);
        auto local = GetOptimalLocalWorkGroupSizes(global);

        kd.gws0 = global[0];
        kd.gws1 = global[1];
        kd.gws2 = global[2];

        kd.lws0 = local[0];
        kd.lws1 = local[1];
        kd.lws2 = local[2];

        return kd;
    }

    KernelsData ReorderKernelBase::GetCommonKernelsData(const ReorderWeightsParams& params, const OptionalParams& options, float estimated_time) const
    {
        assert(params.GetType() == KernelType::REORDER);

        KernelData kd = KernelData::Default<ReorderWeightsParams>(params);
        ReorderWeightsParams& newParams = *static_cast<ReorderWeightsParams*>(kd.params.get());

        DispatchData runInfo;

        runInfo = SetDefault(newParams);

        auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
        auto cldnn_jit = GetJitConstants(newParams);
        std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[0];
        
        FillCLKernelData(kernel, runInfo, kernelName, jit, entry_point);

        kernel.arguments = GetArgsDesc(1, false, false);

        kd.estimatedTime = estimated_time;

        return{ kd };
    }

    KernelsData ReorderKernelBase::GetCommonKernelsData(const ReorderParams& params, const OptionalParams& options, float estimated_time) const
    {
        if (!Validate(params, options))
        {
            return{};
        }
        assert(params.GetType() == KernelType::REORDER);

        KernelData kd = KernelData::Default<ReorderParams>(params);
        ReorderParams& newParams = *static_cast<ReorderParams*>(kd.params.get());

        DispatchData runInfo;

        runInfo = SetDefault(newParams);

        auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
        auto cldnn_jit = GetJitConstants(newParams);
        std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[0];

        FillCLKernelData(kernel, runInfo, kernelName, jit, entry_point);

        kernel.arguments = GetArgsDesc(1, false, false);
        if (newParams.reorderParams.mode == MeanSubtractMode::IN_BUFFER)
        {
            kernel.arguments.push_back({ ArgumentDescriptor::Types::BIAS, 0 });
        }

        kd.estimatedTime = estimated_time;

        return{ kd };
    }
}