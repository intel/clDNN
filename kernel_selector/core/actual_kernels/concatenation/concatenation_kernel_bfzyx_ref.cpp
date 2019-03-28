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

#include "concatenation_kernel_bfzyx_ref.h"
#include "kernel_selector_utils.h"

namespace kernel_selector 
{

    ParamsKey ConcatenationKernel_bfzyx_Ref::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableInputDataType(Datatype::INT8);
        k.EnableInputDataType(Datatype::INT32);
        k.EnableInputDataType(Datatype::INT64);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::INT8);
        k.EnableOutputDataType(Datatype::INT32);
        k.EnableOutputDataType(Datatype::INT64);
        k.EnableInputLayout(DataLayout::bfzyx);
        k.EnableOutputLayout(DataLayout::bfzyx);
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBatching();
        k.EnableConcatAxis(ConcatAxis::X);
        k.EnableConcatAxis(ConcatAxis::Y);
        k.EnableConcatAxis(ConcatAxis::Z);
        k.EnableConcatAxis(ConcatAxis::FEATURE);
        k.EnableConcatAxis(ConcatAxis::BATCH);
        k.EnableConcatKernelPerInput();
        return k;
    }

    bool ConcatenationKernel_bfzyx_Ref::Validate(const Params& p, const optional_params& o) const
    {
        if (!ConcatenationKernelBase::Validate(p, o))
        {
            return false;
        }

        const concatenation_params& params = static_cast<const concatenation_params&>(p);

        // all inputs have to have same layout
        auto same_layout = params.inputs[0].GetLayout();
        for (const auto& lt : params.inputs)
        {
            if (lt.GetLayout() != same_layout)
            {
                return false;
            }
        }

        return true;
    }

    ConcatenationKernelBase::DispatchData ConcatenationKernel_bfzyx_Ref::SetDefault(const concatenation_params& params) const
    {
        DispatchData kd;
        const auto& input = params.inputs[0];

        std::vector<size_t> global;
        global = { 
            input.X().v * input.Y().v,
            input.Z().v, 
            input.Feature().v * input.Batch().v };
        auto local = GetOptimalLocalWorkGroupSizes(global);

        kd.gws0 = global[0]; // X * Y
        kd.gws1 = global[1]; // Z
        kd.gws2 = global[2]; // F * B

        kd.lws0 = local[0];
        kd.lws1 = local[1];
        kd.lws2 = local[2];

        kd.effiency = FORCE_PRIORITY_9;

        return kd;
    }

    KernelsData ConcatenationKernel_bfzyx_Ref::GetKernelsData(const Params& params, const optional_params& optParams) const
    {
        KernelsData kd = GetCommonKernelsData(params, optParams);

        if (!kd.empty())
        {
            for (int i = 0; i < (int)kd[0].kernels.size(); i++)
            {
                auto& kernel = kd[0].kernels[i];

                // to avoid cases when we execute with local work sizes 1x1x1
                if (kernel.workGroups.local[0] == 1 &&
                    kernel.workGroups.global[1] != 1)
                {
                    kernel.workGroups.global[1] = Align(kernel.workGroups.global[1], 32);
                    kernel.workGroups.local[1] = 32;
                }
            }
        }

        return kd;
     }
}