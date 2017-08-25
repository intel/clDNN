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

#include "softmax_kernel_ref.h"
#include "kernel_selector_utils.h" 
 
namespace KernelSelector 
{
    ParamsKey SoftmaxKernelRef::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableInputLayout(DataLayout::bfyx);
        k.EnableInputLayout(DataLayout::yxfb);
        k.EnableInputLayout(DataLayout::bf);
        k.EnableInputLayout(DataLayout::fb);
        k.EnableOutputLayout(DataLayout::bfyx);
        k.EnableOutputLayout(DataLayout::yxfb);
        k.EnableOutputLayout(DataLayout::bf);
        k.EnableOutputLayout(DataLayout::fb);
        k.EnableSoftmaxDim(SoftmaxDim::X);
        k.EnableSoftmaxDim(SoftmaxDim::Y);
        k.EnableSoftmaxDim(SoftmaxDim::FEATURE);
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBatching();
        return k;
    }

    SoftmaxKernelRef::Parent::DispatchData SoftmaxKernelRef::SetDefault(const SoftmaxParams& params, const OptionalParams& optParams) const
    {
        auto runInfo = Parent::SetDefault(params, optParams);

        const auto& out = params.output;
        std::vector<size_t> global;
        switch (params.smParams.dim)
        {
        case SoftmaxDim::X:
            global = { out.Y().v, out.Feature().v, out.Batch().v };
            break;
        case SoftmaxDim::Y:
            global = { out.X().v, out.Feature().v, out.Batch().v };
            break;
        case SoftmaxDim::FEATURE:
            global = { out.X().v, out.Y().v, out.Batch().v };
            break;
        default:
            break;
        }

        auto local = GetOptimalLocalWorkGroupSizes(global);

        runInfo.gws0 = global[0];
        runInfo.gws1 = global[1];
        runInfo.gws2 = global[2];

        runInfo.lws0 = local[0];
        runInfo.lws1 = local[1];
        runInfo.lws2 = local[2];

        return runInfo;
    }

    JitConstants SoftmaxKernelRef::GetJitConstants(const SoftmaxParams& params, DispatchData kd) const
    {
        auto jit = Parent::GetJitConstants(params, kd);

        switch (params.smParams.dim)
        {
        case SoftmaxDim::X:
            jit.AddConstants({
                MakeJitConstant("INPUT0_OTHER0_PITCH",  "INPUT0_Y_PITCH"),
                MakeJitConstant("INPUT0_OTHER1_PITCH",  "INPUT0_FEATURE_PITCH"),
                MakeJitConstant("INPUT0_CLASS_PITCH",   "INPUT0_X_PITCH"),
                MakeJitConstant("INPUT0_CLASS_NUM",     "INPUT0_SIZE_X"),
                MakeJitConstant("OUTPUT_OTHER0_PITCH",  "OUTPUT_Y_PITCH"),
                MakeJitConstant("OUTPUT_OTHER1_PITCH",  "OUTPUT_FEATURE_PITCH"),
                MakeJitConstant("OUTPUT_CLASS_PITCH",   "OUTPUT_X_PITCH"),
            });
            break;
        case SoftmaxDim::Y:
            jit.AddConstants({
                MakeJitConstant("INPUT0_OTHER0_PITCH",  "INPUT0_X_PITCH"),
                MakeJitConstant("INPUT0_OTHER1_PITCH",  "INPUT0_FEATURE_PITCH"),
                MakeJitConstant("INPUT0_CLASS_PITCH",   "INPUT0_Y_PITCH"),
                MakeJitConstant("INPUT0_CLASS_NUM",     "INPUT0_SIZE_Y"),
                MakeJitConstant("OUTPUT_OTHER0_PITCH",  "OUTPUT_X_PITCH"),
                MakeJitConstant("OUTPUT_OTHER1_PITCH",  "OUTPUT_FEATURE_PITCH"),
                MakeJitConstant("OUTPUT_CLASS_PITCH",   "OUTPUT_Y_PITCH"),
            });
            break;
        case SoftmaxDim::FEATURE:
            jit.AddConstants({
                MakeJitConstant("INPUT0_OTHER0_PITCH",  "INPUT0_X_PITCH"),
                MakeJitConstant("INPUT0_OTHER1_PITCH",  "INPUT0_Y_PITCH"),
                MakeJitConstant("INPUT0_CLASS_PITCH",   "INPUT0_FEATURE_PITCH"),
                MakeJitConstant("INPUT0_CLASS_NUM",     "INPUT0_FEATURE_NUM"),
                MakeJitConstant("OUTPUT_OTHER0_PITCH",  "OUTPUT_X_PITCH"),
                MakeJitConstant("OUTPUT_OTHER1_PITCH",  "OUTPUT_Y_PITCH"),
                MakeJitConstant("OUTPUT_CLASS_PITCH",   "OUTPUT_FEATURE_PITCH"),
            });
            break;
        default:
            break;
        }

        // TODO: W/A - currently using low precision accumulator type. (for testing only)
        if (params.output.GetDType() == Datatype::F16)
        {
            jit.AddConstant(MakeJitConstant("ACCUMULATOR_TYPE", "half"));
        }

        return jit;
    }

    KernelsData SoftmaxKernelRef::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        return GetCommonKernelsData(params, options, DONT_USE_IF_HAVE_SOMETHING_ELSE);
    }
}