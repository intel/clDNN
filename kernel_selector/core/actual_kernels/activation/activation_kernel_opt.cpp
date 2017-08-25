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

#include "activation_kernel_opt.h"
#include "kernel_selector_utils.h" 

namespace KernelSelector {

    ParamsKey ActivationKernelOpt::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableAllInputLayout();
        k.EnableAllOutputLayout();
        k.EnableTensorOffset();
        k.EnableBatching();
        return k;
    }

    ActivationKernelOpt::Parent::DispatchData ActivationKernelOpt::SetDefault(const ActivationParams& params) const
    {
        auto runInfo = Parent::SetDefault(params);

        const auto totalSize = params.inputs[0].LogicalSize();

        std::vector<size_t> global = { totalSize/NUM_COLS_WI };
        std::vector<size_t> local = GetOptimalLocalWorkGroupSizes(global);

        runInfo.gws0 = global[0];
        runInfo.gws1 = 1;
        runInfo.gws2 = 1;

        runInfo.lws0 = local[0];
        runInfo.lws1 = 1;
        runInfo.lws2 = 1;

        runInfo.effiency = FORCE_PRIORITY_6;

        return runInfo;
    }

    bool ActivationKernelOpt::Validate(const Params& p, const OptionalParams& o) const
    {
        if (p.GetType() != KernelType::ACTIVATION ||
            o.GetType() != KernelType::ACTIVATION)
        {
            return false;
        }

        const ActivationParams& params = static_cast<const ActivationParams&>(p);

        const auto totalSize = params.inputs[0].LogicalSize();
        if ((totalSize % NUM_COLS_WI) != 0 ||
            (params.inputs[0].GetFirstElementOffset() % NUM_COLS_WI) != 0 ||
            (params.output.GetFirstElementOffset() % NUM_COLS_WI) != 0)
        {
            return false;
        }

        return true;
    }

    JitConstants ActivationKernelOpt::GetJitConstants(const ActivationParams& params, DispatchData) const
    {
        auto jit = MakeActivationJitConstants(params);

        jit.AddConstant(MakeJitConstant("NUM_COLS_WI", NUM_COLS_WI));

        return jit;
    }

    KernelsData ActivationKernelOpt::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        return GetCommonKernelsData(params, options);
    }
}