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
        k.EnableTensorPitches();
        k.EnableBatching();
        return k;
    }

    KernelsData ActivationKernelOpt::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        assert(params.GetType() == KernelType::ACTIVATION);

        KernelData kd = KernelData::Default<ActivationParams>(params);

        ActivationParams& newParams = *static_cast<ActivationParams*>(kd.params.get());

        static const int NUM_ROWS_WI = 1;
        static const int NUM_COLS_WI = 4;
        const size_t nonWidthDim = newParams.inputs[0].LogicalSize() / newParams.inputs[0].X().v;

        const std::string kernel_id = GetEntryPoint(kernelName, params.layerID, options);

        std::stringstream jit;
        jit << GetBaseJit(newParams, kernel_id);

        jit << "#define NUM_ROWS_WI (" << NUM_ROWS_WI << ")\n"
            << "#define NUM_COLS_WI (" << NUM_COLS_WI << ")\n"
            << "#define INPUT_ROWS (" << nonWidthDim << ")\n"
            << "#define INPUT_ROWS_MOD_ROWS_WI " << nonWidthDim % NUM_ROWS_WI << "\n"
            << "#define INPUT_WIDTH_MOD_COLS_WI " << newParams.inputs[0].X().v % NUM_COLS_WI << "\n";

        auto& kernel = kd.kernels[0];
        kernel.workGroups.global = {
            (newParams.inputs[0].X().v + NUM_COLS_WI - 1) / NUM_COLS_WI,
            (nonWidthDim + NUM_ROWS_WI - 1) / NUM_ROWS_WI,
            newParams.output.Batch().v };
        kernel.workGroups.local = GetOptimalLocalWorkGroupSizes(kernel.workGroups.global);
        kernel.kernelString = GetKernelString(kernelName, jit.str(), kernel_id);
        kernel.arguments = GetArgumentDesc(1, false, false);

        kd.estimatedTime = FORCE_PRIORITY_6;

        return{ kd };
    }
}