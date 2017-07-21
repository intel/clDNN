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

#include "eltwise_kernel_ref.h"
#include "kernel_selector_utils.h" 

namespace KernelSelector {

    ParamsKey EltwiseKernelRef::GetSupportedKey() const
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

    KernelsData EltwiseKernelRef::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        assert(params.GetType() == KernelType::ELTWISE);

        KernelData kd = KernelData::Default<EltwiseParams>(params);

        EltwiseParams& newParams = *static_cast<EltwiseParams*>(kd.params.get());
        const std::string kernel_id = GetEntryPoint(kernelName, params.layerID, options);

        std::stringstream jit;
        jit << GetBaseJit(newParams, kernel_id)
            << "#define INPUT_OFFSET1 (" << newParams.inputs[1].GetFirstElementOffset() << ")\n"
            << "#define INPUT_ROW_PITCH1 (" << newParams.inputs[1].Y().pitch << ")\n"
            << "#define INPUT_SLICE_PITCH1 (" << newParams.inputs[1].Feature().pitch << ")\n"
            << "#define INPUT_BATCH_PITCH1 (" << newParams.inputs[1].Batch().pitch << ")\n"
            //<< "#define ELTWISE_MODE_" << toString(newParams.eltwiseParams.mode) << "\n"
            //<< "#define SCALAR (" << newParams.eltwiseParams.scalar << ")\n"
            ;

        const auto& out = newParams.output;
        auto& kernel = kd.kernels[0];
        kernel.workGroups.global = { out.X().v, out.Y().v, out.Feature().v*out.Batch().v };
        kernel.workGroups.local = GetOptimalLocalWorkGroupSizes(kernel.workGroups.global);
        kernel.kernelString = GetKernelString(kernelName, jit.str(), kernel_id);
        kernel.arguments = GetArgumentDesc(2, false, false);

        kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}