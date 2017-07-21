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

#include "locally_connected_kernel_ref.h"
#include "kernel_selector_utils.h" 

namespace KernelSelector {

    ParamsKey LocallyConnectedKernelRef::GetSupportedKey() const
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
        return k;
    }

    KernelsData LocallyConnectedKernelRef::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        assert(params.GetType() == KernelType::LOCALLY_CONNECTED);

        KernelData kd = KernelData::Default<LocallyConnectedParams>(params);

        LocallyConnectedParams& newParams = *static_cast<LocallyConnectedParams*>(kd.params.get());

        const std::string kernel_id = GetEntryPoint(kernelName, params.layerID, options);

        std::stringstream jit;
        jit << GetBaseJit(newParams, kernel_id)
            << "#define KERNEL_WIDTH " << newParams.lcParams.filterSize.x << "\n"
            << "#define KERNEL_HEIGHT (" << newParams.lcParams.filterSize.y << ")\n"
            << "#define STRIDE_X (" << newParams.lcParams.stride.x << ")\n"
            << "#define STRIDE_Y (" << newParams.lcParams.stride.y << ")\n"
            << "#define INPUT_PADDING_X (" << newParams.lcParams.padding.x << ")\n"
            << "#define INPUT_PADDING_Y (" << newParams.lcParams.padding.y << ")\n";

        const auto& out = newParams.output;
        auto& kernel = kd.kernels[0];
        kernel.workGroups.global = { out.X().v, out.Y().v, out.Feature().v*out.Batch().v };
        kernel.workGroups.local = GetOptimalLocalWorkGroupSizes(kernel.workGroups.global);
        kernel.kernelString = GetKernelString(kernelName, jit.str(), kernel_id);
        kernel.arguments = GetArgumentDesc(1, true, true);

        kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}