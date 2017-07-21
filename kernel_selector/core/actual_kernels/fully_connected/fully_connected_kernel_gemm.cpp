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

#include "fully_connected_kernel_gemm.h"
#include "kernel_selector_utils.h"

namespace KernelSelector {

    ParamsKey FullyConnectedKernelGEMM::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableInputWeightsType(WeightsType::F16);
        k.EnableInputWeightsType(WeightsType::F32);
        k.EnableInputLayout(DataLayout::bfyx);
        k.EnableInputLayout(DataLayout::bf);
        k.EnableOutputLayout(DataLayout::bf);
        k.EnableBiasPerFeature();
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBatching();
        return k;
    }

    KernelsData FullyConnectedKernelGEMM::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        assert(params.GetType() == KernelType::FULLY_CONNECTED);

        KernelData kd = KernelData::Default<FullyConnectedParams>(params);

        FullyConnectedParams& newParams = *static_cast<FullyConnectedParams*>(kd.params.get());

        // TODO: handle padding per in x/y (for openvx)
        bool succeed = UpdateWeightsParams(
            newParams,
            options,
            { WeightsLayout::oiyx },
            kd.weightsReorderParams);

        if (!succeed)
        {
            return{};
        }

        std::string entry_point;
        std::stringstream jit;
        if (newParams.inputs[0].GetDType() == Datatype::F16)
        {
            jit << "#define __fc_f16" << "\n";
        }
        else
        {
            jit << "#define __fc_f32" << "\n";
        }

        const uint32_t localWorkSizeX = 64;
        const uint32_t globalWorkSizeX = localWorkSizeX;
        const uint32_t vecSize = 4;
        size_t matrixLineSize = newParams.inputs[0].Batch().pitch;
        const std::string kernel_id = GetEntryPoint(kernelName, params.layerID, options);

        jit << GetBaseJit(newParams, kernel_id)
            << GetFullyConnectedJit(newParams)
            << "#define LAST_INPUT_SIZE_REMAINDER (" << matrixLineSize % (globalWorkSizeX * vecSize) << ")\n"
            << "#define LAST_INPUT_SIZE_DIV_4 (" << matrixLineSize % vecSize << ")\n";
        
        auto& kernel = kd.kernels[0];
        kernel.workGroups.global = { globalWorkSizeX, newParams.output.Feature().v, newParams.output.Batch().v };
        kernel.workGroups.local = { localWorkSizeX, 1, 1 };
        kernel.kernelString = GetKernelString(kernelName, jit.str(), kernel_id);
        kernel.arguments = GetArgumentDesc(1, true, !newParams.bias.empty());

        kd.estimatedTime = FORCE_PRIORITY_6;

        return{ kd };
    }
}
