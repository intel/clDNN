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

#include <cmath>
#include "convolution_kernel_gemm_like.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"

namespace KernelSelector 
{
    
    ParamsKey ConvolutionKernelGEMMLike::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableInputWeightsType(WeightsType::F16);
        k.EnableInputWeightsType(WeightsType::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableInputLayout(DataLayout::bfyx);
        k.EnableOutputLayout(DataLayout::bfyx);
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableSubGroup();
        k.EnableBiasPerFeature();
        k.EnableBiasPerOutput();
        k.EnableNonBiasTerm();
        k.EnableBatching();
        k.EnableSplitSupport();
        return k;
    }

    KernelsData ConvolutionKernelGEMMLike::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        assert(params.GetType() == KernelType::CONVOLUTION && options.GetType() == KernelType::CONVOLUTION);

        const ConvolutionParams& orgParams = static_cast<const ConvolutionParams&>(params);
        const ConvolutionOptionalParams& optParams = static_cast<const ConvolutionOptionalParams&>(options);

        const DataTensor newInput = GetConvolutionBFYXPaddedTensor(orgParams);
        const bool bProperInputDesc = CheckConvolutionPaddedInputDesc(orgParams, newInput);
        // TODO: enable non padding path again
        const bool bInputPadded = optParams.allowPadding || bProperInputDesc;

        if (!bInputPadded)
        {
            return KernelsData();
        }

        std::stringstream jit;
        KernelData kd;
        std::string entry_point;

        auto params_ptr = std::make_shared<ConvolutionParams>(orgParams);
        kd.params = params_ptr;

        ConvolutionParams& newParams = *params_ptr.get();
        const auto& cp = newParams.convParams;
        
        kd.kernels.resize(1);

        SubGroupInfo runInfo;
        
        // for KW only
        kd.reorderInput = false;

        if (optParams.allowPadding || bProperInputDesc)
        {
            jit << "#define INPUT_BUFFER_WIDTH_PADDED" << "\n"
                << "#define INPUT_BUFFER_HEIGHT_PADDED" << "\n";

            if (!bProperInputDesc)
            {
                newParams.inputs[0] = newInput;
                kd.reorderInput = true;
            }
        }
        else
        {
            if (cp.padding.x == 0)
            {
                jit << "#define INPUT_BUFFER_WIDTH_PADDED" << "\n";
            }

            if (cp.padding.y == 0)
            {
                jit << "#define INPUT_BUFFER_HEIGHT_PADDED" << "\n";
            }
        }

        WeightsLayout wLayout;

        std::string newKernelName = kernelName;
        
        if (newParams.inputs[0].GetDType() == Datatype::F16)
        {
            newKernelName += "_fp16";
            runInfo = SubGroupInfo(1, cp.filterSize.x, 32, 1, 16, 1, 32, 1, 1);
            wLayout = WeightsLayout::iy_xs_os_xsv2_osv16__ao32;
            kd.estimatedTime = FORCE_PRIORITY_6;
        }
        else
        {
            newKernelName += "_fp32";
            runInfo = SubGroupInfo(2, cp.filterSize.x, 32, 1, 8, 1, 32, 2, 1);
            wLayout = WeightsLayout::iy_xs_os_xsv2_osv8__ao32;
            kd.estimatedTime = FORCE_PRIORITY_8;
        }

        bool succeed = UpdateWeightsParams(
            newParams,
            options,
            { wLayout },
            kd.weightsReorderParams);

        if (!succeed)
        {
            return{};
        }

        const std::string kernel_id = GetEntryPoint(kernelName, params.layerID, options);

        jit << GetBaseJit(newParams, kernel_id)
            << GetConvolutionJit(newParams, runInfo);

        size_t sgemm_m = RoundUp(newParams.output.X().v * newParams.output.Y().v, (size_t)runInfo.subBlockDimM);
        size_t sgemm_n = RoundUp(newParams.output.Feature().v, (size_t)runInfo.subBlockDimN);

        auto& kernel = kd.kernels[0];
        kernel.workGroups.global = {
            RoundUp(int(std::ceil((float)sgemm_n / (float)runInfo.globalWorkSizeDX)), runInfo.localWorkSizeX),
            RoundUp(int(std::ceil((float)sgemm_m / (float)runInfo.globalWorkSizeDY)), runInfo.localWorkSizeY),
            newParams.output.Batch().v };
        
        kernel.workGroups.local = {
            runInfo.localWorkSizeX,
            runInfo.localWorkSizeY,
            runInfo.localWorkSizeZ };

        kernel.kernelString = GetKernelString(newKernelName, jit.str(), kernel_id, AGE_BASED);
        kernel.arguments = GetArgumentDesc(1, true, !newParams.bias.empty());
        kernel.arguments.push_back({ ArgumentDescriptor::Types::SPLIT, 0 });

        return{ kd };
    }
}