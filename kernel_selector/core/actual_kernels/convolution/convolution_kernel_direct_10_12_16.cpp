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

#include "convolution_kernel_direct_10_12_16.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"
#include <map>

namespace KernelSelector {

    ParamsKey ConvolutionKernelDirect_10_10_12::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableInputWeightsType(WeightsType::F16);
        k.EnableInputWeightsType(WeightsType::F32);
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

    KernelsData ConvolutionKernelDirect_10_10_12::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        assert(params.GetType() == KernelType::CONVOLUTION && options.GetType() == KernelType::CONVOLUTION);

        const ConvolutionParams& orgParams = static_cast<const ConvolutionParams&>(params);
        const ConvolutionOptionalParams& optParams = static_cast<const ConvolutionOptionalParams&>(options);

        const auto& cp = orgParams.convParams;

        const DataTensor newInput = GetConvolutionBFYXPaddedTensor(orgParams);
        const bool bProperInputDesc = CheckConvolutionPaddedInputDesc(orgParams, newInput);
        const bool bInputPadded = optParams.allowPadding || bProperInputDesc;
        const bool bStrideOK = (cp.stride.x == 1 && cp.stride.y == 1);
        const bool bFilter3x3 = (cp.filterSize.x == 3 && cp.filterSize.y == 3);
        const bool bFilter5x5 = (cp.filterSize.x == 5 && cp.filterSize.y == 5);
        const bool bFilterOK = bFilter3x3 || bFilter5x5;

        if (!bInputPadded || !bFilterOK || !bStrideOK)
        {
            return KernelsData();
        }

        std::stringstream jit;
        KernelData kd;

        auto params_ptr = std::make_shared<ConvolutionParams>(orgParams);
        kd.params = params_ptr;

        ConvolutionParams& newParams = *params_ptr.get();
        kd.kernels.resize(1);

        SubGroupInfo runInfo;

        // for KW only
        kd.reorderInput = false;

        if (!bProperInputDesc)
        {
            newParams.inputs[0] = newInput;
            kd.reorderInput = true;
        }

        bool succeed = UpdateWeightsParams(
            newParams,
            options,
            { WeightsLayout::i_yxs_os_yxsv2_osv16 },
            kd.weightsReorderParams);

        if (!succeed)
        {
            return{};
        }

        jit << "#define INPUT_BUFFER_WIDTH_PADDED" << "\n"
            << "#define INPUT_BUFFER_HEIGHT_PADDED" << "\n";

        constexpr uint32_t TILE_N = 16;

        if (bFilter5x5)
        {
            runInfo = SubGroupInfo(1, 1, TILE_N, 1, 1, TILE_N, /*GWS DX*/ 4, /*GWS DY*/ 4, 1);
        }
        else if (bFilter3x3)
        {
            runInfo = SubGroupInfo(1, 1, TILE_N, 1, 1, TILE_N, /*GWS DX*/ 4, /*GWS DY*/ 3, 1);
        }
        const std::string kernel_id = GetEntryPoint(kernelName, params.layerID, options);

        jit << "#define RIGHT_PARTIAL_TILE_K " << orgParams.output.X().v % runInfo.globalWorkSizeDX << "\n"
            << GetBaseJit(newParams, kernel_id)
            << GetConvolutionJit(newParams, runInfo);

        auto& kernel = kd.kernels[0];
        kernel.workGroups.global = {
            RoundUp(orgParams.output.X().v, runInfo.globalWorkSizeDX) / runInfo.globalWorkSizeDX,
            RoundUp(orgParams.output.Y().v, runInfo.globalWorkSizeDY) / runInfo.globalWorkSizeDY,
            RoundUp(orgParams.output.Feature().v, TILE_N) * orgParams.output.Batch().v };

        kernel.workGroups.local = {
            runInfo.localWorkSizeX,
            runInfo.localWorkSizeY,
            runInfo.localWorkSizeZ };

        kernel.kernelString = GetKernelString(kernelName, jit.str(), kernel_id, AGE_BASED);
        kernel.arguments = GetArgumentDesc(1, true, !newParams.bias.empty());
        kernel.arguments.push_back({ ArgumentDescriptor::Types::SPLIT, 0 });

        kd.estimatedTime = FORCE_PRIORITY_4;

        return{ kd };
    }
}