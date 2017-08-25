/*
// Copyright (c) 2017 Intel Corporation
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

#include "convolution_kernel_bfyx_3x3_dw_opt.h"
#include "kernel_selector_utils.h"
 
namespace KernelSelector 
{
    ConvolutionKernel_bfyx_3x3_dw_opt::ConvolutionKernel_bfyx_3x3_dw_opt() : ConvolutionKernelBase("convolution_gpu_bfyx_3x3_dw_opt")
    {
        // Generate the dispatch options to the auto-tuner.
        std::vector<size_t> tileXDimSizes = { 1,2,4,5,6,8,10,12,14,16 };
        std::vector<size_t> tileYDimSizes = { 1,2,3,4,5,6,7 };
        std::vector<std::string> executionModes = { /*AGE_BASED ,*/ ROUND_ROBIN };

        for (auto tileXDim : tileXDimSizes)
        {
            for (auto tileYDim : tileYDimSizes)
            {
                for (auto executionMode : executionModes)
                {
                    autoTuneOptions.emplace_back(tileXDim, tileYDim, executionMode);
                }
            }
        }
    }

    ParamsKey ConvolutionKernel_bfyx_3x3_dw_opt::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F32);
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputWeightsType(WeightsType::F16);
        k.EnableInputWeightsType(WeightsType::F32);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableInputLayout(DataLayout::bfyx);
        k.EnableOutputLayout(DataLayout::bfyx);
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBiasPerFeature();
        k.EnableNonBiasTerm();
        k.EnableBatching();
        k.EnableSplitSupport();
        k.EnableSubGroup();
        k.EnableSubGroupShort();
        k.EnableDepthwiseSeparableOpt();
        return k;
    }

    bool ConvolutionKernel_bfyx_3x3_dw_opt::Validate(const Params& p, const OptionalParams& o) const
    {
        if (!ConvolutionKernelBase::Validate(p, o) ||
            !CovolutionCheckInput(p, o))
        {
            return false;
        }

        const ConvolutionParams& params = static_cast<const ConvolutionParams&>(p);

        if ((params.convParams.filterSize.x != 3) ||
            (params.convParams.filterSize.y != 3) ||
            (params.convParams.stride.x != 1) ||
            (params.convParams.stride.y != 1) ||
            (params.convParams.padding.x != 1) ||
            (params.convParams.padding.y != 1) ||
            (params.inputs[0].Feature().v != params.convParams.split) ||
            params.output.PitchesDifferFromLogicalDims())
        {
            return false;
        }

        return true;
    }

    static stSize GetTileDimensions()
    {
        constexpr int simdSize = 16;

        return{ simdSize - 2, 7 };
    }

    ConvolutionKernelBase::DispatchData ConvolutionKernel_bfyx_3x3_dw_opt::SetDefaultInternal(const ConvolutionParams& params, const size_t tileXDim, const size_t tileYDim) const
    {
        constexpr int simdSize = 16;

        DispatchData runInfo = ConvolutionKernelBase::SetDefault(params);

        stSize tileDims;

        if (tileXDim == 0 || tileYDim == 0)
        {
            tileDims = GetTileDimensions();
        }
        else // Auto-tuner case
        {
            tileDims.x = tileXDim;
            tileDims.y = tileYDim;
        }

        const int numTilesX = static_cast<int>(std::ceil(static_cast<float>(params.inputs[0].X().v) / static_cast<float>(tileDims.x)));
        const int numTilesY = static_cast<int>(std::ceil(static_cast<float>(params.inputs[0].Y().v) / static_cast<float>(tileDims.y)));

        runInfo.gws0 = numTilesX * simdSize;
        runInfo.gws1 = numTilesY;
        runInfo.gws2 = params.inputs[0].Feature().v * params.inputs[0].Batch().v;
        runInfo.lws0 = simdSize;
        runInfo.lws1 = 1;
        runInfo.lws2 = 1;

        runInfo.effiency = FORCE_PRIORITY_5;

        return runInfo;
    }

    JitConstants ConvolutionKernel_bfyx_3x3_dw_opt::GetJitConstants(const ConvolutionParams& params, DispatchData kd) const
    {
        auto tileDims = GetTileDimensions();
        auto mem_consts = ConvolutionKernelBase::GetJitConstants(params, kd);

        if (tileDims.y != 0 && tileDims.x != 0)
        {
            mem_consts.AddConstant(MakeJitConstant("UNIT_BYTE_SIZE", kd.fp16UnitUsed ? sizeof(short) : sizeof(float)));
            mem_consts.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", kd.lws0));
            mem_consts.AddConstant(MakeJitConstant("TILE_HEIGHT", tileDims.y));
            mem_consts.AddConstant(MakeJitConstant("TILE_WIDTH", tileDims.x));
        }

        return mem_consts;
    }

    KernelsData ConvolutionKernel_bfyx_3x3_dw_opt::GetTunedKernelsDataByIndex(const Params& params, const OptionalParams& options, const int autoTuneIndex) const
    {
        if (!Validate(params, options))
        {
            return{};
        }

        if (autoTuneIndex == -1)
        {
            return GetKernelsData(params, options);
        }

        if ((autoTuneIndex < 0) || (autoTuneIndex >= (int)autoTuneOptions.size()))
        {
            return{};
        }

        std::tuple<size_t, size_t, std::string> dispatchData = autoTuneOptions[autoTuneIndex];

        KernelData kd = GetKernelDataInternal(params, options, std::get<2>(dispatchData), std::get<0>(dispatchData), std::get<1>(dispatchData));

        return{ kd };
    }

    KernelData ConvolutionKernel_bfyx_3x3_dw_opt::GetKernelDataInternal(const Params& params, const OptionalParams& options, const std::string exeMode, const size_t tileXDim, const size_t tileYDim) const
    {
        KernelData kd = KernelData::Default<ConvolutionParams>(params);
        ConvolutionParams& newParams = *static_cast<ConvolutionParams*>(kd.params.get());

        DispatchData runInfo = SetDefaultInternal(newParams, tileXDim, tileYDim);

        kd.reorderInput = CovolutionUpdateInputParams(newParams);

        bool succeed = UpdateWeightsParams(
            newParams,
            options,
            { WeightsLayout::oiyx },
            kd.weightsReorderParams);

        if (!succeed)
        {
            return{};
        }

        auto cldnn_jit = GetJitConstants(newParams, runInfo);
        auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, runInfo, kernelName, jit, entry_point, exeMode, true, !newParams.bias.empty());
        kernel.arguments.push_back({ ArgumentDescriptor::Types::SPLIT, 0 });

        kd.estimatedTime = runInfo.effiency;

        return kd;
    }

    KernelsData ConvolutionKernel_bfyx_3x3_dw_opt::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        if (!Validate(params, options))
        {
            return{};
        }

        KernelData kd = GetKernelDataInternal(params, options, ROUND_ROBIN);

        return{ kd };
    }

    KernelsData ConvolutionKernel_bfyx_3x3_dw_opt::GetKernelsDataForAutoTune(const Params& params, const OptionalParams& options) const
    {
        if (!Validate(params, options))
        {
            return{};
        }

        KernelsData res = {};
        int index = 0;

        for (auto autoTuneOption : autoTuneOptions)
        {
            KernelData kd = GetKernelDataInternal(params, options, std::get<2>(autoTuneOption), std::get<0>(autoTuneOption), std::get<1>(autoTuneOption));
            kd.autoTuneIndex = index;
            res.emplace_back(kd);
            index++;
        }

        KernelsData defaultKds = GetKernelsData(params, options);
        res.insert(res.end(), defaultKds.begin(), defaultKds.end());

        return res;
    }
}