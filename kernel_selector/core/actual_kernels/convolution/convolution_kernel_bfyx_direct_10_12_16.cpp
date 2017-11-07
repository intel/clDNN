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

#include "convolution_kernel_bfyx_direct_10_12_16.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"

namespace KernelSelector {

    ParamsKey ConvolutionKernel_bfyx_Direct_10_10_12::GetSupportedKey() const
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
        k.EnableSubGroupShort();
        k.EnableBiasPerFeature();
        k.EnableBiasPerOutput();
        k.EnableNonBiasTerm();
        k.EnableBatching();
        k.EnableSplitSupport();
        return k;
    }

    JitConstants ConvolutionKernel_bfyx_Direct_10_10_12::GetJitConstants(const ConvolutionParams& params, Parent::DispatchData runInfo) const
    {
        JitConstants jit = Parent::GetJitConstants(params, runInfo);
        const auto& cp = params.convParams;

        jit.AddConstants({
            MakeJitConstant("ALIGNED_OFM",                  RoundUp(params.output.Feature().v, runInfo.gemmStyle.subBlockDimN)),
            MakeJitConstant("DX",                           runInfo.gemmStyle.globalWorkSizeDX),
            MakeJitConstant("DY",                           runInfo.gemmStyle.globalWorkSizeDY),
            MakeJitConstant("KERNEL_SLICE_DIV2",            (cp.filterSize.x * cp.filterSize.y) / 2),
            MakeJitConstant("RIGHT_PARTIAL_TILE_K",         params.output.X().v % runInfo.gemmStyle.globalWorkSizeDX),
            MakeJitConstant("INPUT_BUFFER_WIDTH_PADDED",    ""),    // TODO: enable non padding path again
            MakeJitConstant("INPUT_BUFFER_HEIGHT_PADDED",   ""),
        });

        return jit;
    }

    ConvolutionKernel_bfyx_Direct_10_10_12::Parent::DispatchData ConvolutionKernel_bfyx_Direct_10_10_12::SetDefault(const ConvolutionParams& arg, int) const
    {
        Parent::DispatchData runInfo = Parent::SetDefault(arg);

        const auto& cp = arg.convParams;
        constexpr uint32_t TILE_N = 16;

        if (cp.filterSize.x == 5)
        {
            runInfo.gemmStyle = { 1, 1, TILE_N, /*GWS DX*/ 4, /*GWS DY*/ 4, 1 };
        }
        else
        {
            runInfo.gemmStyle = { 1, 1, TILE_N, /*GWS DX*/ 4, /*GWS DY*/ 3, 1 };
        }

        runInfo.gws0 = RoundUp(arg.output.X().v, runInfo.gemmStyle.globalWorkSizeDX) / runInfo.gemmStyle.globalWorkSizeDX;
        runInfo.gws1 = RoundUp(arg.output.Y().v, runInfo.gemmStyle.globalWorkSizeDY) / runInfo.gemmStyle.globalWorkSizeDY;
        runInfo.gws2 = RoundUp(arg.output.Feature().v, TILE_N) * arg.output.Batch().v;

        runInfo.lws0 = 1;
        runInfo.lws1 = 1;
        runInfo.lws2 = TILE_N;

        runInfo.effiency = FORCE_PRIORITY_4;

        return runInfo;
    }

    bool ConvolutionKernel_bfyx_Direct_10_10_12::Validate(const Params& p, const OptionalParams& o) const
    {
        if (!Parent::Validate(p, o) || 
            !CovolutionCheckInput(p, o))
        {
            return false;
        }

        const ConvolutionParams& params = static_cast<const ConvolutionParams&>(p);

        const auto& cp = params.convParams;

        const bool bStrideOK = (cp.stride.x == 1 && cp.stride.y == 1);
        const bool bFilter3x3 = (cp.filterSize.x == 3 && cp.filterSize.y == 3);
        const bool bFilter5x5 = (cp.filterSize.x == 5 && cp.filterSize.y == 5);
        const bool bFilterOK = bFilter3x3 || bFilter5x5;

        if (!bFilterOK || !bStrideOK)
        {
            return false;
        }

        return true;
    }

    KernelsData ConvolutionKernel_bfyx_Direct_10_10_12::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        return GetCommonKernelsData(params, options, AGE_BASED);
    }
}