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

#include "convolution_kernel_yxfb_yxio_b8.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"

namespace KernelSelector 
{

    ParamsKey ConvolutionKernel_yxfb_yxio_b8::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F32);
        k.EnableInputWeightsType(WeightsType::F16);
        k.EnableInputWeightsType(WeightsType::F32);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableInputLayout(DataLayout::yxfb);
        k.EnableOutputLayout(DataLayout::yxfb);
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBiasPerFeature();
        k.EnableNonBiasTerm();
        k.EnableBatching();
        k.EnableSplitSupport();
        k.EnableDilation();
        k.EnableSubGroup();
        return k;
    }

    ConvolutionKernelBase::DispatchData ConvolutionKernel_yxfb_yxio_b8::SetDefault(const ConvolutionParams& arg, int autoTuneIndex) const
    {
        DispatchData runInfo = ConvolutionKernelBase::SetDefault(arg, autoTuneIndex);

        const auto filterOfmNum = arg.weights.OFM().v;
        const auto batchSize = arg.output.Batch().v;

        runInfo.lws0 = batchSize == 8 ? 8 : 16;

        if (((filterOfmNum * batchSize) / 16) % runInfo.lws0)
        {
            runInfo.cldnnStyle.ofmPerWorkItem = 8;
        }
        else
        {
            runInfo.cldnnStyle.ofmPerWorkItem = 16;
        }

        runInfo.gws0 = filterOfmNum * batchSize / (runInfo.cldnnStyle.ofmPerWorkItem * runInfo.cldnnStyle.batchesPerWorkItem);

        runInfo.effiency = FORCE_PRIORITY_9;
        
        return runInfo;
    }

    bool ConvolutionKernel_yxfb_yxio_b8::Validate(const Params& p, const OptionalParams& o) const
    {
        if (!ConvolutionKernelBase::Validate(p, o))
        {
            return false;
        }

        const ConvolutionParams& params = static_cast<const ConvolutionParams&>(p);

        if (!CheckPitchForSplitOnly(params))
        {
            return false;
        }

        const auto filterOfmNum = params.weights.OFM().v;
        const auto batchSize = params.output.Batch().v;

        const bool bInputValidated =
            (filterOfmNum > 0) &&
            (batchSize > 0) &&
            (params.output.Feature().v == filterOfmNum);

        if (!bInputValidated)
        {
            return false;
        }

        const uint32_t lws0 = batchSize == 8 ? 8 : 16;

        if ((filterOfmNum * batchSize) % lws0 != 0 ||
            batchSize > 16 || batchSize == 1)
        {
            return false;
        }

        if (params.output.PitchesDifferFromLogicalDims())
            return false;

        return true;
    }

    KernelsData ConvolutionKernel_yxfb_yxio_b8::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        return GetCommonKernelsData(params, options);
    }
}