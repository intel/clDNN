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

#pragma once

#include "convolution_kernel_base.h"
 
namespace KernelSelector 
{    
    class ConvolutionKernel_bfyx_3x3_dw_opt : public ConvolutionKernelBase
    {
    public:
        ConvolutionKernel_bfyx_3x3_dw_opt();
        virtual ~ConvolutionKernel_bfyx_3x3_dw_opt() {}

        virtual KernelsData GetKernelsData(const Params& params, const OptionalParams& options) const override;
        virtual KernelsData GetKernelsDataForAutoTune(const Params& params, const OptionalParams& options) const override;
        virtual KernelsData GetTunedKernelsDataByIndex(const Params& params, const OptionalParams& options, int autoTuneIndex) const override;
        virtual ParamsKey GetSupportedKey() const override;
    
    protected:
        bool Validate(const Params&, const OptionalParams&) const override;
        virtual std::vector<WeightsLayout> GetSupportedWeightLayouts()  const override { return{ WeightsLayout::oiyx }; }
        KernelData GetKernelDataInternal(const Params& params, const OptionalParams& options, const std::string exeMode, const size_t tileXDim = 0, const size_t tileYDim = 0) const;
        JitConstants GetJitConstants(const ConvolutionParams& params, DispatchData kd) const override;
        DispatchData SetDefaultInternal(const ConvolutionParams& params, const size_t tileXDim = 0, const size_t tileYDim = 0) const;
        std::vector<std::tuple<size_t, size_t, std::string>> autoTuneOptions = {};
    };
}