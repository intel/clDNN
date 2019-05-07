/*
// Copyright (c) 2018-2019 Intel Corporation
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
 
namespace kernel_selector 
{    
    class ConvolutionKernel_bfyx_f16_depthwise : public ConvolutionKernelBase
    {
    public:
        using Parent = ConvolutionKernelBase;
        ConvolutionKernel_bfyx_f16_depthwise() : ConvolutionKernelBase("convolution_gpu_bfyx_f16_depthwise") {}
        virtual ~ConvolutionKernel_bfyx_f16_depthwise() {}

        virtual KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
        virtual ParamsKey GetSupportedKey() const override;
    
    protected:
        bool Validate(const Params&, const optional_params&) const override;
        std::vector<WeightsLayout> GetSupportedWeightLayouts(const convolution_params&)  const override {
            return { WeightsLayout::os_iyx_osv16,};
        }
        bool NeedPaddedInput() const override { return false; }
        JitConstants GetJitConstants(const convolution_params& params, const DispatchData& kd) const override;
        DispatchData SetDefault(const convolution_params& params, int autoTuneIndex = -1) const override;
    };
}
