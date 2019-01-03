/*
// Copyright (c) 2018 Intel Corporation
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

#include "convolution_kernel_imad_1x1.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"

namespace kernel_selector {

    ParamsKey ConvolutionKernel_imad_1x1::GetSupportedKey() const
    {
        return Parent::GetSupportedKey();
    }

    KernelsData
    ConvolutionKernel_imad_1x1::GetKernelsData(
            const Params&          params,
            const optional_params& options) const
    {
        return Parent::GetCommonKernelsData(params, options);
    }

    JitConstants
    ConvolutionKernel_imad_1x1::GetJitConstants(
            const convolution_params& params,
            const DispatchData&       kd) const
    {
        auto mem_consts = Parent::GetJitConstants(params, kd);

        mem_consts.AddConstants({
            // Block reading optimization is implemented for 3x3 only.
            // For 1x1 it should be disabled.
            MakeJitConstant("NON_BLOCK_LOAD", 1),
        });
        return mem_consts;
    }


    ConvolutionKernelBase::DispatchData ConvolutionKernel_imad_1x1::SetDefault(
        const convolution_params& params,
        int) const
    {
        return Parent::SetDefault(params);
    } // SetDefault

    bool
    ConvolutionKernel_imad_1x1::Validate(
            const Params&          params,
            const optional_params& options) const
    {
        if (!ConvolutionKernel_imad_3x3::Parent::Validate(params, options))
        {
            // Skip validation for parent, because it checks 3x3 size
            return false;
        }

        KernelData kd = KernelData::Default<convolution_params>(params);
        convolution_params& newParams = *static_cast<convolution_params*>(kd.params.get());
        return (newParams.filterSize.x == 1 && newParams.filterSize.y == 1);
    }

    KernelsData
    ConvolutionKernel_imad_1x1::GetCommonKernelsData(
            const Params&          params,
            const optional_params& options,
            const std::string      exeMode,
            int                    autoTuneIndex) const
    {
        return Parent::GetCommonKernelsData(params, options, exeMode, autoTuneIndex);
    } // GetCommonKernelsData
}
