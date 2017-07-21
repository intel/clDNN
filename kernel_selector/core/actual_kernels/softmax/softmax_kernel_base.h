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

#pragma once

#include "common_kernel_base.h"
#include "kernel_selector_params.h"

namespace KernelSelector 
{
    class SoftmaxKernelBase : public CommonKernelBase
    {
    public:
        using CommonKernelBase::CommonKernelBase;
        virtual ~SoftmaxKernelBase() {}

        struct DispatchData : public CommonDispatchData
        {
            size_t itemsNum;
            size_t leftovers;
            size_t dataSetsCount;
            size_t dataSetSize;
            size_t normIndex; //which dimension (from in-memory representation) is normalized, e.g. for bfyx and softmax::normalize_f, it will be f's index == 2 (used only by naive kernel)
        };

    protected:
        JitConstants GetJitConstants(const SoftmaxParams& params, DispatchData kd) const;
        virtual DispatchData SetDefault(const SoftmaxParams& params, const OptionalParams& optParams) const;
        KernelsData GetCommonKernelsData(const Params& params, const OptionalParams& optParams, float estimated_time = DONT_USE_IF_HAVE_SOMETHING_ELSE) const;
    };
}