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

#pragma once

#include "common_kernel_base.h"
#include "kernel_selector_params.h"

namespace KernelSelector
{
    class MVNKernelBase : public CommonKernelBase
    {
    public:
        using CommonKernelBase::CommonKernelBase;
        virtual ~MVNKernelBase() {}

        struct DispatchData : public CommonDispatchData
        {
            size_t itemsNum;
            size_t leftovers;
            size_t dataSetsCount;
            size_t dataSetSize;

            DispatchData()
                : itemsNum(0)
                , leftovers(0)
                , dataSetsCount(0)
                , dataSetSize(0)
            {}
        };

    protected:
        virtual JitConstants GetJitConstants(const MVNParams& params, DispatchData kd) const;
        virtual DispatchData SetDefault(const MVNParams& params) const;
        virtual std::string GetKernelName(const MVNParams&) const { return kernelName; }
        KernelsData GetCommonKernelsData(const Params& params, const OptionalParams&, float estimated_time) const;
    };
}