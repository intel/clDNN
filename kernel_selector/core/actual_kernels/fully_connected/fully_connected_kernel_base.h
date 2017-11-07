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
    class FullyConnectedKernelBase : public CommonKernelBase
    {
    public:
        using CommonKernelBase::CommonKernelBase;
        virtual ~FullyConnectedKernelBase() {}

        struct DispatchData : public CommonDispatchData
        {};
    
    protected:
        virtual JitConstants GetJitConstants(const FullyConnectedParams& params, const DispatchData& kd) const;
        virtual std::unique_ptr<DispatchData> SetDefault(const FullyConnectedParams& params) const;
        KernelsData GetCommonKernelsData(const Params& params, const OptionalParams& optParams, DataLayout dl, std::vector<WeightsLayout> wl, float estimated_time = DONT_USE_IF_HAVE_SOMETHING_ELSE) const;

        bool Validate(const Params& p, const OptionalParams&) const override
        {
            if (p.GetType() != KernelType::FULLY_CONNECTED)
            {
                return false;
            }

            return true;
        }
    };
}