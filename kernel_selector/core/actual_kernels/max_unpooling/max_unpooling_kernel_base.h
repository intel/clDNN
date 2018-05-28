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
    class MaxUnpoolingKernelBase : public CommonKernelBase
    {
    public:
        using CommonKernelBase::CommonKernelBase;
        virtual ~MaxUnpoolingKernelBase() {}

        struct DispatchData : public CommonDispatchData
        {
            bool needsBoundary = false;
        };

    protected:
        virtual bool Validate(const Params&, const OptionalParams&) const override;
        virtual JitConstants GetJitConstants(const MaxUnpoolingParams& params) const;
        virtual DispatchData SetDefault(const MaxUnpoolingParams& params) const;
        KernelsData GetCommonKernelsData(const Params& params, const OptionalParams&, float estimatedTime) const;
    };
}