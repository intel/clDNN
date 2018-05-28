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

#include "mvn_kernel_base.h"

namespace KernelSelector
{
    class MVNKernelBfyxOpt : public MVNKernelBase
    {
    public:
        MVNKernelBfyxOpt() : MVNKernelBase("mvn_gpu_bfyx_opt") {}
        virtual ~MVNKernelBfyxOpt() {}

        virtual KernelsData GetKernelsData(const Params& params, const OptionalParams& options) const override;
        virtual ParamsKey GetSupportedKey() const override;
        using Parent = MVNKernelBase;

    private:
        DispatchData SetDefault(const MVNParams& params) const override;
        JitConstants GetJitConstants(const MVNParams& params, MVNKernelBase::DispatchData kd) const override;
    };
}
