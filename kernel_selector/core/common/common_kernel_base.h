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

#include "kernel_base.h"
#include "jitter.h"
#include <sstream>
#include <assert.h>

namespace KernelSelector 
{
    struct CommonDispatchData
    {
        size_t gws0, gws1, gws2;
        size_t lws0, lws1, lws2;
        bool fp16UnitUsed;           ///< Value indicating that FP16 half precision floating point type will be used (instead of single precision).
        float effiency;

        CommonDispatchData() :
            gws0(0), gws1(0), gws2(0),
            lws0(0), lws1(0), lws2(0),
            fp16UnitUsed(false),
            effiency(0.0f)
        {};

    };

    class CommonKernelBase : public KernelBase
    {
    public:
        using KernelBase::KernelBase;
        virtual ~CommonKernelBase() {}

    protected:
        virtual bool                    Validate(const Params&, const OptionalParams&) const { return true; }
        std::string                     CreateJit(const std::string& template_name, JitConstants constants, std::string kernel_name) const;
        std::string                     GetEntryPoint(const std::string& templateName, const std::string& layerID, const OptionalParams& options) const;
        Arguments                       GetArgsDesc(uint32_t num_of_input, bool use_weights, bool use_bias) const;
        std::shared_ptr<KernelString>   GetKernelString(std::string kernel_name, std::string jit, std::string entry_point, std::string exe_mode = ROUND_ROBIN) const;
        void                            FillCLKernelData(clKernelData& kernel, const CommonDispatchData& runInfo, std::string kernel_map_name, std::string jit, std::string entry_point, bool weights = false, bool bias = false) const;
    };

    inline bool CheckActivationSupport(ActivationFunction func)
    {
        switch (func)
        {
        case KernelSelector::ActivationFunction::NONE:
        case KernelSelector::ActivationFunction::RELU:
        case KernelSelector::ActivationFunction::RELU_NEGATIVE_SLOPE:
            return true;
        default:
            return false;
        }
    }
}