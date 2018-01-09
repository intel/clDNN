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

#include "kernel_selector.h"
 
namespace KernelSelector 
{
    class RegionYoloKernelSelctor : public KernelSelctorBase
    {
    public:
        static RegionYoloKernelSelctor &Instance() {
            static RegionYoloKernelSelctor instance_;
            return instance_;
        }

        RegionYoloKernelSelctor();

        virtual ~RegionYoloKernelSelctor() {}

        virtual KernelsData GetBestKernels(const Params& params, const OptionalParams& options) const override;
    };
}
