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

#include "cnn_kernel_base.h"
#include "kernel_selector_params.h"

namespace KernelSelector {

    struct SubGroupInfo
    {
        uint32_t subBlockDimM = 1;
        uint32_t subBlockDimK = 1;
        uint32_t subBlockDimN = 1;
        uint32_t localWorkSizeX = 0;
        uint32_t localWorkSizeY = 0;
        uint32_t localWorkSizeZ = 0;
        uint32_t globalWorkSizeDX = 1;
        uint32_t globalWorkSizeDY = 1;
        uint32_t globalWorkSizeDZ = 1;

        SubGroupInfo() = default;

        SubGroupInfo(
            uint32_t sBlockDimM, uint32_t sBlockDimK, uint32_t sBlockDimN,
            uint32_t lWorkSzX, uint32_t lWorkSzY, uint32_t lWorkSzZ,
            uint32_t gWorkDX, uint32_t gWorkDY, uint32_t gWorkDZ) :
            subBlockDimM(sBlockDimM),
            subBlockDimK(sBlockDimK),
            subBlockDimN(sBlockDimN),
            localWorkSizeX(lWorkSzX),
            localWorkSizeY(lWorkSzY),
            localWorkSizeZ(lWorkSzZ),
            globalWorkSizeDX(gWorkDX),
            globalWorkSizeDY(gWorkDY),
            globalWorkSizeDZ(gWorkDZ)
        {}
    };

    class CNNConvolutionKernelBase : public CNNKernelBase
    {
    public:
        using CNNKernelBase::CNNKernelBase;
        virtual ~CNNConvolutionKernelBase() {}
    
    protected:
        std::string GetConvolutionJit(const ConvolutionParams& params, SubGroupInfo& runInfo) const;
    };
}