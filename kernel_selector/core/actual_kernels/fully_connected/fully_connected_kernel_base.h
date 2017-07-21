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
        {
            bool vload_kernel_type;
            union
            {
                struct
                {
                    uint32_t unit_byte_size;
                    const char* chunk_type;
                    uint32_t chunk_byte_size;
                    uint32_t units_per_chunk;
                    uint32_t bytes_per_sg_read;
                    uint32_t units_per_sg_read;
                    uint32_t rg_count;
                    uint32_t last_rg_size;
                } data_xb_xb_fp16;
                struct
                {
                    uint32_t unit_byte_size;
                    const char* chunk_type;
                    uint32_t chunk_byte_size;
                    uint32_t units_per_chunk;
                    uint32_t bytes_per_sg_read;
                    uint32_t units_per_sg_read;
                    uint32_t responses_per_sg_exec;
                    uint32_t in_chunk_prefetch_size;
                    uint32_t filter_chunk_prefetch_size;
                } data_bx_bs_x_bsv16;
            };
        };
    
    protected:
        virtual JitConstants GetJitConstants(const FullyConnectedParams& params, const DispatchData& kd) const;
        virtual DispatchData SetDefault(const FullyConnectedParams& params) const;
        KernelsData GetCommonKernelsData(const Params& params, const OptionalParams& optParams, DataLayout dl, std::vector<WeightsLayout> wl, float estimated_time = DONT_USE_IF_HAVE_SOMETHING_ELSE) const;

        virtual bool Validate(const Params& p, const OptionalParams&) const override
        {
            if (p.GetType() != KernelType::FULLY_CONNECTED)
            {
                return false;
            }

            return true;
        }

        // how many batches will a single work item compute
        static size_t GetBatchesPerWorkItem(const FullyConnectedParams& params)
        {
            auto batchSize = params.output.Batch().v;
            return std::min(batchSize, static_cast<size_t>(32U));
        }

        static size_t GetLocalGroupsSize(const FullyConnectedParams& params)
        {
            auto batchSize = params.output.Batch().v;
            return std::max(static_cast<size_t>(1U), batchSize / GetBatchesPerWorkItem(params));
        }

        // how many neurons for a single batch will a single work item produce 
        static size_t GetNeuronsPerWorkItem(const FullyConnectedParams& params)
        {
            auto batchSize = params.output.Batch().v;
            auto out_elements_count_per_batch = params.output.LogicalSize() / batchSize;
            if (out_elements_count_per_batch % 16 == 0)
                return 2;
            else
                return 1;
        }
    };
}