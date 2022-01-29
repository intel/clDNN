﻿/*
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

#include "fully_connected_kernel_fb_io_block.h"

namespace kernel_selector 
{
    ParamsKey FullyConnected_fb_io_block::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableInputWeightsType(WeightsType::F16);
        k.EnableInputWeightsType(WeightsType::F32);
        k.EnableAllInputLayout();
        k.EnableOutputLayout(DataLayout::fb);
        k.EnableBatching();
        k.EnableBiasPerFeature();
        k.EnableNonBiasTerm();
        k.EnableSubGroup();
        return k;
    }


    std::unique_ptr<FullyConnected_fb_io_block::FullyConnectedKernelBase::DispatchData> FullyConnected_fb_io_block::SetDefault(const fully_connected_params& arg, int ) const
    {

        auto kd = std::unique_ptr<DispatchData>(new DispatchData(*FullyConnectedKernelBase::SetDefault(arg)));
        const auto& output = arg.output;
        
        auto batch_size = output.Batch().v;
        auto response_size = output.Feature().v;

        constexpr uint32_t unit_byte_size = sizeof(short);
        const char* chunk_type = "uint";
        constexpr uint32_t chunk_byte_size = sizeof(uint32_t);
        constexpr uint32_t sub_group_size = 16;
        constexpr uint32_t units_per_chunk = chunk_byte_size / unit_byte_size;
        constexpr uint32_t units_per_sg_read = sub_group_size * units_per_chunk;

        
        // Number of response groups. Each group (except last) writes units_per_sg_read responses
        // for at least one input data set from batch.
        auto rg_count = CeilDiv(response_size, units_per_sg_read);

        kd->lws0 = sub_group_size;
        // Number of work items needed to process all response groups.
        kd->gws0 = rg_count * sub_group_size;
        kd->lws1 = 1;
        kd->gws1 = batch_size / units_per_sg_read;

        kd->unit_byte_size    = unit_byte_size;
        kd->chunk_type        = chunk_type;
        kd->chunk_byte_size   = chunk_byte_size;
        kd->units_per_chunk   = units_per_chunk;
        kd->bytes_per_sg_read = sub_group_size * chunk_byte_size;
        kd->units_per_sg_read = units_per_sg_read;
        kd->rg_count          = (uint32_t)rg_count;
        kd->last_rg_size      = response_size % units_per_sg_read;
        return kd;
    }

    JitConstants FullyConnected_fb_io_block::GetJitConstants(const fully_connected_params& params, const FullyConnectedKernelBase::DispatchData& run_info) const
    {
        auto &d = static_cast<const DispatchData&>(run_info);
        auto cldnn_jit = FullyConnectedKernelBase::GetJitConstants(params, run_info);
        cldnn_jit.AddConstants({
            MakeJitConstant("SUB_GROUP_SIZE",        d.lws0),
            MakeJitConstant("WORK_ITEMS_PER_BATCH",  d.gws1),
            MakeJitConstant("UNIT_BYTE_SIZE",        d.unit_byte_size),
            MakeJitConstant("CHUNK_TYPE",            d.chunk_type),
            MakeJitConstant("CHUNK_BYTE_SIZE",       d.chunk_byte_size),
            MakeJitConstant("UNITS_PER_CHUNK",       d.units_per_chunk),
            MakeJitConstant("BYTES_PER_SG_READ",     d.bytes_per_sg_read),
            MakeJitConstant("UNITS_PER_SG_READ",     d.units_per_sg_read),
            MakeJitConstant("RG_COUNT",              d.rg_count),
            MakeJitConstant("LAST_RG_SIZE",          d.last_rg_size),
        });
        return cldnn_jit;
    }

    bool FullyConnected_fb_io_block::Validate(const Params& p, const optional_params& o) const
    {
        if (!FullyConnectedKernelBase::Validate(p, o))
        {
            return false;
        }

        const auto& params = static_cast<const fully_connected_params&>(p);

        const auto& output = params.output;
        const auto responseSize = output.Feature().v;
        const auto batches = output.Batch().v;
        const auto xSize = output.LogicalSize() / batches;

        constexpr uint32_t subGroupSize         = 16;
        constexpr uint32_t bytesPerElement      = sizeof(short);
        constexpr uint32_t chunkSizeInBytes     = sizeof(uint32_t);
        constexpr uint32_t chunkSizeInElements  = chunkSizeInBytes / bytesPerElement;
        constexpr uint32_t elementsPerBlockRead = subGroupSize * chunkSizeInElements;

        const bool bSupportedBatch = 
            (batches > 0) && 
            ((batches % 8) == 0) &&
            ((batches % elementsPerBlockRead) == 0);

        const bool bSupportedFeature = 
            (responseSize > 0) && 
            (((responseSize * bytesPerElement) % 4) == 0) &&
            ((xSize % 8) == 0);

        if (!bSupportedBatch ||
            !bSupportedFeature)
        {
            return false;
        }

        return true;
    }

    KernelsData FullyConnected_fb_io_block::GetKernelsData(const Params& params, const optional_params& optParams) const
    {
        assert(params.GetType() == KernelType::FULLY_CONNECTED);

        const auto& orgParams = static_cast<const fully_connected_params&>(params);

        float estimated_time =
            orgParams.inputs[0].GetDType() == Datatype::F16 && orgParams.output.Batch().v >= 16 ?
            FORCE_PRIORITY_3 : FORCE_PRIORITY_5;

        // TODO: it should be fb_io. but the original code use this kernel with yxfb and yxio 
        //       (fb == fyxb flatten fyx, not yxfb flatten yxf).
        //       the order of the add operation cause some numeric changes. in order to avoid them right now we use yxfb/oiyx instead.
        // return GetCommonKernelsData(params, optParams, DataLayout::fb, WeightsLayout::io, estimated_time);
        //return GetCommonKernelsData(params, optParams, DataLayout::yxfb, { WeightsLayout::yxio }, estimated_time);

        KernelsData res = {};
        for (size_t i = 0; i < autoTuneOptions.size(); i++)
        {
            KernelsData kd = GetTunedKernelsDataByIndex(params, optParams, DataLayout::yxfb, { WeightsLayout::yxio }, estimated_time, (int)i);
            if (!kd.empty())
            {
                res.emplace_back(kd[0]);
            }
        }

        return res;
    }
}
