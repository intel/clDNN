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

#include "fully_connected_kernel_bs_f_bsv16_b1.h"
#include "kernel_selector_utils.h"

namespace KernelSelector 
{
    ParamsKey FullyConnected_bs_f_bsv16_b1::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableInputWeightsType(WeightsType::F16);
        k.EnableInputWeightsType(WeightsType::F32);
        k.EnableInputLayout(DataLayout::bfyx);
        k.EnableInputLayout(DataLayout::bf);
        k.EnableOutputLayout(DataLayout::bf);
        k.EnableBiasPerFeature();
        k.EnableNonBiasTerm();
        k.EnableSubGroup();
        return k;
    }

    JitConstants FullyConnected_bs_f_bsv16_b1::GetJitConstants(const FullyConnectedParams& params, const DispatchData& runInfo) const
    {
        auto cldnn_jit = FullyConnectedKernelBase::GetJitConstants(params, runInfo);
        cldnn_jit.AddConstants({
            MakeJitConstant("SUB_GROUP_SIZE",             runInfo.lws0),
            MakeJitConstant("WORK_ITEMS_PER_BATCH",       runInfo.gws1),

            MakeJitConstant("UNIT_BYTE_SIZE",             runInfo.data_bx_bs_x_bsv16.unit_byte_size),
            MakeJitConstant("CHUNK_TYPE",                 runInfo.data_bx_bs_x_bsv16.chunk_type),
            MakeJitConstant("CHUNK_BYTE_SIZE",            runInfo.data_bx_bs_x_bsv16.chunk_byte_size),
            MakeJitConstant("UNITS_PER_CHUNK",            runInfo.data_bx_bs_x_bsv16.units_per_chunk),
            MakeJitConstant("BYTES_PER_SG_READ",          runInfo.data_bx_bs_x_bsv16.bytes_per_sg_read),
            MakeJitConstant("UNITS_PER_SG_READ",          runInfo.data_bx_bs_x_bsv16.units_per_sg_read),
            MakeJitConstant("RESPONSES_PER_SG_EXEC",      runInfo.data_bx_bs_x_bsv16.responses_per_sg_exec),
            MakeJitConstant("IN_CHUNK_PREFETCH_SIZE",     runInfo.data_bx_bs_x_bsv16.in_chunk_prefetch_size),
            MakeJitConstant("FILTER_CHUNK_PREFETCH_SIZE", runInfo.data_bx_bs_x_bsv16.filter_chunk_prefetch_size),
        });
        return cldnn_jit;
    }

    FullyConnected_bs_f_bsv16_b1::DispatchData FullyConnected_bs_f_bsv16_b1::SetDefault(const FullyConnectedParams& arg) const
    {
        DispatchData run_info = FullyConnectedKernelBase::SetDefault(arg);

        // Properties of chunk and unit.
        const     char*    chunk_type           = "uint";
        const     uint32_t unit_byte_size       = run_info.fp16UnitUsed ? sizeof(short) : sizeof(float);
        constexpr uint32_t chunk_byte_size      = sizeof(uint32_t);
        constexpr uint32_t sub_group_size       = 16;
        const     uint32_t units_per_chunk      = chunk_byte_size / unit_byte_size;
        const     uint32_t units_per_sg_read    = sub_group_size * units_per_chunk;
        // Properties of primitive responses.
        constexpr uint32_t responses_per_sg_exec = 16; // Must match batch slice size of weights format (bs_x_bsv16).
                                                       // Number of response groups. Each group (except last) writes responses_per_sg_exec responses
                                                       // for at least one input data set from batch.
        const auto response_size = arg.output.Feature().v;
        auto rg_count = CeilDiv(response_size, responses_per_sg_exec);

        run_info.lws0 = sub_group_size;
        // Number of work items needed to process all response groups.
        run_info.gws0 = rg_count * sub_group_size;
        run_info.lws1 = run_info.lws2 = 1;
        run_info.gws1 = run_info.gws2 = 1;

        auto& kd = run_info.data_bx_bs_x_bsv16;
        kd.unit_byte_size              = unit_byte_size;
        kd.chunk_type                  = chunk_type;
        kd.chunk_byte_size             = chunk_byte_size;
        kd.units_per_chunk             = units_per_chunk;
        kd.bytes_per_sg_read           = sub_group_size * chunk_byte_size;
        kd.units_per_sg_read           = units_per_sg_read;
        kd.responses_per_sg_exec       = responses_per_sg_exec;
        kd.in_chunk_prefetch_size      = 2;
        kd.filter_chunk_prefetch_size  = responses_per_sg_exec;

        return run_info;
    }

    KernelsData FullyConnected_bs_f_bsv16_b1::GetKernelsData(const Params& params, const OptionalParams& optParams) const
    {
        return GetCommonKernelsData(params, optParams, DataLayout::bf, {WeightsLayout::os_i_osv16}, FORCE_PRIORITY_5);
    }
}