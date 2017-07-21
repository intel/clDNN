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

#include "kernel_selector_common.h"
#include "reorder_kernel_base.h"
#include "common_tools.h"

namespace KernelSelector 
{
    inline uint32_t SubGroupSize(WeightsLayout l)
    {
        switch (l)
        {
        case WeightsLayout::os_iyx_osv16:
        case WeightsLayout::os_i_osv16:
        case WeightsLayout::os_i_osv16__ai8:
        case WeightsLayout::i_yxs_os_yxsv2_osv16:
        case WeightsLayout::iy_xs_os_xsv2_osv16__ao32:
            return 16;
        case WeightsLayout::os_i_osv8__ai8:
        case WeightsLayout::iy_xs_os_xsv2_osv8__ao32:
            return 8;
        default:
            return 1;
        }
    }

    inline uint32_t SubGroupSize(DataLayout l)
    {
        switch (l)
        {
        case DataLayout::bs_f_bsv16__af8:
            return 16;
        case DataLayout::bs_f_bsv8__af8:
            return 8;
        default:
            return 1;
        }
    }

    JitConstants ReorderKernelBase::GetJitConstants(const ReorderWeightsParams& params) const
    {
        JitConstants mem_consts = MakeReorderWeightsJitConstants(params);
       
        mem_consts.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", SubGroupSize(params.reorderParams.output.GetLayout())));

        return mem_consts;
    }

    JitConstants ReorderKernelBase::GetJitConstants(const ReorderParams& params) const
    {
        JitConstants mem_consts = MakeReorderJitConstants(params);

        mem_consts.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", SubGroupSize(params.output.GetLayout())));

        return mem_consts;
    }
}