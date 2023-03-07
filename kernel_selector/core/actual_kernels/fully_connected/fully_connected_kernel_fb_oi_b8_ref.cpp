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

#include "fully_connected_kernel_fb_oi_b8_ref.h"

namespace kernel_selector 
{
    ParamsKey FullyConnected_fb_oi_b8_ref::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F32);
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

    std::unique_ptr<FullyConnected_fb_oi_b8_ref::DispatchData> FullyConnected_fb_oi_b8_ref::SetDefault(const fully_connected_params& arg, int ) const
    {
        auto kd = FullyConnectedKernelBase::SetDefault(arg);

        const auto& output = arg.output;
        kd->gws0 = output.Batch().v;
        kd->gws1 = output.LogicalSize() / kd->gws0;
        kd->lws0 = 8;
        kd->lws1 = 1;

        return kd;
    }

    bool FullyConnected_fb_oi_b8_ref::Validate(const Params& p, const optional_params& o) const
    {
        if (!FullyConnectedKernelBase::Validate(p, o))
        {
            return false;
        }

        const auto& params = static_cast<const fully_connected_params&>(p);

        if (params.inputs[0].Batch().v != 8)
        {
            return false;
        }

        return true;
    }

    KernelsData FullyConnected_fb_oi_b8_ref::GetKernelsData(const Params& params, const optional_params& optParams) const
    {
        KernelsData res = {};
        for (size_t i = 0; i < autoTuneOptions.size(); i++)
        {
            KernelsData kd = GetTunedKernelsDataByIndex(params, optParams, DataLayout::fb, { WeightsLayout::oi }, FORCE_PRIORITY_6, (int)i);
            if (!kd.empty())
            {
                res.emplace_back(kd[0]);
            }
        }
        return res;
    }
}
