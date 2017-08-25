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

#include "pooling_kernel_gpu_average_opt.h"
 
namespace KernelSelector 
{
    ParamsKey PoolingKernelGPUAverageOpt::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableInputLayout(DataLayout::bfyx);
        k.EnableOutputLayout(DataLayout::bfyx);
        k.EnablePoolType(PoolType::AVG);
        k.EnablePoolRemainder(PoolRemainder::FLOOR);
        k.EnablePoolRemainder(PoolRemainder::CEIL);
        k.EnablePoolKernelDividerMode(KernelDividerMode::FIXED);
        return k;
    }

    bool PoolingKernelGPUAverageOpt::Validate(const Params& p, const OptionalParams& o) const
    {
        if (!PoolingKernelBase::Validate(p, o))
        {
            return false;
        }

        const PoolingParams& params = static_cast<const PoolingParams&>(p);

        if (params.activationFunc != ActivationFunction::NONE)
        {
            return{};
        }

        if ((params.poolParams.poolSize.x != 3) ||
            (params.poolParams.poolSize.y != 3) ||
            (params.poolParams.poolStride.x != 1) ||
            (params.poolParams.poolStride.y != 1) ||
            (params.poolParams.poolPad.x != 1) ||
            (params.poolParams.poolPad.y != 1) ||
            !(params.inputs[0] == params.output) ||
            params.inputs[0].PitchesDifferFromLogicalDims() ||
            params.output.PitchesDifferFromLogicalDims())
        {
            return false;
        }

        return true;
    }

    static uSize GetTileDimentions()
    {
        constexpr int simdSize = 16;

        return{ simdSize - 2, 7 };
    }

    PoolingKernelBase::DispatchData PoolingKernelGPUAverageOpt::SetDefault(const PoolingParams& params) const
    {
        constexpr int simdSize = 16;

        DispatchData runInfo = PoolingKernelBase::SetDefault(params);

        auto tileDims = GetTileDimentions();

        const int numTilesX = static_cast<int>(std::ceil(static_cast<float>(params.inputs[0].X().v) / static_cast<float>(tileDims.x)));
        const int numTilesY = static_cast<int>(std::ceil(static_cast<float>(params.inputs[0].Y().v) / static_cast<float>(tileDims.y)));

        runInfo.gws0 = numTilesX * simdSize;
        runInfo.gws1 = numTilesY;
        runInfo.gws2 = params.inputs[0].Feature().v;
        runInfo.lws0 = simdSize;
        runInfo.lws1 = 1;
        runInfo.lws2 = 1;

        return runInfo;
    }

    JitConstants PoolingKernelGPUAverageOpt::GetJitConstants(const PoolingParams& params, DispatchData kd) const
    {
        auto tileDims = GetTileDimentions();
        auto mem_consts = PoolingKernelBase::GetJitConstants(params, kd);

        if (tileDims.y != 0 && tileDims.x != 0)
        {
            mem_consts.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", kd.lws0));
            mem_consts.AddConstant(MakeJitConstant("TILE_HEIGHT", tileDims.y));
            mem_consts.AddConstant(MakeJitConstant("TILE_WIDTH", tileDims.x));
            mem_consts.AddConstant(MakeJitConstant("ONE_OVER_POOL_SIZE", 1.f / (params.poolParams.poolSize.x * params.poolParams.poolSize.y)));
        }

        return mem_consts;
    }

    KernelsData PoolingKernelGPUAverageOpt::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        return GetCommonKernelsData(params, options, FORCE_PRIORITY_7);
    }
}