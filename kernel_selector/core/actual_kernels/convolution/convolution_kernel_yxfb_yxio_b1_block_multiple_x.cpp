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

#include "convolution_kernel_yxfb_yxio_b1_block_multiple_x.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"

namespace KernelSelector 
{

    ParamsKey ConvolutionKernel_yxfb_yxio_b1_block_mulitple_x::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F32);
        k.EnableInputWeightsType(WeightsType::F16);
        k.EnableInputWeightsType(WeightsType::F32);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableInputLayout(DataLayout::yxfb);
        k.EnableOutputLayout(DataLayout::yxfb);
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBiasPerFeature();
        k.EnableNonBiasTerm();
        k.EnableBatching();
        k.EnableSplitSupport();
        k.EnableDilation();
        k.EnableSubGroup();
        return k;
    }

    ConvolutionKernelBase::DispatchData ConvolutionKernel_yxfb_yxio_b1_block_mulitple_x::SetDefault(const ConvolutionParams& arg, int autoTuneIndex) const
    {
        DispatchData runInfo = ConvolutionKernelBase::SetDefault(arg, autoTuneIndex);

        const auto filter_ofm_num = arg.weights.OFM().v;
        const auto batch_size = arg.output.Batch().v;

        runInfo.lws0 = 16;

        // We cannot return 8 because we are processing 4 spatial coordinates for batch1,
        // and if we use more than 4 ofm_per_work_item we downgrade simd16 to simd8 which would break this algorithm.
        // NOTE: We could return 8 but then we must process only 2 coordinates, which is slower than processing 4 coordinates using blockread4
        // TODO: experiment with SIMD8 version of algorithm and check if it could be faster
        /*if (output_feature_count % (lws * 8) == 0)
        {
        run_info.ofm_per_work_item = 8;
        run_info.gws1 = static_cast<size_t>(std::ceil(static_cast<float>(run_info.gws1) / 2.0f));
        }
        else*/ if (filter_ofm_num % (runInfo.lws0 * 4) == 0)
        {
            runInfo.cldnnStyle.ofmPerWorkItem = 4;
            // We compute multiple spatial coordinates "x" in a single workitem that's why we must divide
            runInfo.gws1 = static_cast<size_t>(std::ceil(static_cast<float>(runInfo.gws1) / 4.0f));
        }
        else if (filter_ofm_num % (runInfo.lws0 * 2) == 0)
        {
            runInfo.cldnnStyle.ofmPerWorkItem = 2;
            runInfo.gws1 = static_cast<size_t>(std::ceil(static_cast<float>(runInfo.gws1) / 8.0f));
        }
        else
        {
            runInfo.cldnnStyle.ofmPerWorkItem = 1;
            runInfo.gws1 = static_cast<size_t>(std::ceil(static_cast<float>(runInfo.gws1) / 8.0f));
        }

        runInfo.gws0 = filter_ofm_num * batch_size / (runInfo.cldnnStyle.ofmPerWorkItem * runInfo.cldnnStyle.batchesPerWorkItem);
        
        return runInfo;
    }

    JitConstants ConvolutionKernel_yxfb_yxio_b1_block_mulitple_x::GetJitConstants(const ConvolutionParams& params, DispatchData kd) const
    {
        auto cldnn_jit = ConvolutionKernelBase::GetJitConstants(params, kd);

        cldnn_jit.AddConstant(MakeJitConstant("USE_VECTOR", kd.cldnnStyle.ofmPerWorkItem));
        if (kd.cldnnStyle.ofmPerWorkItem == 8)
        {
            cldnn_jit.AddConstant(MakeJitConstant("X_PER_WORK_ITEM", 2));
        }
        else if (kd.cldnnStyle.ofmPerWorkItem == 4)
        {
            cldnn_jit.AddConstant(MakeJitConstant("X_PER_WORK_ITEM", 4));
        }
        else
        {
            cldnn_jit.AddConstant(MakeJitConstant("X_PER_WORK_ITEM", 8));
        }

        return cldnn_jit;
    }

    bool ConvolutionKernel_yxfb_yxio_b1_block_mulitple_x::Validate(const Params& p, const OptionalParams& o) const
    {
        if (!ConvolutionKernelBase::Validate(p, o))
        {
            return false;
        }

        const ConvolutionParams& params = static_cast<const ConvolutionParams&>(p);

        if (!CheckPitchForSplitOnly(params))
        {
            return false;
        }

        const auto filter_ofm_num = params.weights.OFM().v;
        const auto batch_size = params.output.Batch().v;

        const bool bInputValidated =
            (filter_ofm_num > 0) &&
            (batch_size == 1) &&    // current implementation doesn't support batching 
                                    // (subgorup is along batch*ofm and trying to block read filter/bias along batch and filter doesn't contain batching).
            (params.output.Feature().v == filter_ofm_num);

        if (!bInputValidated)
        {
            return false;
        }

        if ((filter_ofm_num * batch_size) % 16 != 0)
        {
            return false;
        }

        return true;
    }

    KernelsData ConvolutionKernel_yxfb_yxio_b1_block_mulitple_x::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        return GetCommonKernelsData(params, options);
    }
}