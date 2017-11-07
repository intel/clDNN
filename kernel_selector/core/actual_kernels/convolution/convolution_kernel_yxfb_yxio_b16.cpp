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

#include "convolution_kernel_yxfb_yxio_b16.h"

namespace KernelSelector 
{

    ParamsKey ConvolutionKernel_yxfb_yxio_b16::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableInputWeightsType(WeightsType::F16);
        k.EnableInputWeightsType(WeightsType::F32);
        k.EnableOutputDataType(Datatype::F16);
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

    std::string ConvolutionKernel_yxfb_yxio_b16::GetKernelName(const ConvolutionParams& params) const
    {
        if (params.inputs[0].GetDType() == Datatype::F32)
        {
            return kernelName + "_fp32";
        }
        else
        {
            return kernelName + "_fp16";
        }
    }

    ConvolutionKernelBase::DispatchData ConvolutionKernel_yxfb_yxio_b16::SetDefault(const ConvolutionParams& arg, int) const
    {
        DispatchData runInfo = ConvolutionKernelBase::SetDefault(arg);

        const auto filter_ofm_num = arg.weights.OFM().v;
        const auto batch_size = arg.output.Batch().v;
        const uint32_t min_lws = 16;

        if (arg.inputs[0].GetDType() == Datatype::F16)
        {
            const uint32_t min_ofm_per_wi = 16;
            const uint32_t min_batches_per_wi = 1;

            runInfo.cldnnStyle.ofmPerWorkItem = min_ofm_per_wi;
            if (batch_size % (4 * min_batches_per_wi * min_lws) == 0)
            {
                runInfo.cldnnStyle.batchesPerWorkItem = 4 * min_batches_per_wi; // USE_BLOCK_READ_2 + as_half4
            }
            else if (batch_size % (2 * min_batches_per_wi * min_lws) == 0)
            {
                runInfo.cldnnStyle.batchesPerWorkItem = 2 * min_batches_per_wi; // USE_BLOCK_READ_1 + as_half2
            }
            else
            {
                runInfo.cldnnStyle.batchesPerWorkItem = min_batches_per_wi;
            }
            
            runInfo.effiency = FORCE_PRIORITY_7;
        }
        else
        {
            runInfo.cldnnStyle.ofmPerWorkItem = 8;
            runInfo.cldnnStyle.batchesPerWorkItem = 2;
            runInfo.effiency = FORCE_PRIORITY_9;
        }

        runInfo.lws0 = min_lws;
        runInfo.gws0 = filter_ofm_num * batch_size / (runInfo.cldnnStyle.ofmPerWorkItem * runInfo.cldnnStyle.batchesPerWorkItem);
        
        return runInfo;
    }

    bool ConvolutionKernel_yxfb_yxio_b16::Validate(const Params& p, const OptionalParams& o) const
    {
        if (!ConvolutionKernelBase::Validate(p, o))
        {
            return false;
        }
        const ConvolutionParams& params = static_cast<const ConvolutionParams&>(p);

        const auto filter_ofm_num = params.weights.OFM().v;
        const auto batch_size = params.output.Batch().v;
        const uint32_t min_lws = 16;

        const bool bInputValidated =
            (filter_ofm_num > 0) &&
            (batch_size > 0) &&
            (params.output.Feature().v == filter_ofm_num);

        if (!bInputValidated)
        {
            return false;
        }

        if (params.inputs[0].GetDType() == Datatype::F16)
        {
            const uint32_t min_ofm_per_wi = 16;
            const uint32_t min_batches_per_wi = 1;

            const bool bFilterOK = filter_ofm_num % min_ofm_per_wi == 0;            // Number of output features dividable by minimum number of output features processed inside work item.
            const bool bBatchOK = batch_size % (min_batches_per_wi * min_lws) == 0; // Batch size dividable by minimum number of batches processed when smallest local work size is used.

            if (!bFilterOK || !bBatchOK)
            {
                return false;
            }
        }
        else
        {
            if ((filter_ofm_num * batch_size) % min_lws != 0 ||
                batch_size < 32) // TODO: check why it's not supported
            {
                return false;
            }
        }

        return true;
    }

    JitConstants ConvolutionKernel_yxfb_yxio_b16::GetJitConstants(const ConvolutionParams& params, DispatchData kd) const
    {
        auto jit = Parent::GetJitConstants(params, kd);

        if (params.inputs[0].GetDType() == Datatype::F32)
        {
            // A LITTLE HACK, for convolutions with low number of input features don't use block reads, and it will speed up by 25%
            // TODO - investigate why is this happening
            if (params.inputs[0].Feature().v > 4)
            {
                jit.AddConstant(MakeJitConstant("USE_BLOCK_READ_2", ""));
            }
        }
        else
        {
            const auto batch_size = params.output.Batch().v;
            const auto batch_pad_before = params.output.Batch().pad.before;
            const auto feature_pitch = params.output.Feature().pitch;

            if (batch_size >= 64 && (feature_pitch % 2 == 0) && (batch_pad_before % 2 == 0))
            {
                jit.AddConstant(MakeJitConstant("USE_BLOCK_READ_2", ""));
            }
            else if (batch_size >= 32 && (feature_pitch % 2 == 0) && (batch_pad_before % 2 == 0))
            {
                jit.AddConstant(MakeJitConstant("USE_BLOCK_READ_1", ""));
            }
        }

        return jit;
    }

    KernelsData ConvolutionKernel_yxfb_yxio_b16::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        return GetCommonKernelsData(params, options);
    }
}