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

#include "jitter.h"
#include "tensor_type.h"
#include "kernel_selector_common.h"
#include "reorder/reorder_weights_kernel_selector.h"

namespace KernelSelector { namespace
{
    inline bool CheckConvolutionPaddedInputDesc(const ConvolutionParams& params, const DataTensor& reqDesc)
    {
        assert(params.inputs.size() == 1);

        bool properPadding =
            reqDesc.X().pad.before          <= params.inputs[0].X().pad.before &&
            reqDesc.Y().pad.before          <= params.inputs[0].Y().pad.before &&
            reqDesc.Feature().pad.before    <= params.inputs[0].Feature().pad.before &&
            reqDesc.Batch().pad.before      <= params.inputs[0].Batch().pad.before;

        properPadding &=
            reqDesc.X().pad.after           <= params.inputs[0].X().pad.after &&
            reqDesc.Y().pad.after           <= params.inputs[0].Y().pad.after &&
            reqDesc.Feature().pad.after     <= params.inputs[0].Feature().pad.after &&
            reqDesc.Batch().pad.after       <= params.inputs[0].Batch().pad.after;

        const auto& cp = params.convParams;
        properPadding &= ((cp.padding.x == 0 && cp.padding.y == 0) || params.inputs[0].GetPaddedVal() == 0.f);

        return properPadding;
    }

    inline DataTensor GetConvolutionBFYXPaddedTensor(const ConvolutionParams& params)
    {
        assert(params.inputs.size() == 1);
        assert(params.inputs[0].GetDims().size() == 4U);

        DataTensor t = params.inputs[0];
        std::vector<Tensor::Pad> pad{ { 0,0 },{ 0,0 },{ 0,0 },{ 0,0 } };

        const auto& cp = params.convParams;

        pad[0].before = cp.padding.x;
        pad[1].before = cp.padding.y;

        const auto inputLimitX = (params.output.X().v - 1) * cp.stride.x + (cp.filterSize.x - 1) * cp.dilation.x + 1;
        const auto inputLimitY = (params.output.Y().v - 1) * cp.stride.y + (cp.filterSize.y - 1) * cp.dilation.y + 1;

        pad[0].after = (size_t)std::max((int)inputLimitX - (int)t.X().v - (int)pad[0].before, (int)0);
        pad[1].after = (size_t)std::max((int)inputLimitY - (int)t.Y().v - (int)pad[1].before, (int)0);

        Tensor::NDims dims(4);
        const Tensor::NDims& orgDims = params.inputs[0].GetDims();
        size_t pitch = 1;
        for (size_t i = 0; i < dims.size(); i++)
        {
            dims[i].pad = pad[i];
            dims[i].v = orgDims[i].v;
            dims[i].pitch = pitch;
            pitch *= dims[i].LogicalDimPadded();
        }

        return{dims, t.GetDType(), t.GetLayout()};
    }

    inline WeightsType DataTypeToWeightsType(Datatype t)
    {
        switch (t)
        {
        case Datatype::F16: return WeightsType::F16;
        case Datatype::F32: return WeightsType::F32;
        default:
            return WeightsType::UNSUPPORTED;
        }
    }

    inline bool CheckWeights(const WeightsTensor& tensor, WeightsType reqType, std::vector<WeightsLayout> reqLayouts)
    {
        if (reqType != tensor.GetDType())
        {
            return false;
        }

        bool bProperWeightsLayout = std::find(reqLayouts.begin(), reqLayouts.end(), tensor.GetLayout()) != reqLayouts.end();
        if (!bProperWeightsLayout && tensor.PitchesDifferFromLogicalDims() == false)
        {
            bProperWeightsLayout =
                (std::find(reqLayouts.begin(), reqLayouts.end(), WeightsLayout::io) != reqLayouts.end() && tensor.GetLayout() == WeightsLayout::iyxo) ||
                (std::find(reqLayouts.begin(), reqLayouts.end(), WeightsLayout::oi) != reqLayouts.end() && tensor.GetLayout() == WeightsLayout::oiyx);
        }

        return bProperWeightsLayout;
    }

    inline bool UpdateWeightsParams(WeightBiasParams& newParams, const OptionalParams& options, std::vector<WeightsLayout> layouts, WeightsReorderParams& weightsReorderParams)
    {
        // TODO: handle padding per in x/y (for openvx)

        const WeightsBiasOptionalParams& optParams = static_cast<const WeightsBiasOptionalParams&>(options);

        const auto dtype = DataTypeToWeightsType(newParams.inputs[0].GetDType());
        bool bProperWeights = CheckWeights(newParams.weights, dtype, layouts);

        if (!bProperWeights)
        {
            if (!optParams.allowWeightsReorder)
            {
                return false;
            }

            auto& reorderKS = ReorderWeightsKernelSelctor::Instance();
            ReorderWeightsParams r_params;

            r_params.layerID = newParams.layerID + "_reorder_";
            r_params.reorderParams.input = newParams.weights;
            r_params.reorderParams.output = newParams.weights.TransformIgnorePadding(layouts[0], dtype);

            ReorderOptionalParams op;
            KernelsData kernels_data = reorderKS.GetBestKernels(r_params, op);

            if (kernels_data.empty())
            {
                return false;
            }

            weightsReorderParams.engine = WeightsReorderParams::Engine::GPU;
            weightsReorderParams.clKernel = std::make_shared<clKernelData>(kernels_data[0].kernels[0]);
            weightsReorderParams.newBufferSize = r_params.reorderParams.output.PhysicalSizeInBytes();
            weightsReorderParams.dtype = dtype;

            newParams.weights = r_params.reorderParams.output;
        }

        return true;
    }

    inline JitConstants GetTensorFriendlyWorkGroupsJit(const DataTensor& t)
    {
        auto b = Tensor::Channelndex(t.GetLayout(), Tensor::DataChannelName::BATCH);
        auto f = Tensor::Channelndex(t.GetLayout(), Tensor::DataChannelName::FEATURE);
        auto x = Tensor::Channelndex(t.GetLayout(), Tensor::DataChannelName::X);

        if (x == -1)
        {
            x = 2;
        }
        else
        {
            b = (b < x) ? b : b - 1;
            f = (f < x) ? f : f - 1;
        }

        JitConstants jit{
            MakeJitConstant("GWS_BATCH", b),
            MakeJitConstant("GWS_FEATURE", f),
            MakeJitConstant("GWS_YX", x),
        };

        return jit;
    }

    inline std::vector<size_t> GetTensorFriendlyWorkGroups(const DataTensor& t)
    {
        std::vector<size_t> sizes;
        auto y = Tensor::Channelndex(t.GetLayout(), Tensor::DataChannelName::Y);
        for (size_t i = 0; i < t.GetDims().size(); i++)
        {
            const auto& o = t.GetDims()[i];
            if (y == (int)i)
            {
                sizes.back() *= o.v;
            }
            else
            {
                sizes.push_back(o.v);
            }
        }

        for (size_t i = sizes.size(); i < 3; i++)
        {
            sizes.push_back(1U);
        }

        return sizes;
    }

    inline std::vector<size_t> GetOptimalLocalWorkGroupSizes(std::vector<size_t> gws)
    {
        const size_t lws_max = 256;
        const size_t optimal_lws_values[] = { 256, 224, 192, 160, 128, 96, 64, 32, 16, 8, 7, 6, 5, 4, 3, 2, 1 };
        size_t total_lws = 1;
        std::vector<size_t> lws;
        for (size_t i = 0; i < gws.size(); ++i)
        {
            auto rest_lws = lws_max / total_lws;
            size_t lws_idx = 0;
            while (rest_lws < optimal_lws_values[lws_idx]) lws_idx++;

            while (gws[i] % optimal_lws_values[lws_idx]) lws_idx++;

            lws.push_back(optimal_lws_values[lws_idx]);
            total_lws *= optimal_lws_values[lws_idx];
        }

        return lws;
    }

    inline bool CheckInputsOutputNoPitchSameDims(const BaseParams& params)
    {
        bool no_pitch_same_dims = true;

        if (params.inputs.size())
        {
            no_pitch_same_dims = !params.inputs[0].PitchesDifferFromLogicalDims();

            for (size_t i = 1; i < params.inputs.size(); i++)
            {
                no_pitch_same_dims = no_pitch_same_dims && (params.inputs[0] == params.inputs[i]);
            }

            no_pitch_same_dims = no_pitch_same_dims && (params.inputs[0] == params.output);
        }

        return no_pitch_same_dims;
    }
} }