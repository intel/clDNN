/*
// Copyright (c) 2018 Intel Corporation
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

#include "convolution_kernel_imad_3x3.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"

//
// Kernel specific constants
//
#define SIMD_SIZE             16
// Threshold value to calculate the block size.
#define OUT_BLOCK_THRESHOLD   7
// For images 7x7 it's 7 (default), for 14x14 and above it's 14.
#define OUT_BLOCK_WIDTH       7
// For images 7x7 it's 1 (default), for 14x14 and above it's 2.
#define OUT_BLOCK_HEIGHT      1

// Calculte actual value of OUT_BLOCK_WIDTH depending on X
static size_t getBlockWidth(size_t X) {
    // X must be multiple of OUT_BLOCK_THRESHOLD
    assert(X % OUT_BLOCK_THRESHOLD == 0);
    size_t Res = (X > OUT_BLOCK_THRESHOLD)
                  ? OUT_BLOCK_WIDTH * 2
                  : OUT_BLOCK_WIDTH;
    return Res;
}

// Calculte actual value of OUT_BLOCK_HEIGHT depending on Y
static size_t getBlockHeight(size_t Y) {
    // Y must be multiple of OUT_BLOCK_THRESHOLD
    assert(Y % OUT_BLOCK_THRESHOLD == 0);
    size_t Res = (Y > OUT_BLOCK_THRESHOLD)
                  ? OUT_BLOCK_HEIGHT * 2
                  : OUT_BLOCK_HEIGHT;
    return Res;
}


namespace kernel_selector {

    ParamsKey ConvolutionKernel_imad_3x3::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::INT8);
        k.EnableInputDataType(Datatype::UINT8);
        k.EnableOutputDataType(Datatype::INT8);
        k.EnableOutputDataType(Datatype::UINT8);
        k.EnableInputWeightsType(WeightsType::INT8);
        k.EnableInputWeightsType(WeightsType::UINT8);
        k.EnableInputLayout(DataLayout::b_fs_yx_fsv4);
        k.EnableOutputLayout(DataLayout::b_fs_yx_fsv4);
        k.EnableDifferentInputWeightsTypes();
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableDilation();
        k.EnableBiasPerFeature();
        k.EnableNonBiasTerm();
        k.EnableBatching();
        k.EnableInt8Quantization();
        k.EnableOutputCalibration();
        k.DisableTuning();
        return k;
    }

    KernelsData
    ConvolutionKernel_imad_3x3::GetKernelsData(
                                    const Params&          params,
                                    const optional_params& options) const
    {
        return GetCommonKernelsData(params, options);
    }

    JitConstants
    ConvolutionKernel_imad_3x3::GetJitConstants(
                                    const convolution_params& params,
                                    const DispatchData&       kd) const
    {
        auto mem_consts = Parent::GetJitConstants(params, kd);

        const auto& input = params.inputs[0];

        const auto& iDims   = input.GetDims();
        const auto& weights = params.weights;
        const auto& wDims   = weights.GetDims();
        const int iX  = DataTensor::Channelndex(
                            input.GetLayout(), Tensor::DataChannelName::X);
        const int iY  = DataTensor::Channelndex(
                            input.GetLayout(), Tensor::DataChannelName::Y);
        const int iB  = DataTensor::Channelndex(
                            input.GetLayout(), Tensor::DataChannelName::BATCH);
        const int iF  = DataTensor::Channelndex(
                            input.GetLayout(), Tensor::DataChannelName::FEATURE);
        const int wOD = WeightsTensor::Channelndex(
                            weights.GetLayout(), Tensor::WeightsChannelName::OFM);
        mem_consts.AddConstants({
            MakeJitConstant("_IMAD_DEFINES",   1),
            //MakeJitConstant("SCALE_FACTOR",     m_ScaleFactor), //(255.0f / 700000.0f);
            MakeJitConstant("_IW",              iDims[iX].v),
            MakeJitConstant("_IH",              iDims[iY].v),
            MakeJitConstant("_ID",              iDims[iF].v),
            MakeJitConstant("IWPAD",            iDims[iX].pad.before + iDims[iX].pad.after),
            MakeJitConstant("IHPAD",            iDims[iY].pad.before + iDims[iY].pad.after),
            MakeJitConstant("_OW",              iDims[iX].v),
            MakeJitConstant("_OH",              iDims[iY].v),
            MakeJitConstant("_OD",              wDims[wOD].v),
            MakeJitConstant("OWPAD",            0),
            MakeJitConstant("OHPAD",            0),
            MakeJitConstant("SIMD_SIZE",        SIMD_SIZE),
            MakeJitConstant("K_HEIGHT",         wDims[iY].v),
            MakeJitConstant("K_WIDTH",          wDims[iX].v),
            MakeJitConstant("K_STRIDE",         params.stride.x), // X and Y must be equal
            MakeJitConstant("BATCH_SIZE",       iDims[iB].v),
            MakeJitConstant("WORKGROUP_SIZE",   "SIMD_SIZE"),
            MakeJitConstant("OUT_BLOCK_WIDTH",  getBlockWidth(iDims[iX].v)),
            MakeJitConstant("OUT_BLOCK_HEIGHT", getBlockHeight(iDims[iY].v))
        });

        // FM_TILE definition
        mem_consts.AddConstants({
            MakeJitConstant("IMAD_LENGTH", 4),
            MakeJitConstant("SYSTOLIC_DEPTH", 1),
            MakeJitConstant("FM_TILE", "(IMAD_LENGTH * SYSTOLIC_DEPTH)")
        });

        if (input.GetDType() == Datatype::UINT8) {
            // For unsigned types IMAD convolution kernel should skip
            // all negative values.
            mem_consts.AddConstants({
                MakeJitConstant("CONVO_UNSIGNED", 1)
            });
        }

        if (params.output.GetLayout() != DataLayout::b_fs_yx_fsv4) {
            mem_consts.AddConstants({
                // Produce unswizzelled results.
                MakeJitConstant("TO_UNSWIZZLE", 1),
            });
        }

        return mem_consts;

    } // GetJitConstants


    ConvolutionKernelBase::DispatchData ConvolutionKernel_imad_3x3::SetDefault(
                                               const convolution_params& params,
                                               int) const
    {
        DispatchData kd;

        const auto& in      = params.inputs[0];
        const auto& weights = params.weights;
        const auto& iDims   = in.GetDims();
        const auto& wDims   = weights.GetDims();
        const int iX  = DataTensor::Channelndex(
                            in.GetLayout(), Tensor::DataChannelName::X);
        const int iY  = DataTensor::Channelndex(
                            in.GetLayout(), Tensor::DataChannelName::Y);
        const int iB  = DataTensor::Channelndex(
                            in.GetLayout(), Tensor::DataChannelName::BATCH);
        const int wOD = WeightsTensor::Channelndex(
                            weights.GetLayout(), Tensor::WeightsChannelName::OFM);

        std::vector<size_t> global = {
            //globalRange[0] = ((_IW / K_STRIDE) + (OTW - 1)) / OTW;
            // number of tiles needed to cover output width
            (((iDims[iX].v / params.stride.x) +
                   (getBlockWidth(iDims[iX].v) - 1)) / getBlockWidth(iDims[iX].v)),

            //globalRange[1] = ((_IH / K_STRIDE) + (OTH - 1)) / OTH;
            // number of tiles needed to cover output height
            (((iDims[iY].v / params.stride.y) +
                   (getBlockHeight(iDims[iY].v) - 1)) / getBlockHeight(iDims[iY].v)),

            // globalRange[2] = (_OD * _B) + ((_B *_OD) % __WORKGROUP_SIZE);
            // round depth range up
            ((wDims[wOD].v * iDims[iB].v) + ((wDims[wOD].v * iDims[iB].v) % SIMD_SIZE))
        };

        std::vector<size_t> local = {1, 1, SIMD_SIZE};

        kd.gws0 = global[0];
        kd.gws1 = global[1];
        kd.gws2 = global[2];

        kd.lws0 = local[0];
        kd.lws1 = local[1];
        kd.lws2 = local[2];

        kd.cldnnStyle = { 0 };
        kd.gemmStyle  = { 0 };
        kd.effiency   = FORCE_PRIORITY_1;

        return kd;

    } // SetDefault

    bool
    ConvolutionKernel_imad_3x3::Validate(
            const Params&          params,
            const optional_params& options) const
    {
        if (!Parent::Validate(params, options))
        {
            return false;
        }

        KernelData kd = KernelData::Default<convolution_params>(params);
        convolution_params& newParams = *static_cast<convolution_params*>(kd.params.get());
        return (newParams.filterSize.x == 3 && newParams.filterSize.y == 3);
    }

    KernelsData
    ConvolutionKernel_imad_3x3::GetCommonKernelsData(
                                const Params&          params,
                                const optional_params& options,
                                const std::string      exeMode,
                                int                    autoTuneIndex) const
    {
        if (!Validate(params, options))
        {
            return{};
        }

        KernelData kd = KernelData::Default<convolution_params>(params);
        convolution_params& newParams = *static_cast<convolution_params*>(kd.params.get());

        if (newParams.stride.x != newParams.stride.y) {
            // Strides must be equial
            return{};
        }
        else {
            const auto& in    = newParams.inputs[0];
            const auto& iDims = in.GetDims();
            const int iX = DataTensor::Channelndex(
                in.GetLayout(), Tensor::DataChannelName::X);
            if (iDims[iX].v % OUT_BLOCK_THRESHOLD != 0) {
                // Input size must be multiple of OUT_BLOCK_THRESHOLD
                return{};
            }
        }

        DispatchData runInfo = SetDefault(newParams, autoTuneIndex);

        if (!CheckWorkGroups(runInfo))
        {
            // Internal Error - wrong calculation of global/local work group sizes
            return{};
        }

        bool succeed = UpdateWeightsParams(
            newParams,
            options,
            GetSupportedWeightLayouts(newParams),
            kd.weightsReorderParams,
            GetSupportedKey());

        if (!succeed)
        {
            return{};
        }

        auto finalKernelName = GetKernelName(newParams);
        auto cldnnJit = GetJitConstants(newParams, runInfo);
        auto entryPoint = GetEntryPoint(finalKernelName, newParams.layerID, options);
        auto jit = CreateJit(finalKernelName, cldnnJit, entryPoint);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, runInfo, params.engineInfo, finalKernelName, jit, entryPoint, exeMode, true, !newParams.bias.empty(), 1, newParams.int8_quantization, newParams.output_calibration);

        kd.estimatedTime = runInfo.effiency;
        kd.autoTuneIndex = autoTuneIndex;

        return{ kd };

    } // GetCommonKernelsData
}
