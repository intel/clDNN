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

#include <string>
#include <cstddef>
#include <memory>
#include "common_types.h"
#include "common_tools.h"
#include "tensor_type.h"

namespace KernelSelector
{
    using DataTensor = Tensor::DataTensor;
    using WeightsTensor = Tensor::WeightsTensor;
    using DataLayout = Tensor::DataLayout;
    using WeightsLayout = Tensor::WeightsLayout;
    using MultiDataTensor = std::vector<DataTensor>;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ParamsKey
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class ParamsKey
    {
    public:
        ParamsKey()
        {
            key.restrict.raw = 0;
            key.machineInfo.raw = 0;
            key.inputType.raw = 0;
            key.outputType.raw = 0;
            key.inputWeightsType.raw = 0;
            key.outputWeightsType.raw = 0;
            key.inputLayout = 0;
            key.outputLayout = 0;
            key.weightsInputLayout = 0;
            key.weightsOutputLayout = 0;
        }

        struct Key
        {
            union restrict_t
            {
                struct val_t
                {
                    uint32_t different_types : 1;
                    uint32_t offset : 1;
                    uint32_t pitches : 1;
                    uint32_t batching : 1;
                    uint32_t biasPerFeatureMap : 1;
                    uint32_t biasPerOutput : 1;
                    uint32_t nonBias : 1;
                    uint32_t activationAdditionalParamsAsInput : 1;
                    uint32_t FP16Emulation : 1;

                    union dedicated_t
                    {
                        struct norm_t
                        {
                            uint32_t across : 1;
                            uint32_t within : 1;
                            uint32_t fixedKenrelDivider : 1;
                            uint32_t dynamicKenrelDivider : 1;
                        } norm;
                        struct pooling_t
                        {
                            uint32_t max : 1;
                            uint32_t avg : 1;
                            uint32_t floor : 1;
                            uint32_t ceil : 1;
                            uint32_t fixedKenrelDivider : 1;
                            uint32_t dynamicKenrelDivider : 1;
                            uint32_t dynamicKenrelDividerWithPadding : 1;
                        } pooling;
                        struct conv_t 
                        {
                            uint32_t split : 1;
                            uint32_t dilation : 1;
                            uint32_t depthwiseSeparableOpt : 1;
                        } conv;
                        struct fc_t {} fc;
                        struct softmax_t 
                        {
                            uint32_t dimX : 1;
                            uint32_t dimY : 1;
                            uint32_t dimFeature : 1;
                        } softmax;
                        struct concat_t
                        {
                            uint32_t axisX : 1;
                            uint32_t axisY : 1;
                            uint32_t axisFeature : 1;
                            uint32_t axisBatch : 1;
                            uint32_t kernelPerInput : 1;
                            uint32_t oneKernel : 1;
                        } concat;
                        struct upsample_t
                        {
                            uint32_t nearest : 1;
                            uint32_t bilinear : 1;
                        } upsample;
                        struct reorder_t
                        {
                            uint32_t winograd : 1;
                        } reorder;
                    } dedicated;
                } val;
                uint64_t raw;
            } restrict;

            union machine_info_t
            {
                struct val_t
                {
                    uint32_t subgroup : 1;
                    uint32_t subgroupShort : 1;
                } val;
                uint32_t raw;
            } machineInfo;

            static_assert(sizeof(restrict_t) == sizeof(uint64_t), "problem with union");

            typedef union DataTypesKey_t
            {
                struct val_t
                {
                    uint32_t int8 : 1;
                    uint32_t uint8 : 1;
                    uint32_t int16 : 1;
                    uint32_t uint16 : 1;
                    uint32_t int32 : 1;
                    uint32_t uint32 : 1;
                    uint32_t F16 : 1;
                    uint32_t F32 : 1;
                } val;
                uint32_t raw;
            } DataTypesKey;


            DataTypesKey inputType;
            DataTypesKey outputType;
            DataTypesKey inputWeightsType;
            DataTypesKey outputWeightsType;
            uint32_t inputLayout;
            uint32_t outputLayout;
            uint32_t weightsInputLayout;
            uint32_t weightsOutputLayout;
        };

        void EnableInputDataType(Datatype dt)
        {
            switch (dt)
            {
            case Datatype::INT8:
                key.inputType.val.int8 = 1;
                break;
            case Datatype::UINT8:
                key.inputType.val.uint8 = 1;
                break;
            case Datatype::INT16:
                key.inputType.val.int16 = 1;
                break;
            case Datatype::UINT16:
                key.inputType.val.uint16 = 1;
                break;
            case Datatype::INT32:
                key.inputType.val.int32 = 1;
                break;
            case Datatype::UINT32:
                key.inputType.val.uint32 = 1;
                break;
            case Datatype::F16:
                key.inputType.val.F16 = 1;
                break;
            case Datatype::F32:
                key.inputType.val.F32 = 1;
                break;
            default:
                break;
            }
        }

        void EnableAllInputDataType()
        {
            key.inputType.raw = 0xffffffff;
        }

        void EnableOutputDataType(Datatype dt)
        {
            switch (dt)
            {
            case Datatype::INT8:
                key.outputType.val.int8 = 1;
                break;
            case Datatype::UINT8:
                key.outputType.val.uint8 = 1;
                break;
            case Datatype::INT16:
                key.outputType.val.int16 = 1;
                break;
            case Datatype::UINT16:
                key.outputType.val.uint16 = 1;
                break;
            case Datatype::INT32:
                key.outputType.val.int32 = 1;
                break;
            case Datatype::UINT32:
                key.outputType.val.uint32 = 1;
                break;
            case Datatype::F16:
                key.outputType.val.F16 = 1;
                break;
            case Datatype::F32:
                key.outputType.val.F32 = 1;
                break;
            default:
                break;
            }
        }

        void EnableAllOutputDataType()
        {
            key.outputType.raw = 0xffffffff;
        }

        void EnableInputWeightsType(WeightsType wt)
        {
            switch (wt)
            {
            case WeightsType::F16:
                key.inputWeightsType.val.F16 = 1;
                break;
            case WeightsType::F32:
                key.inputWeightsType.val.F32 = 1;
                break;
            case WeightsType::INT8:
                key.inputWeightsType.val.int8 = 1;
                break;
            default:
                break;
            }
        }

        void EnableAllInputWeightsType()
        {
            key.inputWeightsType.raw = 0xffffffff;
        }

        void EnableOutputWeightsType(WeightsType wt)
        {
            switch (wt)
            {
            case WeightsType::F16:
                key.outputWeightsType.val.F16 = 1;
                break;
            case WeightsType::F32:
                key.outputWeightsType.val.F32 = 1;
                break;
            case WeightsType::INT8:
                key.outputWeightsType.val.int8 = 1;
                break;
            default:
                break;
            }
        }

        void EnableAllOutputWeightsType()
        {
            key.outputWeightsType.raw = 0xffffffff;
        }

        void EnableFP16Emulation()
        {
            key.restrict.val.FP16Emulation = 1;
        }

        void EnableDifferentTypes()
        {
            key.restrict.val.different_types = 1;
        }

        void EnableInputLayout(DataLayout l)
        {
            key.inputLayout |= (1 << l);
        }

        void EnableAllInputLayout()
        {
            key.inputLayout = 0xffffffff;
        }

        void EnableOutputLayout(DataLayout l)
        {
            key.outputLayout |= (1 << l);
        }

        void EnableAllOutputLayout()
        {
            key.outputLayout = 0xffffffff;
        }

        void EnableInputWeightsLayout(WeightsLayout l)
        {
            key.weightsInputLayout |= (1 << l);
        }

        void EnableAllInputWeightsLayout()
        {
            key.weightsInputLayout = 0xffffffff;
        }

        void EnableOutputWeightsLayout(WeightsLayout l)
        {
            key.weightsOutputLayout |= (1 << l);
        }

        void EnableAllOutputWeightsLayout()
        {
            key.weightsOutputLayout = 0xffffffff;
        }

        void EnableTensorOffset()
        {
            key.restrict.val.offset = 1;
        }

        void EnableTensorPitches()
        {
            key.restrict.val.pitches = 1;
        }

        void EnableBatching()
        {
            key.restrict.val.batching = 1;
        }

        void EnableSubGroup()
        {
            key.machineInfo.val.subgroup = 1;
        }

        void EnableSubGroupShort()
        {
            key.machineInfo.val.subgroupShort = 1;
        }

        void EnableNonBiasTerm()
        {
            key.restrict.val.nonBias = 1;
        }

        void EnableBiasPerFeature()
        {
            key.restrict.val.biasPerFeatureMap = 1;
        }

        void EnableBiasPerOutput()
        {
            key.restrict.val.biasPerOutput = 1;
        }

        void EnableActivationAdditionalParamsAsInput()
        {
            key.restrict.val.activationAdditionalParamsAsInput = 1;
        }

        void EnableLRNMode(LRNMode m)
        {
            switch (m)
            {
            case LRNMode::ACROSS_CHANNEL:
                key.restrict.val.dedicated.norm.across = 1;
                break;
            case LRNMode::WITHIN_CHANNEL:
                key.restrict.val.dedicated.norm.within = 1;
                break;
            default:
                break;
            }
        }

        void EnableNormalizeMode(NormalizeMode m)
        {
            switch (m)
            {
            case NormalizeMode::ACROSS_SPATIAL:
                key.restrict.val.dedicated.norm.across = 1;
                break;
            case NormalizeMode::WITHIN_SPATIAL:
                key.restrict.val.dedicated.norm.within = 1;
                break;
            default:
                break;
            }
        }

        void EnableLRNKernelDividerMode(KernelDividerMode m)
        {
            switch (m)
            {
            case KernelDividerMode::FIXED:
                key.restrict.val.dedicated.norm.fixedKenrelDivider = 1;
                break;
            case KernelDividerMode::DYNAMIC:
                key.restrict.val.dedicated.norm.dynamicKenrelDivider = 1;
                break;
            default:
                break;
            }
        }

        void EnablePoolKernelDividerMode(KernelDividerMode m)
        {
            switch (m)
            {
            case KernelDividerMode::FIXED:
                key.restrict.val.dedicated.pooling.fixedKenrelDivider = 1;
                break;
            case KernelDividerMode::DYNAMIC:
                key.restrict.val.dedicated.pooling.dynamicKenrelDivider = 1;
                break;
            case KernelDividerMode::DYNAMIC_WITH_PADDING:
                key.restrict.val.dedicated.pooling.dynamicKenrelDividerWithPadding = 1;
                break;
            default:
                break;
            }
        }

        void EnablePoolType(PoolType t)
        {
            switch (t)
            {
            case PoolType::MAX:
                key.restrict.val.dedicated.pooling.max = 1;
                break;
            case PoolType::AVG:
                key.restrict.val.dedicated.pooling.avg = 1;
                break;
            default:
                break;
            }
        }

        void EnablePoolRemainder(PoolRemainder r)
        {
            switch (r)
            {
            case PoolRemainder::FLOOR:
                key.restrict.val.dedicated.pooling.floor = 1;
                break;
            case PoolRemainder::CEIL:
                key.restrict.val.dedicated.pooling.ceil = 1;
                break;
            default:
                break;
            }
        }

        void EnableSplitSupport()
        {
            key.restrict.val.dedicated.conv.split = 1;
        }

        void EnableDilation()
        {
            key.restrict.val.dedicated.conv.dilation = 1;
        }

        void EnableDepthwiseSeparableOpt()
        {
            key.restrict.val.dedicated.conv.depthwiseSeparableOpt = 1;
        }

        void EnableWinogradReorder()
        {
            key.restrict.val.dedicated.reorder.winograd = 1;
        }

        void EnableSoftmaxDim(SoftmaxDim d)
        {
            switch (d)
            {
            case SoftmaxDim::X:
                key.restrict.val.dedicated.softmax.dimX = 1;
                break;
            case SoftmaxDim::Y:
                key.restrict.val.dedicated.softmax.dimY = 1;
                break;
            case SoftmaxDim::FEATURE:
                key.restrict.val.dedicated.softmax.dimFeature = 1;
                break;
            default:
                break;
            }
        }

        void EnableConcatAxis(ConcatAxis a)
        {
            switch (a)
            {
            case ConcatAxis::X:
                key.restrict.val.dedicated.concat.axisX = 1;
                break;
            case ConcatAxis::Y:
                key.restrict.val.dedicated.concat.axisY = 1;
                break;
            case ConcatAxis::FEATURE:
                key.restrict.val.dedicated.concat.axisFeature = 1;
                break;
            case ConcatAxis::BATCH:
                key.restrict.val.dedicated.concat.axisBatch = 1;
                break;
            default:
                break;
            }
        }

        void EnableUpSamplingSampleType(SampleType a)
        {
            switch (a)
            {
            case SampleType::NEAREST:
                key.restrict.val.dedicated.upsample.nearest = 1;
                break;
            case SampleType::BILINEAR:
                key.restrict.val.dedicated.upsample.bilinear = 1;
                break;
            default:
                break;
            }
        }

        void EnableConcatKernelPerInput()
        {
            key.restrict.val.dedicated.concat.kernelPerInput = 1;
        }

        void EnableConcatOneKernel()
        {
            key.restrict.val.dedicated.concat.oneKernel = 1;
        }

        bool Support(const ParamsKey& k) const
        {
            return
                ((key.restrict.raw & k.key.restrict.raw) == k.key.restrict.raw) && // check if this kernel supports this params
                ((key.machineInfo.raw & k.key.machineInfo.raw) == key.machineInfo.raw) && // check if machine supports this kernel
                ((key.inputType.raw & k.key.inputType.raw) == k.key.inputType.raw) &&
                ((key.outputType.raw & k.key.outputType.raw) == k.key.outputType.raw) &&
                ((key.inputWeightsType.raw & k.key.inputWeightsType.raw) == k.key.inputWeightsType.raw) &&
                ((key.outputWeightsType.raw & k.key.outputWeightsType.raw) == k.key.outputWeightsType.raw) &&
                ((key.inputLayout & k.key.inputLayout) != 0 || key.inputLayout == k.key.inputLayout) &&
                ((key.outputLayout & k.key.outputLayout) != 0 || key.outputLayout == k.key.outputLayout) &&
                ((key.weightsInputLayout & k.key.weightsInputLayout) != 0 || key.weightsInputLayout == k.key.weightsInputLayout) &&
                ((key.weightsOutputLayout & k.key.weightsOutputLayout) != 0 || key.weightsOutputLayout == k.key.weightsOutputLayout);
        }

        ParamsKey Merge(const ParamsKey& k) const
        {
            ParamsKey ret;
            ret.key.restrict.raw = key.restrict.raw | k.key.restrict.raw;
            ret.key.machineInfo.raw = key.machineInfo.raw | k.key.machineInfo.raw;
            ret.key.inputType.raw = key.inputType.raw | k.key.inputType.raw;
            ret.key.outputType.raw = key.outputType.raw | k.key.outputType.raw;
            ret.key.inputWeightsType.raw = key.inputWeightsType.raw | k.key.inputWeightsType.raw;
            ret.key.outputWeightsType.raw = key.outputWeightsType.raw | k.key.outputWeightsType.raw;
            ret.key.inputLayout = key.inputLayout | k.key.inputLayout;
            ret.key.outputLayout = key.outputLayout | k.key.outputLayout;
            ret.key.weightsInputLayout = key.weightsInputLayout | k.key.weightsInputLayout;
            ret.key.weightsOutputLayout = key.weightsOutputLayout | k.key.weightsOutputLayout;
            return ret;
        }

    private:
        Key key;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // EngineInfo
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct EngineInfo
    {
        bool bSubGroupSupport = false;
        bool bSubGroupShortSupport = false;
        bool bFP16Support = false;
        bool bFP64Support = false;
        uint64_t maxWorkGroupSize = 0;
        uint64_t maxLocalMemSize = 0;
        std::string deviceId = "";
        std::string driverVersion = "";
        std::string hostVersion = "";
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct Params
    {
        virtual ~Params() {}

        KernelType GetType() const { return kType; }
        virtual ParamsKey GetParamsKey() const
        {
            ParamsKey k;

            if (engineInfo.bSubGroupSupport)
            {
                k.EnableSubGroup();
            }

            if (engineInfo.bSubGroupShortSupport)
            {
                k.EnableSubGroupShort();
            }

            return k;
        }

    protected:
        Params(KernelType kt, const std::string& id) : kType(kt), layerID(id) {}
        KernelType kType;

    public:
        std::string layerID;
        EngineInfo engineInfo;

        virtual std::string to_string() const;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // BaseParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct BaseParams : public Params
    {
        virtual ~BaseParams() {}

        ActivationFunction  activationFunc = ActivationFunction::NONE;
        NonLinearParams     activationParams;
        MultiDataTensor     inputs;
        DataTensor          output;        

        virtual std::string to_string() const;

        virtual ParamsKey GetParamsKey() const
        {
            ParamsKey k = Params::GetParamsKey();

            bool bBatching = false;
            bool bPitches = false;
            bool bOffests = false;
            bool bDifferentTypes = false;
            bool bFP16Used = (output.GetDType() == Datatype::F16);

            for (const auto& i : inputs)
            {
                k.EnableInputDataType(i.GetDType());
                k.EnableInputLayout(i.GetLayout());

                bBatching       |= (i.Batch().v > 1);
                bPitches        |= (i.PitchesDifferFromLogicalDims());
                bOffests        |= (i.GetFirstElementOffset() != 0);
                bDifferentTypes |= (i.GetDType() != output.GetDType());
                bFP16Used       |= (i.GetDType() == Datatype::F16);
            }

            k.EnableOutputDataType(output.GetDType());
            k.EnableOutputLayout(output.GetLayout());

            if (bBatching)
            {
                k.EnableBatching();
            }

            if (bPitches ||
                output.PitchesDifferFromLogicalDims())
            {
                k.EnableTensorPitches();
            }

            if (bDifferentTypes)
            {
                k.EnableDifferentTypes();
            }

            if (bOffests ||
                output.GetFirstElementOffset() != 0)
            {
                k.EnableTensorOffset();
            }

            if (!engineInfo.bFP16Support &&
                bFP16Used)
            {
                // I'm not sure it's the best idea, but we can live with it right now
                k.EnableFP16Emulation();
            }

            return k;
        }

    protected:

        BaseParams(KernelType kt) : Params(kt, ""), inputs(1){}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ConvolutionParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct WeightBiasParams : public BaseParams
    {
        WeightBiasParams(KernelType kt) : BaseParams(kt) {}

        WeightsTensor weights;
        MultiDataTensor bias;

        virtual ParamsKey GetParamsKey() const override
        {
            ParamsKey k = BaseParams::GetParamsKey();

            k.EnableInputWeightsType(weights.GetDType());
            
            // not needed - can be changed by reorder params
            //k.EnableWeightsLayout(weights.layout);

            assert(bias.size() <= 1);

            if (bias.empty())
            {
                k.EnableNonBiasTerm();
            }
            else if (bias[0].GetLayout() == DataLayout::bf ||
                     bias[0].GetLayout() == DataLayout::fb)
            {
                k.EnableBiasPerFeature();
            }
            else if (bias[0].GetLayout() == output.GetLayout())
            {
                k.EnableBiasPerOutput();
            }

            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ConvolutionParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ConvolutionParams : public WeightBiasParams
    {
        ConvolutionParams() : WeightBiasParams(KernelType::CONVOLUTION), convParams() {}
    
        struct DedicatedParams
        {
            uSize    filterSize;
            uSize    stride;
            uSize    dilation;
            uSize    padding;
            uint32_t winograd_tile_n;
            uint32_t winograd_tile_m;
            uint32_t winograd_input_tile_width;
            uint32_t winograd_input_tile_height;
            uint32_t split = 1;
            bool     depthwiseSeparableOpt = false;
        };

        DedicatedParams convParams;

        virtual std::string to_string() const override;

        virtual ParamsKey GetParamsKey() const override
        {
            ParamsKey k = WeightBiasParams::GetParamsKey();

            if (convParams.split > 1)
            {
                k.EnableSplitSupport();
            }

            if (convParams.dilation.x != 1 ||
                convParams.dilation.y != 1)
            {
                k.EnableDilation();
            }

            if (convParams.depthwiseSeparableOpt)
            {
                k.EnableDepthwiseSeparableOpt();
            }

            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // DeconvolutionParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct DeconvolutionParams : public WeightBiasParams
    {
        DeconvolutionParams() : WeightBiasParams(KernelType::DECONVOLUTION), deconvParams() {}

        struct DedicatedParams
        {
            uSize    filterSize;
            uSize    stride;
            uSize    dilation;
            uSize    padding;
            uint32_t split = 1;
            bool     depthwiseSeparableOpt = false;
        };

        DedicatedParams deconvParams;

        virtual std::string to_string() const override;

        virtual ParamsKey GetParamsKey() const override
        {
            ParamsKey k = WeightBiasParams::GetParamsKey();

            if (deconvParams.split > 1)
            {
                k.EnableSplitSupport();
            }

            if (deconvParams.dilation.x != 1 ||
                deconvParams.dilation.y != 1)
            {
                k.EnableDilation();
            }

            if (deconvParams.depthwiseSeparableOpt)
            {
                k.EnableDepthwiseSeparableOpt();
            }

            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // LRNParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct LRNParams : public BaseParams
    {
        LRNParams() : BaseParams(KernelType::LRN), lrnParams() {}

        struct DedicatedParams
        {
            LRNMode             normMode    = LRNMode::ACROSS_CHANNEL;
            KernelDividerMode   divMode     = KernelDividerMode::DONT_CARE;
            float               alpha       = 0.f;
            float               beta        = 0.f;
            float               k           = 0.f;
            uint32_t            localSize   = 0;
        };

        DedicatedParams lrnParams;

        virtual ParamsKey GetParamsKey() const
        {
            ParamsKey k = BaseParams::GetParamsKey();

            k.EnableLRNMode(lrnParams.normMode);
            k.EnableLRNKernelDividerMode(lrnParams.divMode);

            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // NormalizeParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct NormalizeParams : public BaseParams
    {
        NormalizeParams() : BaseParams(KernelType::NORMALIZE), normParams() {}

        struct DedicatedParams
        {
            NormalizeMode normMode = NormalizeMode::ACROSS_SPATIAL;
            float         epsilon  = 1e-10f;
            DataTensor    scaleTable;
        };

        DedicatedParams normParams;

        virtual ParamsKey GetParamsKey() const
        {
            ParamsKey k = BaseParams::GetParamsKey();

            k.EnableNormalizeMode(normParams.normMode);

            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // PoolingParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct PoolingParams : public BaseParams
    {
        PoolingParams() : BaseParams(KernelType::POOLING), poolParams() {}

        struct DedicatedParams
        {
            PoolType            poolType        = PoolType::MAX;
            PoolRemainder       remainderAction = PoolRemainder::FLOOR;
            KernelDividerMode   divMode         = KernelDividerMode::DONT_CARE;
            uSize               poolSize;
            uSize               poolStride;
            uSize               poolPad;
        };

        DedicatedParams poolParams;

        virtual ParamsKey GetParamsKey() const
        {
            ParamsKey k = BaseParams::GetParamsKey();

            k.EnablePoolType(poolParams.poolType);
            k.EnablePoolRemainder(poolParams.remainderAction);
            k.EnablePoolKernelDividerMode(poolParams.divMode);

            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ROIPoolingParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ROIPoolingParams : public BaseParams
    {
        ROIPoolingParams() : BaseParams(KernelType::ROI_POOLING) {}

        struct DedicatedParams
        {
            PoolType    mode         = PoolType::MAX;
            size_t      pooledWidth  = 0;
            size_t      pooledHeight = 0;
            size_t      groupSize    = 0;
            float       spatialScale = 1.f;
        };

        DedicatedParams roiParams;

        virtual ParamsKey GetParamsKey() const
        {
            return BaseParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // FullyConnectedParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct FullyConnectedParams : public WeightBiasParams
    {
        FullyConnectedParams() : WeightBiasParams(KernelType::FULLY_CONNECTED) {}

        virtual ParamsKey GetParamsKey() const
        {
            return WeightBiasParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ActivationParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ActivationParams : public BaseParams
    {
        ActivationParams() : BaseParams(KernelType::ACTIVATION), actParams() {}

        struct DedicatedParams
        {
            MultiDataTensor inputActivationParams;
        };

        DedicatedParams actParams;

        virtual ParamsKey GetParamsKey() const
        {
            auto k = BaseParams::GetParamsKey();
            if (!actParams.inputActivationParams.empty())
            {
                k.EnableActivationAdditionalParamsAsInput();
            }
            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // SoftMaxParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct SoftmaxParams : public BaseParams
    {
        SoftmaxParams() : BaseParams(KernelType::SOFT_MAX) {}

        struct DedicatedParams
        {
            SoftmaxDim dim = SoftmaxDim::FEATURE;
        };

        DedicatedParams smParams;

        virtual ParamsKey GetParamsKey() const
        {
            auto k = BaseParams::GetParamsKey();
            k.EnableSoftmaxDim(smParams.dim);
            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // EltwiseParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct EltwiseParams : public BaseParams
    {
        EltwiseParams() : BaseParams(KernelType::ELTWISE), eltwiseParams() {}

        struct InputType
        {
            EltwiseInputMode mode       = EltwiseInputMode::INPUT_BUFFER;
            uint32_t         index      = 0;    // for inputs results;
            uint32_t         tmpIndex   = 0;    // for temp results;
            float            scalar     = 0.f;

            static InputType Buffer(uint32_t index) 
            {
                EltwiseParams::InputType input;
                input.mode = EltwiseInputMode::INPUT_BUFFER;
                input.index = index;
                return input;
            }

            static InputType UnorderedAccessBuffer(uint32_t index, uint32_t tmpIndex)
            {
                EltwiseParams::InputType input;
                input.mode = EltwiseInputMode::UNORDERED_ACCESS_INPUT_BUFFER;
                input.index = index;
                input.tmpIndex = tmpIndex;
                return input;
            }

            static InputType Intermediate(uint32_t tmpIndex)
            {
                EltwiseParams::InputType input;
                input.mode = EltwiseInputMode::INTERMEDIATE_RESULTS_INDEX;
                input.tmpIndex = tmpIndex;
                return input;
            }

            static InputType Scalar(float s)
            {
                EltwiseParams::InputType input;
                input.mode = EltwiseInputMode::SCALAR;
                input.scalar = s;
                return input;
            }
        };

        struct Node
        {
            std::vector<InputType> inputs;
            EltwiseMode mode;
        };

        struct DedicatedParams
        {
            std::vector<EltwiseParams::Node> operations;
            bool layoutBased = false;
        };

        DedicatedParams eltwiseParams;

        virtual ParamsKey GetParamsKey() const
        {
            return BaseParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ReorderParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ReorderBaseParams : public BaseParams
    {
        ReorderBaseParams() : BaseParams(KernelType::REORDER) {}

        virtual ParamsKey GetParamsKey() const
        {
            return BaseParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // PermuteParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct PermuteParams : public ReorderBaseParams
    {
        PermuteParams() {}
        
        struct DedicatedParams
        {
            std::vector<uint16_t> order;
        };

        DedicatedParams permuteParams;

        virtual ParamsKey GetParamsKey() const
        {
            return BaseParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ReorderParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ReorderParams : public ReorderBaseParams
    {
        ReorderParams() : reorderParams() {}

        struct DedicatedParams
        {
            MeanSubtractMode    mode = MeanSubtractMode::NONE;
            std::vector<float>  meanValues;
            DataTensor          mean;
            uint32_t            winograd_input_offset_x;
            uint32_t            winograd_input_offset_y;
            uint32_t            winograd_nr_tiles_x;
            bool                winograd = false;
        };

        DedicatedParams reorderParams;

        virtual ParamsKey GetParamsKey() const
        {
            auto k = ReorderBaseParams::GetParamsKey();

            if (reorderParams.winograd)
            {
                k.EnableWinogradReorder();
            }
            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ReorderWeightsParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ReorderWeightsParams : public Params
    {
        ReorderWeightsParams() : Params(KernelType::REORDER, ""), reorderParams() {}

        struct DedicatedParams
        {
            WeightsTensor input;
            WeightsTensor output;
            bool winograd = false;
        };

        DedicatedParams reorderParams;

        virtual ParamsKey GetParamsKey() const
        {
            ParamsKey k;
            const auto& input = reorderParams.input;
            const auto& output = reorderParams.output;
            k.EnableInputWeightsType(input.GetDType());
            k.EnableOutputWeightsType(output.GetDType());
            k.EnableInputWeightsLayout(input.GetLayout());
            k.EnableOutputWeightsLayout(output.GetLayout());

            if (input.PitchesDifferFromLogicalDims() ||
                output.PitchesDifferFromLogicalDims())
            {
                k.EnableTensorPitches();
            }

            if (input.GetFirstElementOffset() != 0 || output.GetFirstElementOffset() != 0)
            {
                k.EnableTensorOffset();
            }

            if (reorderParams.winograd)
            {
                k.EnableWinogradReorder();
            }
            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ConcatenationParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ConcatenationParams : public BaseParams
    {
        ConcatenationParams() : BaseParams(KernelType::CONCATENATION), concatParams() {}

        struct DedicatedParams
        {
            ConcatAxis axis = ConcatAxis::FEATURE;
        };

        DedicatedParams concatParams;

        virtual ParamsKey GetParamsKey() const
        {
            auto k =  BaseParams::GetParamsKey();
            k.EnableConcatAxis(concatParams.axis);
            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // UpSamplingParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct UpSamplingParams : public BaseParams
    {
        UpSamplingParams() : BaseParams(KernelType::UPSAMPLING) {}

        struct DedicatedParams
        {
            uint32_t scale = 1;
            uint32_t num_filter = 0;
            SampleType sampleType = SampleType::NEAREST;
        };

        DedicatedParams usParams;

        virtual ParamsKey GetParamsKey() const
        {
            auto k = BaseParams::GetParamsKey();
            k.EnableUpSamplingSampleType(usParams.sampleType);
            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Auto tuner parameters
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class KernelRunnerInterface;
    struct TuningParams
    {
        TuningMode mode;
        std::string cacheFilePath;
        std::shared_ptr<KernelRunnerInterface> runner;

        TuningParams() : mode(TuningMode::TUNING_DISABLED), cacheFilePath(""), runner(nullptr) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // OptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct OptionalParams
    {
        virtual ~OptionalParams() {}

        KernelType GetType() const { return kType; }

        std::vector<DataLayout> inputLayouts;
        std::vector<DataLayout> outputLayouts;

        bool meaningfulKernelsNames     = false;    // use layer name instead of internal kernel name
        bool allowStaticInputReordering = true;     // allow kernel to provide a kernel which reorder static data like weights/bias/tables...
        bool allowInputReordering       = false;    // allow kernel to ask graph compiler to reorder the input data before executing its
        bool allowOutputReordering      = false;    // allow kernel to ask graph compiler to reorder the output data before executing the next kernel

        TuningParams tuningParams;

        virtual ParamsKey GetSupportedKey() const
        {
            ParamsKey k;

            for (auto l : inputLayouts)
            {
                k.EnableInputLayout(l);
            }

            for (auto l : outputLayouts)
            {
                k.EnableOutputLayout(l);
            }

            return k;
        }

    protected:
        OptionalParams(KernelType kt) : kType(kt) {}
        KernelType kType;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // WeightsBiasOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct WeightsBiasOptionalParams : OptionalParams
    {
    protected:
        WeightsBiasOptionalParams(KernelType kt) : OptionalParams(kt) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ConvolutionOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ConvolutionOptionalParams : WeightsBiasOptionalParams
    {
        ConvolutionOptionalParams() : WeightsBiasOptionalParams(KernelType::CONVOLUTION) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // DeconvolutionOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct DeconvolutionOptionalParams : WeightsBiasOptionalParams
    {
        DeconvolutionOptionalParams() : WeightsBiasOptionalParams(KernelType::DECONVOLUTION) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // LRNOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct LRNOptionalParams : OptionalParams
    {
        LRNOptionalParams() : OptionalParams(KernelType::LRN) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // NormalizeOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct NormalizeOptionalParams : OptionalParams
    {
        NormalizeOptionalParams() : OptionalParams(KernelType::NORMALIZE) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // PoolingOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct PoolingOptionalParams : OptionalParams
    {
        PoolingOptionalParams() : OptionalParams(KernelType::POOLING) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ROIPoolingOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ROIPoolingOptionalParams : OptionalParams
    {
        ROIPoolingOptionalParams() : OptionalParams(KernelType::ROI_POOLING) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // FullyConnectedOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct FullyConnectedOptionalParams : WeightsBiasOptionalParams
    {
        FullyConnectedOptionalParams() : WeightsBiasOptionalParams(KernelType::FULLY_CONNECTED) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ActivationOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ActivationOptionalParams : OptionalParams
    {
        ActivationOptionalParams() : OptionalParams(KernelType::ACTIVATION) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // SoftmaxOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct SoftmaxOptionalParams : OptionalParams
    {
        SoftmaxOptionalParams() : OptionalParams(KernelType::SOFT_MAX) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // EltwiseOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct EltwiseOptionalParams : OptionalParams
    {
        EltwiseOptionalParams() : OptionalParams(KernelType::ELTWISE) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ReorderVxOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ReorderVxOptionalParams : OptionalParams
    {
        ReorderVxOptionalParams() : OptionalParams(KernelType::REORDER) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ReorderVxOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ReorderOptionalParams : OptionalParams
    {
        ReorderOptionalParams() : OptionalParams(KernelType::REORDER) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ConcatenationOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ConcatenationOptionalParams : OptionalParams
    {
        ConcatenationOptionalParams() : OptionalParams(KernelType::CONCATENATION) {}
        bool kernelPerInput = true;

        virtual ParamsKey GetSupportedKey() const
        {
            ParamsKey k = OptionalParams::GetSupportedKey();

            if (kernelPerInput)
            {
                k.EnableConcatKernelPerInput();
            }
            else
            {
                k.EnableConcatOneKernel();
            }

            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // UpSamplingOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct UpSamplingOptionalParams : OptionalParams
    {
        UpSamplingOptionalParams() : OptionalParams(KernelType::UPSAMPLING) {}
    };
}
