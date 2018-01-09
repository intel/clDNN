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
#include "api/C/cldnn.h"
#include "api/CPP/program.hpp"
#include "program_impl.h"
#include "gpu/ocl_toolkit.h"
#include "tensor_type.h"
#include "kernel_selector_params.h"
#include "convolution/convolution_kernel_selector.h"
#include "deconvolution/deconvolution_kernel_selector.h"
#include "lrn/lrn_kernel_selector.h"
#include "normalize/normalize_kernel_selector.h"
#include "pooling/pooling_kernel_selector.h"
#include "roi_pooling/roi_pooling_kernel_selector.h"
#include "fully_connected/fully_connected_kernel_selector.h"
#include "activation/activation_kernel_selector.h"
#include "softmax/softmax_kernel_selector.h"
#include "region_yolo/region_yolo_kernel_selector.h"
#include "reorg_yolo/reorg_yolo_kernel_selector.h"
#include "eltwise/eltwise_kernel_selector.h"
#include "reorder/reorder_kernel_selector.h"
#include "permute/permute_kernel_selector.h"
#include "reshape/reshape_kernel_selector.h"
#include "concatenation/concatenation_kernel_selector.h"
#include "upsampling/upsampling_kernel_selector.h"
#include "jitter.h"

using namespace cldnn;

namespace kernel_selector
{
    using n_dims                            = KernelSelector::Tensor::NDims;
    using kernel_data                       = KernelSelector::KernelData;
    using kernel_string                     = KernelSelector::KernelString;
    using cl_kernel_data                    = KernelSelector::clKernelData;
    using kernel_arguments                  = KernelSelector::Arguments;
    using kernel_argument_element           = KernelSelector::ArgumentDescriptor;
    using kernel_argument_types             = KernelSelector::ArgumentDescriptor::Types;
    using kernel_scalar_arguments           = KernelSelector::Scalars;
    using kernel_scalar_argument_types      = KernelSelector::ScalarDescriptor::Types;
    using jit_constants                     = KernelSelector::JitConstants;

    using data_type                         = KernelSelector::Datatype;
    using kernel_type                       = KernelSelector::KernelType;
    using weights_type                      = KernelSelector::WeightsType;
    using activation_function               = KernelSelector::ActivationFunction;
    using pool_type                         = KernelSelector::PoolType;
    using pool_remainder                    = KernelSelector::PoolRemainder;
    using lrn_mode                          = KernelSelector::LRNMode;
    using normalize_mode                    = KernelSelector::NormalizeMode;
    using kernel_divider_mode               = KernelSelector::KernelDividerMode;
    using eltwise_mode                      = KernelSelector::EltwiseMode;
    using eltwise_input_mode                = KernelSelector::EltwiseInputMode;
    using softmax_dim                       = KernelSelector::SoftmaxDim;
    using mean_subtruct_mode                = KernelSelector::MeanSubtractMode;
    using concat_axis                       = KernelSelector::ConcatAxis;
    using tuning_mode                       = KernelSelector::TuningMode;
    using sample_type                       = KernelSelector::SampleType;

    using data_tensor                       = KernelSelector::DataTensor;
    using weights_tensor                    = KernelSelector::WeightsTensor;
    using data_layout                       = KernelSelector::DataLayout;
    using weights_layout                    = KernelSelector::WeightsLayout;
    using multi_data_tensor                 = KernelSelector::MultiDataTensor;

    using params                            = KernelSelector::Params;
    using base_params                       = KernelSelector::BaseParams;
    using weight_bias_params                = KernelSelector::WeightBiasParams;
    using convolution_params                = KernelSelector::ConvolutionParams;
    using deconvolution_params              = KernelSelector::DeconvolutionParams;
    using lrn_params                        = KernelSelector::LRNParams;
    using normalize_params                  = KernelSelector::NormalizeParams;
    using pooling_params                    = KernelSelector::PoolingParams;
    using roi_pooling_v1_params             = KernelSelector::ROIPoolingParams;
    using fully_connected_params            = KernelSelector::FullyConnectedParams;
    using activation_params                 = KernelSelector::ActivationParams;
    using softmax_params                    = KernelSelector::SoftmaxParams;
    using region_yolo_params                = KernelSelector::RegionYoloParams;
    using reorg_yolo_params                 = KernelSelector::ReorgYoloParams;
    using eltwise_params                    = KernelSelector::EltwiseParams;
    using reorder_base_params               = KernelSelector::ReorderBaseParams;
    using permute_params                    = KernelSelector::PermuteParams;
    using reorder_params                    = KernelSelector::ReorderParams;
    using reorder_weights_params            = KernelSelector::ReorderWeightsParams;
    using concatenation_params              = KernelSelector::ConcatenationParams;
    using weights_reorder_params            = KernelSelector::WeightsReorderParams;
    using generic_kernel_params             = KernelSelector::GenericKernelParams;
    using upsampling_params                 = KernelSelector::UpSamplingParams;

    using optional_params                   = KernelSelector::OptionalParams;
    using weights_bias_optional_params      = KernelSelector::WeightsBiasOptionalParams;
    using convolution_optional_params       = KernelSelector::ConvolutionOptionalParams;
    using deconvolution_optional_params     = KernelSelector::DeconvolutionOptionalParams;
    using lrn_optional_params               = KernelSelector::LRNOptionalParams;
    using normalize_optional_params         = KernelSelector::NormalizeOptionalParams;
    using pooling_optional_params           = KernelSelector::PoolingOptionalParams;
    using roi_pooling_optional_params       = KernelSelector::ROIPoolingOptionalParams;
    using fully_connected_optional_params   = KernelSelector::FullyConnectedOptionalParams;
    using activation_optional_params        = KernelSelector::ActivationOptionalParams;
    using softmax_optional_params           = KernelSelector::SoftmaxOptionalParams;
    using region_yolo_optional_params       = KernelSelector::RegionYoloOptionalParams;
    using reorg_yolo_optional_params        = KernelSelector::ReorgYoloOptionalParams;
    using eltwise_optional_params           = KernelSelector::EltwiseOptionalParams;
    using reorder_optional_params           = KernelSelector::ReorderOptionalParams;
    using concatenation_optional_params     = KernelSelector::ConcatenationOptionalParams;
    using upsampling_optional_params        = KernelSelector::UpSamplingOptionalParams;

    using convolution_kernel_selector       = KernelSelector::ConvolutionKernelSelctor;
    using deconvolution_kernel_selector     = KernelSelector::DeconvolutionKernelSelctor;
    using lrn_kernel_selector               = KernelSelector::LRNKernelSelctor;
    using normalize_kernel_selector         = KernelSelector::NormalizeKernelSelctor;
    using pooling_kernel_selector           = KernelSelector::PoolingKernelSelctor;
    using roi_pooling_v1_kernel_selector    = KernelSelector::ROIPoolingKernelSelctor;
    using fully_connected_kernel_selector   = KernelSelector::FullyConnectedKernelSelctor;
    using activation_kernel_selector        = KernelSelector::ActivationKernelSelctor;
    using softmax_kernel_selector           = KernelSelector::SoftmaxKernelSelctor;
    using region_yolo_kernel_selector       = KernelSelector::RegionYoloKernelSelctor;
    using reorg_yolo_kernel_selector        = KernelSelector::ReorgYoloKernelSelctor;
    using eltwise_kernel_selector           = KernelSelector::EltwiseKernelSelctor;
    using reorder_kernel_selector           = KernelSelector::ReorderKernelSelctor;
    using reshape_kernel_selector           = KernelSelector::ReshapeKernelSelctor;
    using permute_kernel_selector           = KernelSelector::PermuteKernelSelctor;
    using concatenation_kernel_selector     = KernelSelector::ConcatenationKernelSelctor;
    using upsampling_kernel_selector        = KernelSelector::UpSamplingKernelSelector;
}

inline kernel_selector::data_type to_data_type(data_types dt)
{
    switch (dt)
    {
    case cldnn::data_types::i8:     return kernel_selector::data_type::INT8;
    case cldnn::data_types::u8:     return kernel_selector::data_type::UINT8;
    case cldnn::data_types::f16:    return kernel_selector::data_type::F16;
    case cldnn::data_types::f32:    return kernel_selector::data_type::F32;
    default:
        assert(0);
        return kernel_selector::data_type::F16;
    }
}

inline data_types from_data_type(kernel_selector::data_type dt)
{
    switch (dt)
    {
    case kernel_selector::data_type::INT8:   return cldnn::data_types::i8;
    case kernel_selector::data_type::UINT8:   return cldnn::data_types::u8;
    case kernel_selector::data_type::F16:    return cldnn::data_types::f16;
    case kernel_selector::data_type::F32:    return cldnn::data_types::f32;
    default:
        assert(0);
        return cldnn::data_types::f16;
    }
}

inline kernel_selector::weights_type to_weights_type(data_types dt)
{
    switch (dt)
    {
    case cldnn::data_types::i8:     return kernel_selector::weights_type::INT8;
    case cldnn::data_types::f16:    return kernel_selector::weights_type::F16;
    case cldnn::data_types::f32:    return kernel_selector::weights_type::F32;
    default:
        assert(0);
        return kernel_selector::weights_type::F16;
    }
}

inline data_types from_weights_type(kernel_selector::weights_type dt)
{
    switch (dt)
    {
    case kernel_selector::weights_type::INT8:   return data_types::i8;
    case kernel_selector::weights_type::F16:    return data_types::f16;
    case kernel_selector::weights_type::F32:    return data_types::f32;
    default:
        assert(0);
        return data_types::f16;;
    }
}

inline kernel_selector::data_layout to_data_layout(format f)
{
    switch (f)
    {
    case format::bfyx:              return kernel_selector::data_layout::bfyx;
    case format::yxfb:              return kernel_selector::data_layout::yxfb;
    case format::byxf:              return kernel_selector::data_layout::byxf;
    case format::fyxb:              return kernel_selector::data_layout::fyxb;
    case format::bs_x_bsv16:        return kernel_selector::data_layout::bs_f_bsv16__af8;
    case format::bs_xs_xsv8_bsv8:   return kernel_selector::data_layout::bs_f_bsv8__af8;
    case format::bs_xs_xsv8_bsv16:  return kernel_selector::data_layout::bs_f_bsv16__af8;
    case format::bf8_xy16:          return kernel_selector::data_layout::bf8_xy16;
    case format::winograd_2x3_s1_data:  return kernel_selector::data_layout::winograd_2x3_s1_data;
//     case format::brfyx:          return kernel_selector::data_layout::brfyx;
    default:
        return kernel_selector::data_layout::bfyx;
    }
}

static inline cldnn::format from_data_layout(kernel_selector::data_layout l)
{
    switch (l)
    {
    case kernel_selector::data_layout::bf:                return cldnn::format::bfyx;
    case kernel_selector::data_layout::fb:                return cldnn::format::fyxb;
    case kernel_selector::data_layout::bfyx:              return cldnn::format::bfyx;
    case kernel_selector::data_layout::yxfb:              return cldnn::format::yxfb;
    case kernel_selector::data_layout::byxf:              return cldnn::format::byxf;
    case kernel_selector::data_layout::fyxb:              return cldnn::format::fyxb;
    case kernel_selector::data_layout::bs_f_bsv8__af8:    return cldnn::format::bs_xs_xsv8_bsv8;
    case kernel_selector::data_layout::bs_f_bsv16__af8:   return cldnn::format::bs_x_bsv16;
    case kernel_selector::data_layout::bf8_xy16:          return cldnn::format::bf8_xy16;
    case kernel_selector::data_layout::brfyx:             return cldnn::format::bfyx;
    case kernel_selector::data_layout::winograd_2x3_s1_data:   return cldnn::format::winograd_2x3_s1_data;
    default:
        return cldnn::format::bfyx;
        break;
    }
}

inline kernel_selector::weights_layout to_weights_layout(format f)
{
    switch (f)
    {
    case format::bfyx:              return kernel_selector::weights_layout::oiyx;
    case format::fyxb:              return kernel_selector::weights_layout::iyxo;
    case format::byxf:              return kernel_selector::weights_layout::oyxi;
    case format::yxfb:              return kernel_selector::weights_layout::yxio;
    case format::os_iyx_osv16:      return kernel_selector::weights_layout::os_iyx_osv16;
    case format::bs_xs_xsv8_bsv8:   return kernel_selector::weights_layout::os_i_osv8__ai8;
    case format::bs_xs_xsv8_bsv16:  return kernel_selector::weights_layout::os_i_osv16__ai8;
    case format::bs_x_bsv16:        return kernel_selector::weights_layout::os_i_osv16;
    case format::image_2d_weights_c4_fyx_b:     return kernel_selector::weights_layout::image_2d_weights_c4_fyx_b;
    case format::image_2d_weights_c1_b_fyx:     return kernel_selector::weights_layout::image_2d_weights_c1_b_fyx;
    case format::winograd_2x3_s1_weights:       return kernel_selector::weights_layout::winograd_2x3_s1_weights;
    case format::winograd_2x3_s1_fused_weights: return kernel_selector::weights_layout::winograd_2x3_s1_fused_weights;
    default:
        return kernel_selector::weights_layout::oi;
    }
}

static inline cldnn::format::type from_weights_layout(kernel_selector::weights_layout l)
{
    switch (l)
    {
    case kernel_selector::weights_layout::oi:
    case kernel_selector::weights_layout::oiyx:               return cldnn::format::bfyx;
    case kernel_selector::weights_layout::oyxi:               return cldnn::format::byxf;
    case kernel_selector::weights_layout::io:
    case kernel_selector::weights_layout::iyxo:               return cldnn::format::fyxb;
    case kernel_selector::weights_layout::yxio:               return cldnn::format::yxfb;
    case kernel_selector::weights_layout::os_iyx_osv16:       return cldnn::format::os_iyx_osv16;
    case kernel_selector::weights_layout::os_i_osv16:         return cldnn::format::bs_x_bsv16;
    case kernel_selector::weights_layout::os_i_osv8__ai8:     return cldnn::format::bs_xs_xsv8_bsv8;
    case kernel_selector::weights_layout::os_i_osv16__ai8:    return cldnn::format::bs_xs_xsv8_bsv16;
    case kernel_selector::weights_layout::image_2d_weights_c4_fyx_b:        return cldnn::format::image_2d_weights_c4_fyx_b;
    case kernel_selector::weights_layout::image_2d_weights_c1_b_fyx:        return cldnn::format::image_2d_weights_c1_b_fyx;
    case kernel_selector::weights_layout::winograd_2x3_s1_weights:          return cldnn::format::winograd_2x3_s1_weights;
    case kernel_selector::weights_layout::winograd_2x3_s1_fused_weights:    return cldnn::format::winograd_2x3_s1_fused_weights;
    default:
        return cldnn::format::bfyx;
    }
}

inline kernel_selector::tuning_mode to_tuning_mode(cldnn::tuning_mode mode)
{
    switch (mode)
    {
    case cldnn::tuning_mode::tuning_disabled:         return kernel_selector::tuning_mode::TUNING_DISABLED;
    case cldnn::tuning_mode::tuning_use_cache:        return kernel_selector::tuning_mode::TUNING_USE_CACHE;
    case cldnn::tuning_mode::tuning_tune_and_cache:   return kernel_selector::tuning_mode::TUNING_TUNE_AND_CACHE;
    default:
        return kernel_selector::tuning_mode::TUNING_DISABLED;
    }
}

inline std::string to_host_version(const cldnn::version_t& version)
{
    std::stringstream ss;
    ss << version.major << "." << version.minor << "." << version.build << "." << version.revision;
    return ss.str();
}

inline kernel_selector::data_tensor convert_data_tensor(const layout& l, uint32_t split = 1, const tensor view_offset = {})
{
    const auto& pad = l.data_padding;
    const auto& vals = l.size.sizes(l.format);
    const auto& add_offsets = view_offset.sizes(l.format);
    const auto& lower_pad = pad.lower_size().sizes(l.format);
    const auto& upper_pad = pad.upper_size().sizes(l.format);
    const auto ks_layout = to_data_layout(l.format);
    kernel_selector::n_dims vec(KernelSelector::Tensor::ChannelsCount(ks_layout));

    size_t pitch = 1;
    size_t offset = 0;

    for (size_t i = 0; i < vec.size(); i++)
    {
        const size_t tensor_index = vec.size() - 1 - i;
        const auto d = vals[tensor_index];
        const auto lp = lower_pad[tensor_index];
        const auto up = upper_pad[tensor_index];

        auto& elm = vec[i];
        elm.v = static_cast<size_t>(d - add_offsets[tensor_index]);
        elm.pitch = pitch;
        elm.pad.before = lp;
        elm.pad.after = up;

        offset += pitch*(add_offsets[tensor_index]);
        pitch *= (d + lp + up);
    }

    const int feature_index = KernelSelector::Tensor::Channelndex(ks_layout, KernelSelector::Tensor::DataChannelName::FEATURE);
    vec[feature_index].v /= split;

    return kernel_selector::data_tensor(
        vec,
        to_data_type(l.data_type),
        ks_layout,
        offset);
}

inline kernel_selector::weights_tensor convert_weights_tensor(const layout& l)
{
    assert(l.format.dimension() == 4);
    const auto& t = l.size.sizes(format::bfyx);
    const auto base_layout = kernel_selector::weights_layout::oiyx;
    const auto ks_type = to_weights_type(l.data_type);
    const auto ks_layout = to_weights_layout(l.format);
    std::vector<size_t> vec(KernelSelector::Tensor::ChannelsCount(base_layout));

    for (size_t i = 0; i < vec.size(); i++)
    {
        const size_t tensor_index = t.size() - 1 - i;
        const auto d = t[tensor_index];
        vec[i] = static_cast<size_t>(d);
    }

    return kernel_selector::weights_tensor(
        vec,
        ks_type,
        base_layout).TransformIgnorePadding(ks_layout);
}

template <typename p_type>
inline void convert_activation_func_params(const p_type primitive, kernel_selector::base_params& params)
{
    const float negative_slope = primitive->activation_negative_slope;
    if (negative_slope)
    {
        params.activationParams.m = negative_slope;
        params.activationFunc = kernel_selector::activation_function::RELU_NEGATIVE_SLOPE;
    }
    else
    {
        params.activationFunc = kernel_selector::activation_function::RELU;
    }
}

inline kernel_selector::activation_function get_kernel_selector_activation_param(cldnn_activation_func activation_func)
{
    switch (activation_func)
    {
    case activation_none:
        return kernel_selector::activation_function::NONE;
    case activation_logistic:
        return kernel_selector::activation_function::LOGISTIC;
    case activation_hyperbolic_tan:
        return kernel_selector::activation_function::HYPERBOLIC_TAN;
    case activation_relu:
        return kernel_selector::activation_function::RELU;
    case activation_relu_negative_slope:
        return kernel_selector::activation_function::RELU_NEGATIVE_SLOPE;
    case activation_clamp:
        return kernel_selector::activation_function::CLAMP;
    case activation_softrelu:
        return kernel_selector::activation_function::SOFTRELU;
    case activation_abs:
        return kernel_selector::activation_function::ABS;
    case activation_linear:
        return kernel_selector::activation_function::LINEAR;
    case activation_square:
        return kernel_selector::activation_function::SQUARE;
    case activation_sqrt:
        return kernel_selector::activation_function::SQRT;
    default:
        throw std::runtime_error("Unknown activation function");
        break;
    }
}

template <typename arg_t>
inline void convert_fused_activation_func_params(const arg_t& arg, kernel_selector::base_params& params)
{
    params.activationParams.m = arg.get_fused_activation_params().a;
    params.activationParams.n = arg.get_fused_activation_params().b;
    params.activationFunc = get_kernel_selector_activation_param(arg.get_fused_activation_func());
}

template <typename p_type>
inline void convert_new_activation_func(const p_type primitive, kernel_selector::base_params& params)
{        
    params.activationFunc = get_kernel_selector_activation_param(primitive->activation_func);
    params.activationParams.m = primitive->additional_params.a;
    params.activationParams.n = primitive->additional_params.b;
}

template <typename params_t, typename arg_t>
inline params_t get_default_params(const arg_t& arg, uint32_t split = 1)
{
    params_t params;

    const auto& context = arg.get_program().get_engine().get_context();
    const auto& engine_info = context->get_engine_info();

    params.engineInfo.bSubGroupSupport      = context->extension_supported("cl_intel_subgroups");
    params.engineInfo.bSubGroupShortSupport = context->extension_supported("cl_intel_subgroups_short");
    params.engineInfo.bFP16Support          = context->extension_supported("cl_khr_fp16");
    params.engineInfo.bFP64Support          = context->extension_supported("cl_khr_fp64");
    params.engineInfo.maxWorkGroupSize      = engine_info.max_work_group_size;
    params.engineInfo.maxLocalMemSize       = engine_info.max_local_mem_size;
    params.engineInfo.deviceId              = engine_info.dev_id;
    params.engineInfo.driverVersion         = engine_info.driver_version;
    params.engineInfo.hostVersion           = to_host_version(cldnn::get_version());
    
    const auto& input_layout    = arg.input().get_output_layout();
    const auto& output_layout   = arg.get_output_layout();

    params.inputs[0] = convert_data_tensor(input_layout, split);
    params.output = convert_data_tensor(output_layout, split);

    params.layerID = arg.id();

    convert_fused_activation_func_params(arg, params);

    return params;
}

template <typename params_t, typename arg_t>
inline params_t get_weights_bias_default_params(const arg_t& arg, uint32_t split = 1)
{
    params_t params = get_default_params<params_t>(arg, split);

    const auto& weights_layout = arg.weights().get_output_layout();
    params.weights = convert_weights_tensor(weights_layout);

    if (arg.bias_term())
    {
        const auto& bias_layout = arg.bias().get_output_layout();
        // bias per output is not supported on cldnn
        params.bias.push_back(convert_data_tensor(bias_layout).FlattenFeatureAndSpatials());
    }

    return params;
}

template <typename optional_params_t>
inline optional_params_t get_default_optional_params(const program_impl& program)
{
    optional_params_t params;
    
    const auto& context = program.get_engine().get_context();

    params.meaningfulKernelsNames       = context->get_configuration().meaningful_kernels_names;
    params.allowStaticInputReordering   = program.get_options().get<build_option_type::optimize_data>()->enabled();
    params.allowInputReordering         = false;
    params.allowOutputReordering        = false;
    
    const auto& tuning_config = program.get_options().get<build_option_type::tuning_config>();
    params.tuningParams.mode = to_tuning_mode(tuning_config->config.mode);
    params.tuningParams.cacheFilePath = tuning_config->config.cache_file_path;

    return params;
}

template <typename optional_params_t>
inline optional_params_t get_default_weights_bias_optional_params(const program_impl& program)
{
    return get_default_optional_params<optional_params_t>(program);
}
