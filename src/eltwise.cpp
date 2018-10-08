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

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "eltwise_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"

namespace cldnn
{
primitive_type_id eltwise_type_id()
{
    static primitive_type_base<eltwise> instance;
    return &instance;
}

layout eltwise_inst::calc_output_layout(eltwise_node const& node)
{
    auto input_node_layout = node.input().get_non_padded_output_layout();
    //list of operations supported for integer types
    if (input_node_layout.data_type == data_types::i8 ||
        input_node_layout.data_type == data_types::i32 ||
        input_node_layout.data_type == data_types::i64)
    {
        auto mode = node.get_primitive()->mode;
        std::vector<eltwise_mode> eltwise_int_modes = { eltwise_mode::sum, eltwise_mode::sub, eltwise_mode::prod, eltwise_mode::div };
        if (std::find(eltwise_int_modes.begin(), eltwise_int_modes.end(), mode) == eltwise_int_modes.end())
            CLDNN_ERROR_MESSAGE(node.id(), "Requested eltwise mode is not supported for integer types.");
    }

    auto eltw = std::static_pointer_cast<const eltwise>((node.get_primitive()));
    if (!eltw->stride.empty())
    {
        // we can safely use only first stride, since we're using first input, and input / stride should give exact same value for every input
        input_node_layout.size.spatial[0] /= eltw->stride[0].spatial[0];
        input_node_layout.size.spatial[1] /= eltw->stride[0].spatial[1];
    }
    return input_node_layout;
}

static inline std::string stringify_vector(const std::vector<float>& v)
{
    std::stringstream s;

    s << "{ ";

    for (size_t i = 0; i < v.size(); ++i)
    {
        s << v.at(i);
        if (i + 1 < v.size()) s << ", ";
    }

    s << " }";

    return s.str();
}

std::string eltwise_inst::to_string(eltwise_node const& node)
{
    auto node_info  = node.desc_to_json();
    auto desc       = node.get_primitive();
    auto activation = desc->with_activation ? " true" : "false";

    std::stringstream primitive_description;
    std::string       str_mode;

    switch(desc->mode)
    {
    case eltwise_mode::sum:
            str_mode = "sum";
            break;
    case eltwise_mode::sub:
            str_mode = "subtract";
            break;
    case eltwise_mode::max:
            str_mode = "max";
            break;
    case eltwise_mode::prod:
            str_mode = "product";
            break;
    case eltwise_mode::div:
            str_mode = "div";
            break;
    case eltwise_mode::min:
            str_mode = "min";
         break;
    case eltwise_mode::pow:
            str_mode = "pow";
            break;
    case eltwise_mode::mod:
            str_mode = "mod";
            break;
    default:
            str_mode = "not supported mode";
            break;
    }

    json_composite eltwise_info;
    for (size_t i = 0; i < node.inputs_count(); i++)
    {
        eltwise_info.add("input_"+std::to_string(i), node.input(i).id());
    }
    eltwise_info.add("mode", str_mode);
    if (desc->mode == eltwise_mode::sum)
    {
        eltwise_info.add("coefficients", stringify_vector(desc->coefficients));
    }
    if (desc->with_activation)
    {
        eltwise_info.add("with activation", activation);
        eltwise_info.add("slope", desc->activation_negative_slope);
    }
    node_info->add("eltwise info", eltwise_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

eltwise_inst::typed_primitive_inst(network_impl& network, eltwise_node const& node)
    :parent(network, node)
{
    auto input_layout = node.input().get_output_layout();
    auto batch_size = input_layout.size.batch[0];
    auto feature_size = input_layout.size.feature[0];

    auto input_batch_size = input_layout.size.batch[0];
    auto input_feature_size = input_layout.size.feature[0];

    if (batch_size != 1)
    {
        CLDNN_ERROR_NOT_EQUAL(node.id(), "Eltwise batch size", batch_size, "input batch size", input_batch_size, "");
    }

    if (feature_size != 1)
    {
        CLDNN_ERROR_NOT_EQUAL(node.id(), "Eltwise feature size", feature_size, "input feature size", input_feature_size, "");
    }

    // check for stride
    auto prim = node.get_primitive();
    if (!prim->stride.empty())
    {
        // number of strides must match number of inputs
        CLDNN_ERROR_NOT_EQUAL(node.id(), "Eltwise inputs count", node.inputs_count(), "Eltwise strides count", prim->stride.size(), "");

        const auto out_x = node.get_output_layout().size.spatial[0];
        const auto out_y = node.get_output_layout().size.spatial[1];
        // check if strides are correctly set. I.e INPUT_SIZE_X / STRIDE_X = OUTPUT_SIZE_X, same for Y dimension
        for (size_t i = 0; i < node.inputs_count(); i++)
        {
            const auto& in_layout = node.input(i).get_output_layout();
            auto stride = prim->stride[i];

            const auto in_x_div_stride_x = in_layout.size.spatial[0] / stride.spatial[0];
            if(in_x_div_stride_x != out_x)
                CLDNN_ERROR_NOT_EQUAL(node.id(), "Eltwise input_x / stride_x", in_x_div_stride_x, "Eltwise output_x", out_x, "");

            const auto in_y_div_stride_y = in_layout.size.spatial[1] / stride.spatial[1];
            if(in_y_div_stride_y != out_y)
                CLDNN_ERROR_NOT_EQUAL(node.id(), "Eltwise inputyx / stride_y", in_y_div_stride_y, "Eltwise output_y", out_y, "");
        }
    }
}
}
