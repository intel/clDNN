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

#include "activation_inst.h"
#include "primitive_type_base.h"

namespace cldnn
{
primitive_type_id activation_type_id()
{
    static primitive_type_base<activation> instance;
    return &instance;
}

layout activation_inst::calc_output_layout(activation_node const& node)
{
    return node.input().get_output_layout();
}

std::string activation_inst::to_string(activation_node const& node)
{
    std::stringstream            primitive_description;
    auto desc                   = node.get_primitive();
    auto input                  = node.input();

    primitive_description << "id: " << desc->id << ", type: activation" <<
        "\n\tinput: " << input.id() << ", count: " << input.get_output_layout().count() << ", size: "  << input.get_output_layout().size <<
        "\n\tactivation_func: " << desc->activation_func <<
        "\n\tadditional_params.a: " << desc->additional_params.a <<
        "\n\tadditional_params.b: " << desc->additional_params.b <<
        "\n\tadditional_params input: " << desc->additional_params_input <<
        "\n\toutput padding lower size: " << desc->output_padding.lower_size() <<
        "\n\toutput padding upper size: " << desc->output_padding.upper_size() <<
        "\n\toutput: count: " << node.get_output_layout().count() << ",  size: " << node.get_output_layout().size << '\n';

    return primitive_description.str();
}

activation_inst::typed_primitive_inst(network_impl& network, activation_node const& node)
    : parent(network, node)
{
    auto input_arg  = input_memory().get_layout();
    auto output_arg = output_memory().get_layout();
    
    if (input_arg.size.raw.size() != output_arg.size.raw.size())
        throw std::runtime_error("ReLU input/output number of dimension does not match.");

    if (is_parameterized())
    {
        /// Slope input x dimension should be equal to input feature size (one slope per channel).
        auto slope_input_size = slope_memory().get_layout().size;
        auto input_feature_size = input_memory().get_layout().size.feature[0];

        if (slope_input_size.spatial[0] < input_feature_size)
        {
            throw std::invalid_argument("Dimensions mismatch between input and slope input in Activation layer (slope x size should be equal to input feature size)!");
        }

        // All other dimensions should be 1
        if ((int32_t)slope_input_size.count() != slope_input_size.spatial[0])
        {
            throw std::invalid_argument("Dimensions mismatch of slope input in Activation layer!");
        }
    }
}
}
