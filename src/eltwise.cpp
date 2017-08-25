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

namespace cldnn
{
primitive_type_id eltwise_type_id()
{
    static primitive_type_base<eltwise> instance;
    return &instance;
}

layout eltwise_inst::calc_output_layout(eltwise_node const& node)
{
    return node.input().get_output_layout();
}

std::string eltwise_inst::to_string(eltwise_node const& node)
{
    std::stringstream           primitive_description;
    auto desc                   = node.get_primitive();
    auto& input_1               = node.input();
    auto& input_2               = node.input2();
    auto activation             = desc->with_activation ? " true" : "false";
    std::string                 str_mode;
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
    default:
            str_mode = "not supported mode";
            break;
    }

    primitive_description << "id: " << desc->id << ", type: eltwise" << 
        "\n\tinput_1: " << input_1.id() << ", count: " << input_1.get_output_layout().count() << ",  size: " << input_1.get_output_layout().size <<
        "\n\tinput_2: " << input_2.id() << ", count: " << input_2.get_output_layout().count() << ",  size: " << input_2.get_output_layout().size <<
        "\n\tmode: " << str_mode <<
        "\n\twith activation: " << activation << ", slope: " << desc->activation_negative_slope << 
        "\n\toutput padding lower size: " << desc->output_padding.lower_size() <<
        "\n\toutput padding upper size: " << desc->output_padding.upper_size() <<
        "\n\toutput: count: " << node.get_output_layout().count() << ",  size: " << node.get_output_layout().size << '\n';

    return primitive_description.str();
}

eltwise_inst::typed_primitive_inst(network_impl& network, eltwise_node const& node)
    :parent(network, node)
{
    auto input_layout = input_memory(0).get_layout();
    auto input2_layout = input_memory(1).get_layout();

    CLDNN_ERROR_LAYOUT_MISMATCH(node.id(), "input layout", input_layout, "input_2 layout", input2_layout, "Different layouts of eltwise's inputs");
}
}
