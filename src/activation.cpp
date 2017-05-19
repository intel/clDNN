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
    static primitive_type_base<activation, activation_inst> instance;
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
        "\n\tslope: " << desc->negative_slope <<
        "\n\toutput padding lower size: " << desc->output_padding.lower_size() <<
        "\n\toutput padding upper size: " << desc->output_padding.upper_size() <<
        "\n\toutput: count: " << node.get_output_layout().count() << ",  size: " << node.get_output_layout().size << '\n';

    return primitive_description.str();
}

activation_inst::typed_primitive_inst(network_impl& network, activation_node const& node)
    :parent(network, node)
{
    auto input_arg  = input_memory().get_layout();
    auto output_arg = output_memory().get_layout();
    
    if (input_arg.size.raw.size() != output_arg.size.raw.size())
        throw std::runtime_error("ReLU input/output number of dimension does not match.");
}
}
