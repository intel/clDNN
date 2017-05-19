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
#include "permute_inst.h"
#include "primitive_type_base.h"

#include <algorithm>

namespace cldnn
{

primitive_type_id permute_type_id()
{
    static primitive_type_base<permute, permute_inst> instance;
    return &instance;
}

layout permute_inst::calc_output_layout(permute_node const& node)
{
    auto input_layout = node.input().get_output_layout();
    auto permute_order = node.get_primitive()->permute_order;
    auto input_size = tensor(input_layout.size.raw[permute_order[0]], input_layout.size.raw[permute_order[1]],
        input_layout.size.raw[permute_order[2]], input_layout.size.raw[permute_order[3]]);
    auto op = node.get_primitive()->output_padding;

    return layout(input_layout.data_type, input_layout.format, input_size, op);
}

std::string permute_inst::to_string(permute_node const& node)
{
    std::stringstream           primitive_description;
    auto desc                   = node.get_primitive();
    auto input                  = node.input();
    auto permute_order          = desc->permute_order;
    std::stringstream           ss_permute_order;
    for (size_t i = 0; i < permute_order.size(); ++i)
    {
        ss_permute_order << permute_order.at(i);
        i != (permute_order.size() - 1) ? ss_permute_order << ", " : ss_permute_order << "";
    }

    primitive_description << "id: " << desc->id << ", type: permute" 
        "\n\tinput: " << input.id() << ", count: " << input.get_output_layout().count() << ", size: " << input.get_output_layout().size <<
        "\n\tpermute order: "  << ss_permute_order.str() <<
        "\n\toutput padding lower size: " << desc->output_padding.lower_size() <<
        "\n\toutput padding upper size: " << desc->output_padding.upper_size() << '\n';

    return primitive_description.str();
}

permute_inst::typed_primitive_inst(network_impl& network, permute_node const& node)
    : parent(network, node)
{
    auto permute_order = argument.permute_order;

    if (permute_order.size() != 4)
        throw std::runtime_error("Permute order size needs to be 4.");

    std::vector<uint16_t> required_order_values = { 0, 1, 2, 3 };
    auto required_order_values_size = required_order_values.size();

    for (decltype(required_order_values_size) i = 0; i < required_order_values_size; i++)
    {
        if (!(std::find(permute_order.begin(), permute_order.end(), required_order_values[i]) != permute_order.end()))
            throw std::runtime_error("Permute order does not contain all of required values.");
    }

    if (node.has_padded_dependency())
    {
        throw std::runtime_error("Permute with input which contains padding is NOT IMPLEMENTED yet!");
    }
}
}
