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
#include "reorder_inst.h"
#include "primitive_type_base.h"

#include <algorithm>

namespace cldnn
{

primitive_type_id reorder_type_id()
{
    static primitive_type_base<reorder, reorder_inst> instance;
    return &instance;
}

layout reorder_inst::calc_output_layout(reorder_node const& node)
{
    auto input_layout = node.input().get_output_layout();
    auto odt = node.get_primitive()->output_data_type;
    auto of = node.get_primitive()->output_format;
    auto op = node.get_primitive()->output_padding;

    if(of == format::bs_xs_xsv8_bsv8 || of == format::bs_x_bsv16)
        return layout(odt, of, input_layout.size.transform(of, 1), op);
    else
        return layout(odt, of, input_layout.size, op);
}

std::string reorder_inst::to_string(reorder_node const& node)
{
    std::stringstream           primitive_description;
    auto desc                   = node.get_primitive();
    auto input                  = node.input();
    auto output_layout_data     = desc->output_data_type == data_types::f16 ? "f16" : "f32";
    auto mean = desc->mean;

    primitive_description << "id: " << desc->id << ", type: reorder" 
        "\n\tinput: " << input.id() << ", count: " << input.get_output_layout().count() << ", size: " << input.get_output_layout().size <<
        "\n\tmean: "  << mean <<
        "\n\toutput padding lower size: " << desc->output_padding.lower_size() <<
        "\n\toutput padding upper size: " << desc->output_padding.upper_size() <<
        "\n\toutput: data_type:" << output_layout_data <<", count: " << node.get_output_layout().count() << ",  size: " << node.get_output_layout().size << '\n';

    return primitive_description.str();
}

reorder_inst::typed_primitive_inst(network_impl& network, reorder_node const& node)
    : parent(network, node)
{
    auto& input_mem = input_memory();
    auto& output_mem = output_memory();

    if (input_mem.get_layout().size.raw.size() < output_mem.get_layout().size.raw.size())
        throw std::runtime_error("Input dimension < output dimension. Reorder primitive woks only with same dimension sizes (reorder) or when input > output (flatten).");

    if (!argument.subtract_per_feature.empty())
    {
        if (input_mem.get_layout().size.feature.size() > 1)
        {
            throw std::runtime_error("Subtracting values work only for formats that have feature dimension == 1");
        }
        if (static_cast<size_t>(input_mem.get_layout().size.feature[0]) != argument.subtract_per_feature.size())
            throw std::runtime_error("Number of features/channels in input does not match the number of features/channels in values to subtract");
    }
    if (node.has_padded_dependency())
    {
        throw std::runtime_error("Reorder with input which contains padding is NOT IMPLEMENTED yet!");
    }
}
}
