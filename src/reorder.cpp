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
#include "error_handler.h"

#include <algorithm>

namespace cldnn
{

primitive_type_id reorder_type_id()
{
    static primitive_type_base<reorder> instance;
    return &instance;
}

layout reorder_inst::calc_output_layout(reorder_node const& node)
{
    auto input_layout = node.input().get_output_layout();
    auto odt = node.get_primitive()->output_data_type;
    auto of = node.get_primitive()->output_format;
    auto op = node.get_primitive()->output_padding;

    if(of == format::bs_xs_xsv8_bsv8 || of == format::bs_xs_xsv8_bsv16 || of == format::bs_x_bsv16)
        return layout(odt, of, input_layout.size.transform(of, 1), op);
    else
        return layout(odt, of, input_layout.size, op);
}

std::string reorder_inst::to_string(reorder_node const& node)
{
    std::stringstream           primitive_description;
    auto desc                   = node.get_primitive();
    auto& input                 = node.input();
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
    : parent(network, node, !node.can_be_optimized())
{
    if (node.can_be_optimized())
        reuse_input();

    auto& input_mem = input_memory();
    auto& output_mem = output_memory();

    CLDNN_ERROR_LESS_THAN(node.id(), "Input dimension size", input_mem.get_layout().size.raw.size(), "ouput dimension size", output_mem.get_layout().size.raw.size(), "Input dimension < output dimension. Reorder primitive woks only with same dimension sizes (reorder) or when input > output (flatten).");
    
    if (!argument.subtract_per_feature.empty())
    {
        CLDNN_ERROR_GREATER_THAN(node.id(), "Input feature dimension size", input_mem.get_layout().size.feature.size(), "value", 1, "Subtracting values work only for formats that have feature dimension == 1");
        CLDNN_ERROR_NOT_EQUAL(node.id(), "Input feature size[0]", static_cast<size_t>(input_mem.get_layout().size.feature[0]), "argument subtract per feature size", argument.subtract_per_feature.size(), "Number of features/channels in input does not match the number of features/channels in values to subtract");
    }
}

void reorder_inst::on_execute()
{
    if (node.can_be_optimized())
        reuse_input();
}

void reorder_inst::reuse_input()
{
    if (!node.can_be_optimized())
        return;

    if (node.requires_reinterpret())
    {
        if (!_output || !_network.get_engine().is_the_same_buffer(output_memory(), input_memory()))
            _output = _network.get_engine().reinterpret_buffer(input_memory(), node.get_output_layout());
    }
    else if (!_output)
        _output = &input_memory();
}

}
