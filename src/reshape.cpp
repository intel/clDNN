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
#include "reshape_inst.h"
#include "primitive_type_base.h"
#include "memory_impl.h"

namespace cldnn
{

primitive_type_id reshape_type_id()
{
    static primitive_type_base<reshape, reshape_inst> instance;
    return &instance;
}

layout reshape_inst::calc_output_layout(reshape_node const& node)
{
    auto input_layout = node.input().get_output_layout();
    input_layout.size = node.get_primitive()->output_shape;
    return input_layout;
}

std::string reshape_inst::to_string(reshape_node const& node)
{
    std::stringstream           primitive_description;
    auto desc = node.get_primitive();
    auto input = node.input();
    primitive_description << "id: " << desc->id << ", type: reshape" <<
        "\n\tinput: " << input.id() << ", count: " << input.get_output_layout().count() << ", size: " << input.get_output_layout().size <<
        "\n\toutput shape: " << desc->output_shape <<
        "\n\toutput padding lower size: " << desc->output_padding.lower_size() <<
        "\n\toutput padding upper size: " << desc->output_padding.upper_size() <<
        "\n\toutput: count: " << node.get_output_layout().count() << ", size: " << node.get_output_layout().size << '\n';

    return primitive_description.str();
}

reshape_inst::typed_primitive_inst(network_impl& network, reshape_node const& node)
    : parent(network, node, false)
{
    auto input_layout = node.input().get_output_layout();
    auto output_layout = node.get_output_layout();
    if (input_layout.data_type != output_layout.data_type)
        throw std::domain_error("Output layout of reshape primitive has different data type than it's input");
    if (input_layout.count() != output_layout.count())
        throw std::domain_error("Output layout of reshape pirmitive changes size of input buffer");

    //if reshape operated in-place, postpone creation of the output until network run,
    //then create new memory object as the reinterpreted output of the previous primitive
    if (!node.is_in_place())
        _output = allocate_output();
    else
        reuse_input();
}

void reshape_inst::on_execute()
{
    if (!node.is_in_place())
        return;

    if (_output && _output->is_the_same_buffer(input_memory()))
        return;

    reuse_input();
}

void reshape_inst::reuse_input()
{
    _output = api_cast(_network.get_engine()->reinterpret_buffer(api_cast(input_memory().get()), node.get_output_layout()));
}

}