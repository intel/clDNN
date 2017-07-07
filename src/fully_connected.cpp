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
#include "fully_connected_inst.h"
#include "primitive_type_base.h"

namespace cldnn
{
primitive_type_id fully_connected_type_id()
{
    static primitive_type_base<fully_connected> instance;
    return &instance;
}

namespace
{
bool is_batch_after_spatial(const std::string order)
{
    bool spatial_found = false;
    bool batch_found = false;
    for (auto c : order)
    {
        switch (c)
        {
        case 'b':
        case 'n':
            batch_found = true;
            if (spatial_found)
                return true;
            /* fallthru */
        case 'x':
        case 'y':
        case 'z':
        case 'w':
        case 's':
            spatial_found = true;
            if (batch_found)
                return false;
        default: break;
        }
    }
    return false;
}
}

layout fully_connected_inst::calc_output_layout(fully_connected_node const& node)
{
    auto desc = node.get_primitive();
    
    auto input_layout = node.input().get_output_layout();
    auto weights_layout = node.weights().get_output_layout();

    if(is_batch_after_spatial(input_layout.format.order()) || 
        (input_layout.format == format::bfyx &&                //this condition tests whether our input is batch>1 in bfyx format, if yes there will be
        input_layout.size.batch[0] > 1))                            //extra reorder between input and this fc from bfyx to yxfb format (so "is_batch_after_spatial" should return true)
    {
        auto result = layout(input_layout.data_type, format::yxfb, tensor(input_layout.size.batch[0], 1, weights_layout.size.batch[0], 1));
        return result;
    }
    else
    {
        auto result = layout(input_layout.data_type, format::bfyx, tensor(input_layout.size.batch[0], 1, weights_layout.size.batch[0], 1));
        return result;
    }
}

std::string fully_connected_inst::to_string(fully_connected_node const& node)
{
    std::stringstream           primitive_description;
    auto desc                   = node.get_primitive();
    auto input                  = node.input();
    auto weights_id             = desc->weights;
    auto weights_count          = node.weights().get_output_layout().count();
    auto bias_id                = desc->bias != "" ? desc->bias : "no bias";
    auto bias_count             = desc->bias != "" ? node.bias().get_output_layout().count() : 0;
    auto activation             = desc->with_activation ? " true" : "false";

    primitive_description << "id: " << desc->id << ", type: fully connected" <<
        "\n\tinput: " << input.id() << ", count: " << input.get_output_layout().count() << ", size: " << input.get_output_layout().size <<
        "\n\tweights id: "<< weights_id <<", count: " << weights_count << ", bias id: "<< bias_id <<",count: " << bias_count <<
        "\n\twith activation: " << activation <<
        "\n\toutput padding lower size: " << desc->output_padding.lower_size() <<
        "\n\toutput padding upper size: " << desc->output_padding.upper_size() <<
        "\n\toutput: count: " << node.get_output_layout().count() << ",  size: " << node.get_output_layout().size << '\n';

    return primitive_description.str();
}

fully_connected_inst::typed_primitive_inst(network_impl& network, fully_connected_node const& node)
    :parent(network, node)
{
    auto input_size = input_memory().get_layout();
    auto output_size = output_memory().get_layout();

    if(input_size.format != format::yxfb
        && input_size.format != format::bfyx //special batch1 case
        && (input_size.size.raw.size() != output_size.size.raw.size()) )
    {
        throw std::invalid_argument("Fully connected input/output number of dimension does not match.");
    }
}
}
