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
#include "input_layout_inst.h"
#include "primitive_type_base.h"
#include "memory_impl.h"

namespace cldnn
{
primitive_type_id input_layout_type_id()
{
    static primitive_type_base<input_layout, input_layout_inst> instance;
    return &instance;
}

input_layout_inst::typed_primitive_inst(network_impl& network, input_layout_node const& node)
    : parent(network, node)
{
    _has_valid_input = false; //by default input for 'input_layout' is invalid as long as user doesn't call set_data
}

void input_layout_inst::set_data(memory_impl* mem)
{
    if (mem->get_layout() != output_memory().get_layout())
        throw std::invalid_argument("data layout does not match");

    if (mem->is_allocated_by(get_network().get_engine()))
    {
        _output = memory(api_cast(mem), true);
    }
    else
    {
        pointer<char> src(memory(api_cast(mem), true));
        pointer<char> dst(output_memory());
        std::copy(src.begin(), src.end(), dst.begin());
    }

    _has_valid_input = true;
    _output_changed = true;
}

std::string input_layout_inst::to_string(input_layout_node const& node)
{
    std::stringstream   primitive_description;
    auto desc           = node.get_primitive();
    auto count          = node.get_output_layout().count();

    primitive_description << "id: " << desc->id << ", type: input" << 
        "\n\tcount: " << count << ", size: " << node.get_output_layout().size << '\n';

    return primitive_description.str();
}

}
