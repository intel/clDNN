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
#include "data_inst.h"
#include "primitive_type_base.h"
#include "memory_impl.h"

namespace cldnn
{
primitive_type_id data_type_id()
{
    static primitive_type_base<data, data_inst> instance;
    return &instance;
}

namespace {
    cldnn::memory attach_or_copy_data(network_impl& network, const memory& mem)
    {
        auto engine = network.get_engine();
        auto mem_ref = mem.get();
        auto mem_impl = api_cast(mem_ref);
        if (mem_impl->is_allocated_by(engine))
        {
            return mem;
        }

        memory result(api_cast(engine->allocate_buffer(mem.get_layout())));
        pointer<char> src(mem);
        pointer<char> dst(result);
        std::copy(src.begin(), src.end(), dst.begin());
        return result;
    }
}

std::string data_inst::to_string(data_node const& node)
{
    std::stringstream           primitive_description;
    auto desc                   = node.get_primitive();
    auto count                  = node.get_output_layout().count();

    primitive_description << "id: " << desc->id << ", type: data" <<
        "\n\tcount: "     << count <<", size: " << node.get_output_layout().size <<'\n';

    return primitive_description.str();
}

data_inst::typed_primitive_inst(network_impl& network, data_node const& node)
    : parent(network, node, attach_or_copy_data(network, node.get_primitive()->mem))
{
}

}
