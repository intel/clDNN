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
#pragma once
#include "api/CPP/mutable_data.hpp"
#include "api_extension/CPP/fused_conv_eltwise.hpp"
#include "primitive_inst.h"
#include "program_impl.h"

namespace cldnn
{

template <>
struct typed_program_node<mutable_data> : public typed_program_node_base<mutable_data>
{
    using parent = typed_program_node_base<mutable_data>;

    typed_program_node(const std::shared_ptr<mutable_data> prim, program_impl& prog);

    memory_impl& get_attached_memory() const { return *mem; }
    memory_impl::ptr get_attached_memory_ptr() const { return mem; }
    void attach_memory(memory_impl& new_mem, bool invalidate_users_if_changed = true);

    program_node& input(size_t idx = 0) const { return get_dependency(idx); }

private:
    memory_impl::ptr mem;
    CLDNN_SERIALIZATION_MEMBERS(
        auto mem_layout = &mem->get_layout();
        ar & CLDNN_SERIALIZATION_BASE_OBJECT_NVP(parent) & CLDNN_SERIALIZATION_NVP(mem_layout);
        if (!Archive::is_saving::value)
        {
            if (this->get_dependency(0).is_type<mutable_data>() && (*this->get_users().begin())->is_type<fused_conv_eltwise>()) // Solution for conv_eltw_read_write_opt
                mem = &this->get_dependency(0).as<mutable_data>().get_attached_memory();
            else if (this->get_dependency(0).is_type<fused_conv_eltwise>() && this->get_dependency(0).get_dependency(1).is_type<mutable_data>())
                mem = &this->get_dependency(0).get_dependency(1).as<mutable_data>().get_attached_memory();
            else
                mem = this->get_program().get_engine().allocate_memory(*mem_layout);
        }
    )

    void fill_memory();
    void fill_memory_xavier();
    void fill_memory_constant(float value);
};

using mutable_data_node = typed_program_node<mutable_data>;

template <>
class typed_primitive_inst<mutable_data> : public typed_primitive_inst_base<mutable_data>
{
    using parent = typed_primitive_inst_base<mutable_data>;

public:
    static layout calc_output_layout(mutable_data_node const& node)
    {
        return node.get_attached_memory().get_layout();
    }
    static std::string to_string(mutable_data_node const& node);

public:
    typed_primitive_inst(network_impl& network, mutable_data_node const& node);
};

using mutable_data_inst = typed_primitive_inst<mutable_data>;

}
CLDNN_SERIALIZATION_TYPED_PROGRAM_NODE_CLASS(mutable_data)
