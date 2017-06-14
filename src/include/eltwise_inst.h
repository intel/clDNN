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
#include "api/CPP/eltwise.hpp"
#include "primitive_inst.h"
#include <memory>
#include "topology_impl.h"

namespace cldnn
{
template <>
struct typed_program_node<eltwise> : public typed_program_node_base<eltwise>
{
    using parent = typed_program_node_base<eltwise>;

public:
    using parent::parent;

    auto& input() const { return get_dependency(0); }
    auto& input2() const { return get_dependency(1); }
};

using eltwise_node = typed_program_node<eltwise>;

template <>
class typed_primitive_inst<eltwise> : public typed_primitive_inst_base<eltwise>
{
    using parent = typed_primitive_inst_base<eltwise>;

public:
    static layout calc_output_layout(eltwise_node const& node);
    static std::string to_string(eltwise_node const& node);

public:
    typed_primitive_inst(network_impl& network, eltwise_node const& node);

    const memory& input_memory() const { return dep_memory(0); }
    const memory& input2_memory() const { return dep_memory(1); }
};

using eltwise_inst = typed_primitive_inst<eltwise>;

}
