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
#include "api/CPP/batch_norm.hpp"
#include "primitive_inst.h"

namespace cldnn
{

template <>
struct typed_program_node<batch_norm> : public typed_program_node_base<batch_norm>
{
    using parent = typed_program_node_base<batch_norm>;

public:
    using parent::parent;

    auto& input() const { return get_dependency(0); }
    auto& mean() const { return get_dependency(1); }
    auto& variance() const { return get_dependency(2); }
};

using batch_norm_node = typed_program_node<batch_norm>;

template <>
class typed_primitive_inst<batch_norm> : public typed_primitive_inst_base<batch_norm>
{
    using parent = typed_primitive_inst_base<batch_norm>;

public:
    static layout calc_output_layout(batch_norm_node const& node);
    static std::string to_string(batch_norm_node const& node);

public:
    typed_primitive_inst(network_impl& network, batch_norm_node const& node);

    const memory& input_memory() const { return dep_memory(0); }
    const memory& mean_memory() const { return dep_memory(1); }
    const memory& variance_memory() const { return dep_memory(2); }
};

using batch_norm_inst = typed_primitive_inst<batch_norm>;

}
