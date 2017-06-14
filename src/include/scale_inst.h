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
#include "api/CPP/scale.hpp"
#include "primitive_inst.h"

namespace cldnn
{

template <>
struct typed_program_node<scale> : public typed_program_node_base<scale>
{
    using parent = typed_program_node_base<scale>;

public:
    using parent::parent;

    auto& input() const { return get_dependency(0); }
    auto& scale_in() const { return get_dependency(1); }
    auto& bias() const { return get_dependency(2); }
};

using scale_node = typed_program_node<scale>;

template <>
class typed_primitive_inst<scale> : public typed_primitive_inst_base<scale>
{
    using parent = typed_primitive_inst_base<scale>;

public:
    static layout calc_output_layout(scale_node const& node);
    static std::string to_string(scale_node const& node);

public:
    typed_primitive_inst(network_impl& network, scale_node const& desc);

    const memory& input_memory() const { return dep_memory(0); }
    const memory& scale_memory() const { return dep_memory(1); }
    const memory& bias_memory() const { return dep_memory(2); }

    bool bias_term() const { return _deps.size() > 2; }
};

using scale_inst = typed_primitive_inst<scale>;

}
