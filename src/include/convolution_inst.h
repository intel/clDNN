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
#include "api/CPP/convolution.hpp"
#include "primitive_inst.h"

#include <memory>

namespace cldnn
{

template <>
struct typed_program_node<convolution> : public typed_program_node_base<convolution>
{
    using parent = typed_program_node_base<convolution>;

public:
    using parent::parent;

    auto& input() const { return get_dependency(0); }

    auto& weights(size_t idx) const
    {
        if (idx >= typed_desc()->weights.size())
            throw std::range_error("weights offset too big");

        return get_dependency(1 + idx);
    }

    auto& bias(size_t idx) const 
    { 
        if (idx >= typed_desc()->bias.size()) 
            throw std::range_error("bias offset too big");

        return get_dependency(1 + typed_desc()->weights.size() + idx);
    }

    bool bias_term() const
    {
        if (get_primitive()->bias.size() != 0)
            return true;
        else
            return false;
    }
};

using convolution_node = typed_program_node<convolution>;

template <>
class typed_primitive_inst<convolution> : public typed_primitive_inst_base<convolution>
{
    using parent = typed_primitive_inst_base<convolution>;

public:
    static layout calc_output_layout(convolution_node const& node);
    static std::string to_string(convolution_node const& node);

public:
    typed_primitive_inst(network_impl& network, convolution_node const& node);

    const memory& input_memory() const { return dep_memory(0); }

    const memory& weights_memory(size_t index) const
    {
        if (index >= argument.weights.size())
            throw std::range_error("weights offset too big");
        
        return dep_memory(1 + index);
    }

    const memory& bias_memory(size_t index) const
    { 
        if (index >= argument.bias.size()) 
            throw std::range_error("bias offset too big");

        return dep_memory(1 + argument.weights.size() + index);
    }

    bool bias_term() const
    {
        if (argument.bias.size() != 0)
            return true;
        else
            return false;
    }
};

using convolution_inst = typed_primitive_inst<convolution>;

}
