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
#include "primitive_type.h"
#include "network_impl.h"
#include "engine_impl.h"
#include <memory>

namespace cldnn
{
template<class PType, class PType_Inst>
struct primitive_type_base : ::cldnn_primitive_type
{
    static_assert(std::is_base_of<primitive, PType>::value, "Primitive type passed to primitive_type_base should derive from cldnn::primitive");

    std::shared_ptr<primitive> from_dto(const CLDNN_PRIMITIVE_DESC(primitive)* dto) const override
    {
        if (dto->type != this)
            throw std::invalid_argument("primitive_type_base::from_dto: primitive type mismatch");

        return std::make_shared<PType>(as_dto<PType>(dto));
    }

    std::shared_ptr<cldnn::primitive_inst> create_instance(network_impl& network, const cldnn::program_node& node) const override
    {
        if (node.get_primitive()->type != this)
            throw std::invalid_argument("primitive_type_base::create_instance: primitive type mismatch");

        return std::make_shared<PType_Inst>(network, node);
    }

    std::unique_ptr<primitive_impl> choose_impl(engine_impl& engine, const cldnn::program_node& node) const override
    {
        if (node.get_primitive()->type != this)
            throw std::invalid_argument("primitive_type_base::choose_impl: primitive type mismatch");

        return engine.create_primitive_impl(node.as<PType>());
    }

    cldnn::layout calc_output_layout(const cldnn::program_node& node) const override
    {
        if (node.get_primitive()->type != this)
            throw std::invalid_argument("primitive_type_base::calc_output_layout: primitive type mismatch");

        return PType_Inst::calc_output_layout(node);
    }

    std::string to_string(const cldnn::program_node& node) const
    {
        if (node.get_primitive()->type != this)
            throw std::invalid_argument("primitive_type_base::to_string: primitive type mismatch");

        return PType_Inst::to_string(node);
    }

};
}
