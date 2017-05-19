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
#include "../C/softmax.h"
#include "primitive.hpp"

namespace cldnn
{
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Normalizes results so they sum to 1.
/// @details
/// @par Algorithm:
///   b = e^a/sum(N-1; j=0; e^j)
/// @par Where:
///   @li N : number of values to normalize
///   @li b : value after normalization
///   @li a : value before normalization
struct softmax : public primitive_base<softmax, CLDNN_PRIMITIVE_DESC(softmax)>
{
    CLDNN_DECLATE_PRIMITIVE(softmax)

    /// @brief Constructs softmax primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    softmax(
        const primitive_id& id,
        const primitive_id& input,
        const padding& output_padding = padding()
    )
        :primitive_base(id, {input}, output_padding)
    {}

    /// @brief Constructs a copy from C API @CLDNN_PRIMITIVE_DESC{softmax}
    softmax(const dto* dto)
        :primitive_base(dto)
    {}

private:
    void update_dto(dto&) const override {}
};
/// @}
/// @}
/// @}
}