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
#include "../C/activation.h"
#include "primitive.hpp"

namespace cldnn
{
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Activiation using rectified linear unit.
struct activation : public primitive_base<activation, CLDNN_PRIMITIVE_DESC(activation)>
{
    CLDNN_DECLATE_PRIMITIVE(activation)

    /// @brief Constructs activation primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param slope Relu activation slope.
    activation(
        const primitive_id& id,
        const primitive_id& input,
        float slope,
        const padding& output_padding = padding()
        )
        : primitive_base(id, {input}, output_padding)
        , negative_slope(slope)
    {
    }

    /// @brief Constructs a copy from basic C API @CLDNN_PRIMITIVE_DESC{activation}
    activation(const dto* dto)
        : primitive_base(dto)
        , negative_slope(dto->negative_slope)
    {
    }

    /// @brief Relu activation slope.
    float negative_slope;

protected:
    void update_dto(dto& dto) const override
    {
        dto.negative_slope = negative_slope;
    }
};
/// @}
/// @}
/// @}
}