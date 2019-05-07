/*
// Copyright (c) 2017 Intel Corporation
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
#include "pooling.hpp"
#include "../C/roi_pooling.h"
#include "primitive.hpp"


namespace cldnn
{
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

struct roi_pooling : public primitive_base<roi_pooling, CLDNN_PRIMITIVE_DESC(roi_pooling)>
{
    CLDNN_DECLARE_PRIMITIVE(roi_pooling)

    roi_pooling(
        const primitive_id& id,
        const primitive_id& input_data,
        const primitive_id& input_rois,
        pooling_mode mode,
        int pooled_width,
        int pooled_height,
        float spatial_scale,
        int group_sz = 0,
        const padding& output_padding = padding()
        )
        : primitive_base(id, {input_data, input_rois}, output_padding)
        , mode(mode)
        , pooled_width(pooled_width)
        , pooled_height(pooled_height)
        , spatial_scale(spatial_scale)
        , group_sz(group_sz)
    {}

    roi_pooling(const dto* dto)
        : primitive_base(dto)
        , mode(static_cast<pooling_mode>(dto->mode))
        , pooled_width(dto->pooled_width)
        , pooled_height(dto->pooled_height)
        , spatial_scale(dto->spatial_scale)
        , group_sz(dto->group_sz)
    {}

    pooling_mode mode;
    int pooled_width;
    int pooled_height;
    float spatial_scale;
    int group_sz;

protected:
    void update_dto(dto& dto) const override
    {
        dto.mode = static_cast<int32_t>(mode);
        dto.pooled_width = pooled_width;
        dto.pooled_height = pooled_height;
        dto.spatial_scale = spatial_scale;
        dto.group_sz = group_sz;
    }
private:
    roi_pooling() : primitive_base() {} // Constructor necessary for serialization process
    CLDNN_SERIALIZATION_MEMBERS(
        ar & CLDNN_SERIALIZATION_BASE_OBJECT_NVP_PRIMITIVE_BASE(roi_pooling) & CLDNN_SERIALIZATION_NVP(mode) & CLDNN_SERIALIZATION_NVP(pooled_width)
           & CLDNN_SERIALIZATION_NVP(pooled_height) & CLDNN_SERIALIZATION_NVP(spatial_scale) & CLDNN_SERIALIZATION_NVP(group_sz);
    )
};

/// @}
/// @}
/// @}
}
CLDNN_SERIALIZATION_EXPORT_NODE_KEY(roi_pooling)
