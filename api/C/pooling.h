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
#ifndef POOLING_H
#define POOLING_H

#include "cldnn.h"
/// @addtogroup c_api C API
/// @{
/// @addtogroup c_topology Network Topology
/// @{
/// @addtogroup c_primitives Primitives
/// @{

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Select method for Pooling layer ( @CLDNN_PRIMITIVE_DESC{pooling} ).
typedef enum /*:int32_t*/
{
    /// @brief Maximum-pooling method.
    cldnn_pooling_max,
    /// @brief Average-pooling method.
    cldnn_pooling_average
} cldnn_pooling_mode;

/// @brief Performs "pooling" operation which is a form of non-linear down-sampling.
/// @details Pools the input image by taking the max, average, etc. within regions.
CLDNN_BEGIN_PRIMITIVE_DESC(pooling)
/// @brief Pooling method. See #cldnn_pooling_mode.
int32_t mode;
/// @brief Defines a shift, relative to (0,0) position of the input buffer, where (0,0) point of the pooling window should start calculations.
cldnn_tensor input_offset;
/// @brief Defines shift in input buffer between adjacent calculations of output values.
cldnn_tensor stride;
/// @brief Pooling kernel size.
cldnn_tensor size;
CLDNN_END_PRIMITIVE_DESC(pooling)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(pooling);

#ifdef __cplusplus
}
#endif

/// @}
/// @}
/// @}
#endif /* POOLING_H */

