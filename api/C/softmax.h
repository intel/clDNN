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
#ifndef SOFTMAX_H
#define SOFTMAX_H

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

/// @brief Normalizes results so they sum to 1.
/// @details
/// @par Algorithm:
///   b = e^a/sum(N-1; j=0; e^j)
/// @par Where:
///   @li N : number of values to normalize
///   @li b : value after normalization
///   @li a : value before normalization
CLDNN_BEGIN_PRIMITIVE_DESC(softmax)
CLDNN_END_PRIMITIVE_DESC(softmax)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(softmax);

#ifdef __cplusplus
}
#endif

/// @}
/// @}
/// @}
#endif /* SOFTMAX_H */

