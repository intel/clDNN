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

#include "include/cnn_common.cl"

// TODO: move it from layout based to memory based
KERNEL(eltwise)(
    __global DATA_TYPE* input0,
    __global DATA_TYPE* input1,
    __global DATA_TYPE* output)
{
    const unsigned x = get_global_id(0);
    const unsigned y = get_global_id(1);
#if OUTPUT_BATCH_NUM == 1
    const unsigned feature = get_global_id(2);
    const unsigned batch = 0;
#else
    const unsigned feature = get_global_id(2) % OUTPUT_FEATURE_NUM;
    const unsigned batch = get_global_id(2) / OUTPUT_FEATURE_NUM;
#endif

    const unsigned src_index0 = batch*INPUT_BATCH_PITCH + feature*INPUT_FEATURE_PITCH + y*INPUT_Y_PITCH + x*INPUT_X_PITCH + INPUT_OFFSET;
    const unsigned src_index1 = batch*INPUT_BATCH_PITCH1 + feature*INPUT_SLICE_PITCH1 + y*INPUT_ROW_PITCH1 + x*INPUT_X_PITCH1 + INPUT_OFFSET1;
    const unsigned dst_index = batch*OUTPUT_BATCH_PITCH + feature*OUTPUT_FEATURE_PITCH + y*OUTPUT_Y_PITCH + x + OUTPUT_OFFSET;

#if   defined ELTWISE_MODE_ADD
    DATA_TYPE res = input0[src_index0] + input1[src_index1];
#elif defined ELTWISE_MODE_SUB
    DATA_TYPE res = input0[src_index0] - input1[src_index1];
#elif defined ELTWISE_MODE_MUL
    DATA_TYPE res = input0[src_index0] * input1[src_index1];
#elif defined ELTWISE_MODE_DIV
    DATA_TYPE res = input0[src_index0] / input1[src_index1];
#elif defined ELTWISE_MODE_MAX
    DATA_TYPE res = fmax(input0[src_index0], input1[src_index1]);
#endif

#if   defined ELTWISE_SCALAR_MODE_ADD
    res = res + (DATA_TYPE)SCALAR;
#elif defined ELTWISE_SCALAR_MODE_SUB
    res = res - (DATA_TYPE)SCALAR;
#elif defined ELTWISE_SCALAR_MODE_MUL
    res = res * (DATA_TYPE)SCALAR;
#elif defined ELTWISE_SCALAR_MODE_DIV
    res = res / (DATA_TYPE)SCALAR;
#elif defined ELTWISE_SCALAR_MODE_MAX
    res = fmax(res, (DATA_TYPE)SCALAR);
#endif
    output[dst_index] = FUNC_CALL(activation_function)(res, NL_M, NL_N);
}
