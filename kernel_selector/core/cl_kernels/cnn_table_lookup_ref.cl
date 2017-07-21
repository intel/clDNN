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

KERNEL(table_lookup)(
    __global DATA_TYPE* input,
    __global DATA_TYPE* output,
    __global LUT_TYPE* lut)
{
    const unsigned x = get_global_id(0);
    const unsigned y = get_global_id(1);
#if OUTPUT_BATCH_NUM == 1
    const unsigned z = get_global_id(2);
    const unsigned w = 0;
#else
    const unsigned z = get_global_id(2) % OUTPUT_FEATURE_NUM;
    const unsigned w = get_global_id(2) / OUTPUT_FEATURE_NUM;
#endif
    
    const unsigned src_index = w*INPUT_BATCH_PITCH + z*INPUT_FEATURE_PITCH + y*INPUT_Y_PITCH + x + INPUT_OFFSET;
    const unsigned dst_index = w*OUTPUT_BATCH_PITCH + z*OUTPUT_FEATURE_PITCH + y*OUTPUT_Y_PITCH + x + OUTPUT_OFFSET;
    
    unsigned lut_idx = (unsigned)input[src_index];
    
    DATA_TYPE val = (lut_idx < TABLE_SIZE) ? (DATA_TYPE)lut[lut_idx] : 0;
    output[dst_index] = FUNC_CALL(activation_function)(val, NL_M, NL_N);
}
