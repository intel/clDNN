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

#if   defined REORDER_MODE_XYZW
inline unsigned int get_output_index(unsigned int x, unsigned int y, unsigned int z, unsigned int w)
#elif defined REORDER_MODE_XYWZ
inline unsigned int get_output_index(unsigned int x, unsigned int y, unsigned int w, unsigned int z)
#elif defined REORDER_MODE_XWYZ
inline unsigned int get_output_index(unsigned int x, unsigned int w, unsigned int y, unsigned int z)
#elif defined REORDER_MODE_WXYZ
inline unsigned int get_output_index(unsigned int w, unsigned int x, unsigned int y, unsigned int z)
#elif defined REORDER_MODE_XZYW
inline unsigned int get_output_index(unsigned int x, unsigned int z, unsigned int y, unsigned int w)
#elif defined REORDER_MODE_ZYXW
inline unsigned int get_output_index(unsigned int z, unsigned int y, unsigned int x, unsigned int w)
#elif defined REORDER_MODE_YXZW
inline unsigned int get_output_index(unsigned int y, unsigned int x, unsigned int z, unsigned int w)
#endif
{ 
   return OUTPUT_OFFSET + w*OUTPUT_BATCH_PITCH + z*OUTPUT_FEATURE_PITCH + y*OUTPUT_Y_PITCH + x;
}

KERNEL(reorder)(
    __global DATA_TYPE* input,
    __global DATA_TYPE* output)
{
    const unsigned x = get_global_id(0);
    const unsigned y = get_global_id(1);
#if INPUT_BATCH_NUM == 1
    const unsigned z = get_global_id(2);
    const unsigned w = 0;
#else
    const unsigned z = get_global_id(2) % INPUT_FEATURE_NUM;
    const unsigned w = get_global_id(2) / INPUT_FEATURE_NUM;
#endif
    
    const unsigned int src_index = INPUT_OFFSET + w*INPUT_BATCH_PITCH + z*INPUT_FEATURE_PITCH + y*INPUT_Y_PITCH;
    output[get_soruce_index(x, y, z, w)] = FUNC_CALL(activation_function)(input[src_index + x], NL_M, NL_N);
}

