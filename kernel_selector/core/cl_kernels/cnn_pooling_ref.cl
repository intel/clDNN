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

KERNEL(pooling)(__global DATA_TYPE *src, __global DATA_TYPE *out)
{
    const unsigned int out_x = get_global_id(0); 
    const unsigned int out_y = get_global_id(1);
#if OUTPUT_BATCH_NUM == 1
    const unsigned int outPlane = get_global_id(2);
    const unsigned int outBatch = 0;
#else
    const unsigned int outPlane = get_global_id(2) % OUTPUT_FEATURE_NUM;
    const unsigned int outBatch = get_global_id(2) / OUTPUT_FEATURE_NUM;
#endif
    unsigned int src_width_step = INPUT_Y_PITCH * POOL_STRIDE_Y;
    
    __global DATA_TYPE* src_ptr = src + outBatch*INPUT_BATCH_PITCH + outPlane*INPUT_FEATURE_PITCH + INPUT_OFFSET;


#ifdef MAX_POOLING
    DATA_TYPE res = DATA_TYPE_MIN;
#else
    DATA_TYPE res = 0;
#endif
    uint num_elementes = 0;

    for(unsigned int y = 0; y < POOL_SIZE_Y; ++y)
    {
        for(unsigned int x = 0; x < POOL_SIZE_X; ++x)
        {
            int src_x = out_x*POOL_STRIDE_X + x - POOL_PAD_X;
            int src_y = out_y*POOL_STRIDE_Y + y - POOL_PAD_Y;

            if(src_x >= 0 && src_x < INPUT_SIZE_X && src_y >= 0 && src_y < INPUT_SIZE_Y)
            {
                DATA_TYPE tmpRes = src_ptr[src_y * INPUT_Y_PITCH + src_x];
                #ifdef MAX_POOLING
                    res = fmax(res, tmpRes);
                #else
                    res += tmpRes;
                #endif 
                
                num_elementes++;
            }
        }
    }

    #ifdef MAX_POOLING
        res = num_elementes != 0 ? res : 0;
    #else
        #ifdef DYNAMIC_KERNEL_DIVIDER
            res = res / (DATA_TYPE)max(num_elementes, 1);
        #else
            res = res / (DATA_TYPE)(POOL_SIZE_X * POOL_SIZE_Y);
        #endif
    #endif
    
    unsigned int out_index = out_x + out_y * OUTPUT_Y_PITCH + outPlane*OUTPUT_FEATURE_PITCH + outBatch*OUTPUT_BATCH_PITCH + OUTPUT_OFFSET;
    out[out_index] = FUNC_CALL(activation_function)(res, NL_M, NL_N);
}
