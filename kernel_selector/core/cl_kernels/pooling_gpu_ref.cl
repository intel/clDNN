// Copyright (c) 2016-2017 Intel Corporation
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


#include "include/common.cl"

#if   defined MAX_POOLING
    #define UNIT_INIT_VAL UNIT_VAL_MIN
#elif defined AVG_POOLING
    #define UNIT_INIT_VAL UNIT_VAL_ZERO
#else
#error
#endif


inline UNIT_TYPE FUNC(apply_pooling)(UNIT_TYPE tmp, UNIT_TYPE in)
{
#if   defined MAX_POOLING
    return max(tmp, in);
#elif defined AVG_POOLING
    return tmp + in;
#endif
}

KERNEL(pooling_gpu)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
#if   defined OUTPUT_LAYOUT_BFYX
    const uint x    = (uint)get_global_id(0);
    const uint y    = (uint)get_global_id(1);
    const uint bf   = (uint)get_global_id(2);
    const uint f    = bf % INPUT_FEATURE_NUM;
    const uint b    = bf / INPUT_FEATURE_NUM;
    
    if (x >= OUTPUT_SIZE_X)
    {
        return;
    }
#elif defined OUTPUT_LAYOUT_YXFB
    const uint x    = (uint)get_global_id(1);
    const uint y    = (uint)get_global_id(2);
    const uint bf   = (uint)get_global_id(0);
    const uint f    = bf / INPUT_BATCH_NUM;
    const uint b    = bf % INPUT_BATCH_NUM;
#endif

    const int offset_x = (int)x*STRIDE_SIZE_X - PADDING_SIZE_X;
    const int offset_y = (int)y*STRIDE_SIZE_Y - PADDING_SIZE_Y;
    
    UNIT_TYPE result = UNIT_INIT_VAL;
    
#ifdef CHECK_BOUNDRY
    if (offset_x + WINDOW_SIZE_X < 0 || offset_x >= INPUT_SIZE_X ||
        offset_y + WINDOW_SIZE_Y < 0 || offset_y >= INPUT_SIZE_Y)
    {
        return;
    }

#ifdef DYNAMIC_KERNEL_DIVIDER
    uint num_elementes = 0;
#endif
    
    const uint batch_and_feature_offset = INPUT_OFFSET + b*INPUT_BATCH_PITCH + f*INPUT_FEATURE_PITCH;
    for(uint j = 0; j < WINDOW_SIZE_Y; j++)
    {
        int input_offset_y = offset_y + j;
        bool zero_y = input_offset_y >= INPUT_SIZE_Y || input_offset_y < 0;
        if(!zero_y)
        {
            for(uint i = 0; i < WINDOW_SIZE_X; i++)
            {
                int input_offset_x = offset_x + i;
                bool zero = input_offset_x >= INPUT_SIZE_X || input_offset_x < 0;
                if(!zero)
                {
                    const uint input_idx = batch_and_feature_offset + input_offset_y*INPUT_Y_PITCH + input_offset_x*INPUT_X_PITCH;
                    result = FUNC_CALL(apply_pooling)(result, input[input_idx]);
                    
#ifdef DYNAMIC_KERNEL_DIVIDER
                    num_elementes++;
#endif
                }
            }
        }
    }
#else
    uint input_idx = INPUT_OFFSET + b*INPUT_BATCH_PITCH + f*INPUT_FEATURE_PITCH + offset_y*INPUT_Y_PITCH + offset_x*INPUT_X_PITCH;

    for(uint j = 0; j < WINDOW_SIZE_Y; j++)
    {
        for(uint i = 0; i < WINDOW_SIZE_X; i++)
        {
            result = FUNC_CALL(apply_pooling)(result, input[input_idx]);
            input_idx += INPUT_X_PITCH;
        }
        input_idx += (INPUT_Y_PITCH - WINDOW_SIZE_X*INPUT_X_PITCH);
    }
    
#ifdef DYNAMIC_KERNEL_DIVIDER
    const uint num_elementes = WINDOW_SIZE_X*WINDOW_SIZE_Y;
#endif
#endif

#if defined AVG_POOLING
    #ifdef DYNAMIC_KERNEL_DIVIDER
        result /= (UNIT_TYPE)max(num_elementes, (uint)1);
    #else
        result /= (UNIT_TYPE)(WINDOW_SIZE_Y * WINDOW_SIZE_X);
    #endif
#endif

    const uint output_pos = OUTPUT_OFFSET + b*OUTPUT_BATCH_PITCH + f*OUTPUT_FEATURE_PITCH + y*OUTPUT_Y_PITCH + x*OUTPUT_X_PITCH;
    output[output_pos] = result;
}

#undef UNIT_INIT_VAL