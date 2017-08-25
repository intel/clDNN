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


#include "include/include_all.cl"

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

KERNEL(pooling_gpu_bfyx_block_opt)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{

    const uint x    = (uint)get_global_id(0);
    const uint y    = (uint)get_global_id(1) * POOL_SIZE_Y;
    const uint bf   = (uint)get_global_id(2);
    const uint f    = bf % INPUT0_FEATURE_NUM;
    const uint b    = bf / INPUT0_FEATURE_NUM;
    
    if ((x >= OUTPUT_SIZE_X) || (y >= OUTPUT_SIZE_Y))
        return;

    const int offset_x = (int)x*STRIDE_SIZE_X - PADDING_SIZE_X;
    const int offset_y = (int)y*STRIDE_SIZE_Y - PADDING_SIZE_Y;
    
    UNIT_TYPE result = UNIT_INIT_VAL;
    
    uint input_idx = GET_DATA_INDEX(INPUT0, b, f, offset_y, offset_x);

    UNIT_TYPE max_x[BLOCK_SIZE_Y];
    UNIT_TYPE out[POOL_SIZE_Y];

    for(uint i = 0; i < BLOCK_SIZE_Y; i++)
    {
        max_x[i] = UNIT_INIT_VAL;
    }

    // we do max in "x" dimension
    for(uint j = 0; j < BLOCK_SIZE_Y; j++)
    {
        for(uint i = 0; i < POOL_SIZE_X; i++)
        {
            max_x[j] = FUNC_CALL(apply_pooling)(max_x[j], input[input_idx]);
            input_idx += INPUT0_X_PITCH;
        }
        input_idx += (INPUT0_Y_PITCH - POOL_SIZE_X*INPUT0_X_PITCH);
    }

    for(uint i = 0; i < POOL_SIZE_Y; i++)
    {
        out[i] = max_x[i * STRIDE_SIZE_Y];
    }

    // now we do max in "y" dimension
    for(uint i = 0; i < POOL_SIZE_Y; i++)
    {
        for(uint j = 1; j < POOL_SIZE_Y; j++)
        {
            out[i] = FUNC_CALL(apply_pooling)(out[i], max_x[j + i * STRIDE_SIZE_Y]);
        }
    }

    uint output_pos = GET_DATA_INDEX(OUTPUT, b, f, y, x);

    for(uint i = 0; i < POOL_SIZE_Y; i++)
    {
        if((y + i) < OUTPUT_SIZE_Y)
        {
#if defined AVG_POOLING
            out[i] /= (UNIT_TYPE)(POOL_SIZE_Y * POOL_SIZE_X);
#endif
            output[output_pos] = ACTIVATION(out[i], NL_M ,NL_N);
            output_pos += OUTPUT_Y_PITCH;
        }
    }
}

#undef UNIT_INIT_VAL