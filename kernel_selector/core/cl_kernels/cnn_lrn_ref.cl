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

inline void FUNC(internal_normalize)(__global const DATA_TYPE* input, int input_index, __global DATA_TYPE* output, int output_index, COUNTER_TYPE sum)
{
    // TODO BDW Compiler Bug - we are using (float) because of compiler bug that convert it into (int) instead of (half)
    DATA_TYPE base = (DATA_TYPE)NORM_K + (DATA_TYPE)((COUNTER_TYPE)ALPHA*sum * (float)NUM_ELEMENTS_DIV);
    DATA_TYPE normalization_factor = pow(base, (DATA_TYPE)-BETA);
    
    DATA_TYPE f_in = input[input_index];
    DATA_TYPE normres =  f_in*normalization_factor;
    output[output_index] = FUNC_CALL(activation_function)(normres, NL_M ,NL_N);
}

KERNEL(normalization)(__global const DATA_TYPE* input, __global DATA_TYPE* output)
{
    const unsigned int x                = get_global_id(0);
    const unsigned int y                = get_global_id(1);
#if OUTPUT_BATCH_NUM == 1
    const unsigned int z                = get_global_id(2);
    const unsigned int w                = 0;
#else
    const unsigned int z                = get_global_id(2) % OUTPUT_FEATURE_NUM;
    const unsigned int w                = get_global_id(2) / OUTPUT_FEATURE_NUM;
#endif
    const unsigned int input_index      = w*INPUT_BATCH_PITCH + z*INPUT_FEATURE_PITCH + y*INPUT_Y_PITCH + x + INPUT_OFFSET;
    const unsigned int output_index     = w*OUTPUT_BATCH_PITCH + z*OUTPUT_FEATURE_PITCH + y*OUTPUT_Y_PITCH + x + OUTPUT_OFFSET;

    COUNTER_TYPE sum = 0.0f;

#ifdef ACROSS_MAPS

    unsigned int j_offset = input_index - ROUND_NORM_HALF_SIZE*INPUT_FEATURE_PITCH;

    for(int j = 0 ; j < ROUND_NORM_SIZE ; j++)
    {
        const int z_idx = (j + z - ROUND_NORM_HALF_SIZE);
        bool zero = (z_idx < 0 || z_idx >= INPUT_FEATURE_NUM);
        DATA_TYPE val = zero ? 0.0f : input[j_offset];
        sum += val*val;
        j_offset += INPUT_FEATURE_PITCH;
    }
    
    FUNC_CALL(internal_normalize)(input, input_index, output, output_index, sum);
    
#else

    const int x_start = ((int)x - ROUND_NORM_HALF_SIZE);
    const int y_start = ((int)y - ROUND_NORM_HALF_SIZE);
    unsigned int input_offset = w*INPUT_BATCH_PITCH + z*INPUT_FEATURE_PITCH + y_start*INPUT_Y_PITCH + x_start + INPUT_OFFSET;

    for (unsigned int j = 0; j < ROUND_NORM_SIZE ; ++j) 
    {
        for (unsigned int i = 0; i < ROUND_NORM_SIZE ; ++i) 
        {
            int input_offset_x = x_start + i;
            int input_offset_y = y_start + j;
            bool zero = false;
            zero = input_offset_x < 0 ? true : zero;
            zero = input_offset_y < 0 ? true : zero;
            zero = input_offset_x >= INPUT_SIZE_X ? true : zero;
            zero = input_offset_y >= INPUT_SIZE_Y ? true : zero;

            DATA_TYPE val = zero ? 0.0f : input[input_offset];
            
            sum += val*val;
            ++input_offset;
        }
        input_offset += INPUT_Y_PITCH - ROUND_NORM_SIZE;
    }

    FUNC_CALL(internal_normalize)(input, input_index, output, output_index, sum);
#endif
}
