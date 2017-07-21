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

KERNEL(locally_connected)(
    __global DATA_TYPE* input, 
    __global DATA_TYPE* output, 
    __global DATA_TYPE* weights, 
    __global DATA_TYPE* biases)
{
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);
#if OUTPUT_BATCH_NUM == 1
    const unsigned z = get_global_id(2);
    const unsigned w = 0;
#else
    const unsigned z = get_global_id(2) % OUTPUT_FEATURE_NUM;
    const unsigned w = get_global_id(2) / OUTPUT_FEATURE_NUM;
#endif
    
    const unsigned bias_offset = z*OUTPUT_SIZE_X*OUTPUT_SIZE_Y + y*OUTPUT_SIZE_X + x;
    DATA_TYPE dotProd = biases[bias_offset];
    
    const int conv_size = KERNEL_HEIGHT * KERNEL_WIDTH;

    for (int k = 0; k < INPUT_FEATURE_NUM; ++k)
    {
        for (int j = 0; j < KERNEL_HEIGHT; ++j)
        {
            for (int i = 0; i < KERNEL_WIDTH; ++i)
            {
                const int src_x = x * STRIDE_X + i - INPUT_PADDING_X;
                const int src_y = y * STRIDE_Y + j - INPUT_PADDING_Y;

                if (src_x < 0 || src_x >= INPUT_SIZE_X || src_y < 0 || src_y >= INPUT_SIZE_Y)
                    continue;

                const int conv_idx = z * OUTPUT_SIZE_Y * OUTPUT_SIZE_X * INPUT_FEATURE_NUM * conv_size
                    + y * OUTPUT_SIZE_X * INPUT_FEATURE_NUM * conv_size
                    + x * INPUT_FEATURE_NUM * conv_size
                    + k * conv_size + j * KERNEL_WIDTH + i;

                const int input_idx = w*INPUT_BATCH_PITCH + k*INPUT_FEATURE_PITCH + src_y*INPUT_Y_PITCH + src_x + INPUT_OFFSET;
                
                const DATA_TYPE w = weights[conv_idx];
                const DATA_TYPE v = input[input_idx];
                dotProd += w*v;
            }
        } 
    }
    
    const unsigned int output_idx = w*OUTPUT_BATCH_PITCH + z*OUTPUT_FEATURE_PITCH + y*OUTPUT_Y_PITCH + x + OUTPUT_OFFSET;
    output[output_idx] = FUNC_CALL(activation_function)(dotProd, NL_M, NL_N);
}
