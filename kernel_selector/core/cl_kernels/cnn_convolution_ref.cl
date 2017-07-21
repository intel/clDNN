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

#include "include/cnn_common.cl"

KERNEL(convolution)(
    __global DATA_TYPE* input, 
    __global DATA_TYPE* output, 
    __global DATA_TYPE* weights, 
#ifdef OUTPUT_BIASED
    __global DATA_TYPE* biases,
#endif
    uint split_idx)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
#if OUTPUT_BATCH_NUM == 1
    const uint z = get_global_id(2);
    const uint w = 0;
#else
    const uint z = get_global_id(2) % OUTPUT_FEATURE_NUM;
    const uint w = get_global_id(2) / OUTPUT_FEATURE_NUM;
#endif

    const uint filter_size = INPUT_FEATURE_NUM * KERNEL_HEIGHT * KERNEL_WIDTH;
    
#if   defined BIAS_PER_OUTPUT
    const uint bias_index = z*OUTPUT_SIZE_X*OUTPUT_SIZE_Y + y*OUTPUT_SIZE_X + x;
#elif defined BIAS_PER_OFM
    const uint bias_index = z;
#endif

    DATA_TYPE dotProd = (DATA_TYPE)0.0f;
#ifdef OUTPUT_BIASED
    dotProd = biases[bias_index];
#endif

    const int input_x = x * STRIDE_X - INPUT_PADDING_X;
    const int input_y = y * STRIDE_Y - INPUT_PADDING_Y;

    const uint in_split_offset = split_idx * INPUT_FEATURE_PITCH * INPUT_FEATURE_NUM;
    const uint filter_offset = z*filter_size;
    const uint input_offset = w*INPUT_BATCH_PITCH + INPUT_OFFSET + in_split_offset;

    for (uint k = 0; k < INPUT_FEATURE_NUM; ++k)
    {
        for (uint j = 0; j < KERNEL_HEIGHT ; ++j)
        {
            const int input_offset_y = input_y + j * DILATION_Y;
            const bool zero_y = input_offset_y >= INPUT_SIZE_Y || input_offset_y < 0;

            if(!zero_y)
            {
                for (uint i = 0; i < KERNEL_WIDTH ; ++i)
                {
                    const int input_offset_x = input_x + i * DILATION_X;
                    const bool zero_x = input_offset_x >= INPUT_SIZE_X || input_offset_x < 0;

                    if(!zero_x)
                    {
                        uint input_idx = input_offset + (uint)input_offset_x*INPUT_X_PITCH + (uint)input_offset_y*INPUT_Y_PITCH + k*INPUT_FEATURE_PITCH;
                        uint filter_idx = filter_offset + k*KERNEL_WIDTH*KERNEL_HEIGHT + j*KERNEL_WIDTH + i;
                        dotProd += input[input_idx]*weights[filter_idx];
                    }
                }
            }
        }
    }
    
    const uint out_split_offset = split_idx * OUTPUT_FEATURE_PITCH * OUTPUT_FEATURE_NUM;
    const uint dst_index = w*OUTPUT_BATCH_PITCH + z*OUTPUT_FEATURE_PITCH + y*OUTPUT_Y_PITCH + x + OUTPUT_OFFSET + out_split_offset;
    output[dst_index] = FUNC_CALL(activation_function)(dotProd, NL_M, NL_N);
}
