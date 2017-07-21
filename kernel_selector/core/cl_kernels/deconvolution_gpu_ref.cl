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

KERNEL(deconvolution_gpu_yxfb_ref)(
    const __global UNIT_TYPE* input,
    __global UNIT_TYPE* output,
    const __global UNIT_TYPE* filter,
#if BIAS_TERM
    const __global UNIT_TYPE* bias,
#endif
    uint split_idx)
{
    UNIT_TYPE result = UNIT_VAL_ZERO;

    const uint batch_offset = (uint)get_global_id(0) % INPUT_BATCH_NUM;
    const uint ofm_offset   = (uint)get_global_id(0) / INPUT_BATCH_NUM;
    const uint out_x        = (uint)get_global_id(1);
    const uint out_y        = (uint)get_global_id(2);

    const int x = (int)out_x + PADDING_SIZE_X - (FILTER_SIZE_X - 1);
    const int y = (int)out_y + PADDING_SIZE_Y - (FILTER_SIZE_Y - 1);
    
    const uint in_split_offset = split_idx * INPUT_FEATURE_PITCH * FILTER_IFM_NUM;
    const uint input_offset = INPUT_OFFSET + batch_offset*INPUT_BATCH_PITCH + in_split_offset;

    for (uint i = 0; i < FILTER_SIZE_Y; i++)
    {
        const int input_offset_y = y + i;
        const bool zero_y = (input_offset_y >= INPUT_SIZE_Y * STRIDE_SIZE_Y) || (input_offset_y < 0) || ((input_offset_y % STRIDE_SIZE_Y) != 0);

        if(!zero_y)
        {
            for (uint j = 0; j < FILTER_SIZE_X; j++)
            {
                const int input_offset_x = x + j;
                const bool zero_x = (input_offset_x >= INPUT_SIZE_X * STRIDE_SIZE_X) || (input_offset_x < 0) || ((input_offset_x % STRIDE_SIZE_X) != 0);

                if(!zero_x)
                {
                    uint fixed_input_offset_x = (uint)input_offset_x / STRIDE_SIZE_X;
                    uint fixed_input_offset_y = (uint)input_offset_y / STRIDE_SIZE_Y;
                    uint input_idx = input_offset + (uint)fixed_input_offset_x*INPUT_X_PITCH + (uint)fixed_input_offset_y*INPUT_Y_PITCH;

                    uint filter_idx = ofm_offset*FILTER_OFM_PITCH + (FILTER_SIZE_Y - i)*FILTER_Y_PITCH - (j + 1)*FILTER_X_PITCH;

                    for (uint h = 0; h < FILTER_IFM_NUM; h++)
                    {
                        result = fma(input[input_idx], filter[filter_idx], result);
                        filter_idx += FILTER_IFM_PITCH;
                        input_idx += INPUT_FEATURE_PITCH;
                    }
                }
            }
        }
    }
#if BIAS_TERM
    result += bias[ofm_offset];
#endif
    const uint out_split_offset = split_idx * OUTPUT_FEATURE_PITCH * FILTER_OFM_NUM;
    const uint dst_index = OUTPUT_OFFSET + out_split_offset + batch_offset*OUTPUT_BATCH_PITCH + ofm_offset*OUTPUT_FEATURE_PITCH + out_y*OUTPUT_Y_PITCH + out_x*OUTPUT_X_PITCH;
    ACTIVATION(output[dst_index], result);
}

#undef ACTIVATION
