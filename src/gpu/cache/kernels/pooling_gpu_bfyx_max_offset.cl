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


#if FP16_SUPPORTED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif


KERNEL(pooling_gpu_bfyx_max_offset)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    // constexpr:
    const int input_buffer_size_x = INPUT_PADDING_LOWER_SIZE_X + INPUT_SIZE_X + INPUT_PADDING_UPPER_SIZE_X;
    const int input_buffer_size_y = INPUT_PADDING_LOWER_SIZE_Y + INPUT_SIZE_Y + INPUT_PADDING_UPPER_SIZE_Y;
    const int input_buffer_size_f = INPUT_PADDING_LOWER_FEATURE_NUM + INPUT_FEATURE_NUM + INPUT_PADDING_UPPER_FEATURE_NUM;

    const uint output_buffer_size_x = OUTPUT_PADDING_LOWER_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PADDING_UPPER_SIZE_X;
    const uint output_buffer_size_y = OUTPUT_PADDING_LOWER_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PADDING_UPPER_SIZE_Y;
    const uint output_buffer_size_f = OUTPUT_PADDING_LOWER_FEATURE_NUM + OUTPUT_FEATURE_NUM + OUTPUT_PADDING_UPPER_FEATURE_NUM;

    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    if ((x >= OUTPUT_SIZE_X) || (y >= OUTPUT_SIZE_Y))
        return;

    const int offset_x = INPUT_PADDING_LOWER_SIZE_X + x * STRIDE_SIZE_X + INPUT_OFFSET_SIZE_X;
    const int offset_y = INPUT_PADDING_LOWER_SIZE_Y + y * STRIDE_SIZE_Y + INPUT_OFFSET_SIZE_Y;

	if ((offset_x >= INPUT_PADDING_LOWER_SIZE_X + INPUT_SIZE_X) ||
	    (offset_y >= INPUT_PADDING_LOWER_SIZE_Y + INPUT_SIZE_Y))
        return;

    UNIT_TYPE result = UNIT_INIT_VAL_MAX;

    const int batch_and_feature_offset = get_global_id(2);
    const uint b = batch_and_feature_offset / INPUT_FEATURE_NUM;
    const uint f = batch_and_feature_offset % INPUT_FEATURE_NUM;

    int window_offset = (INPUT_PADDING_LOWER_BATCH_NUM + b) * input_buffer_size_x * input_buffer_size_y * input_buffer_size_f;
    window_offset += (INPUT_PADDING_LOWER_FEATURE_NUM + f) * input_buffer_size_x * input_buffer_size_y;

    for(uint j = 0; j < WINDOW_SIZE_Y; j++)
    {
        int input_offset_y = offset_y + j;
        bool zero_y = input_offset_y >= input_buffer_size_y || input_offset_y < 0;
        if(!zero_y)
        {
            for(uint i = 0; i < WINDOW_SIZE_X; i++)
            {
                int input_offset_x = offset_x + i;
                bool zero = input_offset_x >= input_buffer_size_x || input_offset_x < 0;
                if(!zero)
                {
                    int input_idx = window_offset;
                    input_idx += input_offset_y * input_buffer_size_x + input_offset_x;
                    result = max(result, input[input_idx]);
                }
            }
        }
    }

    uint output_pos = (OUTPUT_PADDING_LOWER_BATCH_NUM + b) * output_buffer_size_x * output_buffer_size_y * output_buffer_size_f;
    output_pos += (OUTPUT_PADDING_LOWER_FEATURE_NUM + f) * output_buffer_size_x * output_buffer_size_y;
    output_pos += (OUTPUT_PADDING_LOWER_SIZE_Y + y) * output_buffer_size_x + OUTPUT_PADDING_LOWER_SIZE_X + x;
    output[output_pos] = result;
}