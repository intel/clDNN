// Copyright (c) 2017 Intel Corporation
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

#if !defined(CONCAT_X) && !defined(CONCAT_Y)
#error "concatenation_gpu_spatial_bfyx kernel used but neither CONCAT_X nor CONCAT_Y was defined"
#endif

KERNEL (concatenation_gpu_spatial_bfyx)(__global UNIT_TYPE* input, __global UNIT_TYPE* output, uint concat_offset)
{
    // constexpr:
    const uint input_buffer_size_x = INPUT_PADDING_LOWER_SIZE_X + INPUT_SIZE_X + INPUT_PADDING_UPPER_SIZE_X;
    const uint input_buffer_size_y = INPUT_PADDING_LOWER_SIZE_Y + INPUT_SIZE_Y + INPUT_PADDING_UPPER_SIZE_Y;
    const uint input_buffer_size_f = INPUT_PADDING_LOWER_FEATURE_NUM + INPUT_FEATURE_NUM + INPUT_PADDING_UPPER_FEATURE_NUM;

    const uint output_buffer_size_x = OUTPUT_PADDING_LOWER_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PADDING_UPPER_SIZE_X;
    const uint output_buffer_size_y = OUTPUT_PADDING_LOWER_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PADDING_UPPER_SIZE_Y;
    const uint output_buffer_size_f = OUTPUT_PADDING_LOWER_FEATURE_NUM + OUTPUT_FEATURE_NUM + OUTPUT_PADDING_UPPER_FEATURE_NUM;

    const uint b = get_global_id(2);
    const uint f = get_global_id(1);
    const uint x_or_y = get_global_id(0);

    uint input_offset = (b + INPUT_PADDING_LOWER_BATCH_NUM) * input_buffer_size_x * input_buffer_size_y * input_buffer_size_f
                      + (f + INPUT_PADDING_LOWER_FEATURE_NUM) * input_buffer_size_x * input_buffer_size_y;

    uint output_offset = (b + OUTPUT_PADDING_LOWER_BATCH_NUM) * output_buffer_size_x * output_buffer_size_y * output_buffer_size_f
                       + (f + OUTPUT_PADDING_LOWER_FEATURE_NUM) * output_buffer_size_x * output_buffer_size_y;
    
#ifdef CONCAT_X
    input_offset += (x_or_y + INPUT_PADDING_LOWER_SIZE_Y) * input_buffer_size_x + INPUT_PADDING_LOWER_SIZE_X; //if we concat in x dimension, global_id(0) specifies row (y)
    const uint read_stride = 1;
    const uint read_length = INPUT_SIZE_X;

    output_offset += (x_or_y + OUTPUT_PADDING_LOWER_SIZE_Y) * output_buffer_size_x + OUTPUT_PADDING_LOWER_SIZE_X + concat_offset;
    const uint write_stride = 1;
#else
    input_offset += INPUT_PADDING_LOWER_SIZE_Y * input_buffer_size_y + INPUT_PADDING_LOWER_SIZE_X + x_or_y; //if we concat in y dimension, global_id(0) specifies column (x)
    const uint read_stride = input_buffer_size_x;
    const uint read_length = INPUT_SIZE_Y;

    output_offset += (concat_offset + OUTPUT_PADDING_LOWER_SIZE_Y) * output_buffer_size_x + x_or_y + OUTPUT_PADDING_LOWER_SIZE_X;
    const uint write_stride = output_buffer_size_x;
#endif

    for (size_t idx = 0; idx < read_length; ++idx)
    {
        output[output_offset] = input[input_offset];
        output_offset += write_stride;
        input_offset += read_stride;
    }
}
