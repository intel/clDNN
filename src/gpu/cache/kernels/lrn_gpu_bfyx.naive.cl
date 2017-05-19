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

#if FP16_UNIT_USED
    #define UNIT_CVT_FUNC(val) convert_half(val)
#else
    #define UNIT_CVT_FUNC(val) (val)
#endif


KERNEL (lrn_gpu_bfyx)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    // constexpr:
    const uint output_buffer_size_x = OUTPUT_PADDING_LOWER_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PADDING_UPPER_SIZE_X;
    const uint output_buffer_size_y = OUTPUT_PADDING_LOWER_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PADDING_UPPER_SIZE_Y;

    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
    if (x >= INPUT_SIZE_X)
        return;
    const uint b_f = get_global_id(2);
    const uint b = b_f / INPUT_FEATURE_NUM;
    const uint f = b_f % INPUT_FEATURE_NUM;

    const uint first_index_in_batch = (b* INPUT_FEATURE_NUM + f) * INPUT_BUFFER_SIZE_X * INPUT_BUFFER_SIZE_Y;
    const uint input_id = first_index_in_batch + (INPUT_PADDING_LOWER_SIZE_Y + y) * INPUT_BUFFER_SIZE_X + INPUT_PADDING_LOWER_SIZE_X + x;

    const uint linear_id = x + INPUT_SIZE_X * (y + INPUT_SIZE_Y * b_f);
    UNIT_TYPE acc = UNIT_VAL_ZERO;

    int input_offset_f = f + HELP_INPUT_OFFSET;
    const uint first_element_index = INPUT_PADDING_LOWER_SIZE_X + INPUT_PADDING_LOWER_SIZE_Y * INPUT_BUFFER_SIZE_X;
    int input_idx = first_element_index + x + INPUT_BUFFER_SIZE_X * (y + INPUT_BUFFER_SIZE_Y * (input_offset_f + INPUT_FEATURE_NUM * b));

    for (int i = 0; i < P_SIZE; i++)
    {
        bool zero = input_offset_f < 0 || input_offset_f >= INPUT_FEATURE_NUM;

        UNIT_TYPE value = zero ? UNIT_VAL_ZERO : UNIT_CVT_FUNC(ALPHA_VAL_FACTOR_DIV_BY_SIZE) * input[input_idx];
        acc = mad(value, value, acc);

        input_offset_f++;
        input_idx += INPUT_BUFFER_SIZE_X * INPUT_BUFFER_SIZE_Y;
    }
    acc = mad(acc, UNIT_CVT_FUNC(ALPHA_DIV_BY_SIZE), UNIT_CVT_FUNC(K));
    acc = native_powr(acc, -UNIT_CVT_FUNC(BETA));

    uint output_pos = (b * OUTPUT_FEATURE_NUM + f) * output_buffer_size_x * output_buffer_size_y;
    output_pos += (OUTPUT_PADDING_LOWER_SIZE_Y + y) * output_buffer_size_x + OUTPUT_PADDING_LOWER_SIZE_X + x;

    output[output_pos] = acc * input[input_id];
}


#undef UNIT_CVT_FUNC
