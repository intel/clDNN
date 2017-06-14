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


#if FP16_UNIT_USED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

KERNEL (batch_norm_gpu)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output, __global UNIT_TYPE* mean, __global UNIT_TYPE* variance)
{
    // constexpr:
    const int input_buffer_size_x = INPUT_PADDING_LOWER_SIZE_X + INPUT_SIZE_X + INPUT_PADDING_UPPER_SIZE_X;
    const int input_buffer_size_y = INPUT_PADDING_LOWER_SIZE_Y + INPUT_SIZE_Y + INPUT_PADDING_UPPER_SIZE_Y;
    const uint output_buffer_size_x = OUTPUT_PADDING_LOWER_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PADDING_UPPER_SIZE_X;
    const uint output_buffer_size_y = OUTPUT_PADDING_LOWER_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PADDING_UPPER_SIZE_Y;

    const uint feature_id = get_global_id(1);
    const uint x = ((uint)get_global_id(2) % INPUT_SIZE_X);
    const uint y = ((uint)get_global_id(2) / INPUT_SIZE_X);

    const uint feature_offset = feature_id * INPUT_BATCH_NUM;

#if BFYX_USED
    const uint input_linear_id = x + INPUT_PADDING_LOWER_SIZE_X + input_buffer_size_x * (INPUT_PADDING_LOWER_SIZE_Y + y + input_buffer_size_y * ((uint)get_global_id(1) + (uint)get_global_id(0) * INPUT_FEATURE_NUM));
    const uint output_linear_id = x + OUTPUT_PADDING_LOWER_SIZE_X + output_buffer_size_x * (OUTPUT_PADDING_LOWER_SIZE_Y + y + output_buffer_size_y * ((uint)get_global_id(1) + (uint)get_global_id(0) * OUTPUT_FEATURE_NUM));
#else
    const uint input_linear_id = (uint)get_global_id(0) + INPUT_BATCH_NUM * ((uint)get_global_id(1) + INPUT_FEATURE_NUM * (x + INPUT_PADDING_LOWER_SIZE_X + (uint)input_buffer_size_x * (INPUT_PADDING_LOWER_SIZE_Y + y)));
    const uint output_linear_id = (uint)get_global_id(0) + OUTPUT_BATCH_NUM * ((uint)get_global_id(1) + OUTPUT_FEATURE_NUM * (x + OUTPUT_PADDING_LOWER_SIZE_X + (uint)output_buffer_size_x * (OUTPUT_PADDING_LOWER_SIZE_Y + y)));
#endif

    //compute mean
    UNIT_TYPE acc = UNIT_VAL_ZERO;
    for(int i = 0; i < INPUT_BATCH_NUM; i++)
    {
        for(int j = 0; j < INPUT_SIZE_X * INPUT_SIZE_Y; j++)
        {
        acc += input[feature_offset + i + j * INPUT_BATCH_NUM * INPUT_FEATURE_NUM];
        }
    }
    UNIT_TYPE mean_val = acc / (INPUT_BATCH_NUM * INPUT_SIZE_X * INPUT_SIZE_Y);

    //compute variance using var(X) = E((X-EX)^2)
    acc = UNIT_VAL_ZERO;
    for(int i = 0; i < INPUT_BATCH_NUM; i++)
    {
        for(int j = 0; j < INPUT_SIZE_X * INPUT_SIZE_Y; j++)
        {
        acc += native_powr(input[feature_offset + i + j * INPUT_BATCH_NUM * INPUT_FEATURE_NUM] - mean_val, UNIT_VAL_SQUARE);
        }
    }

    UNIT_TYPE variance_val = acc / (INPUT_BATCH_NUM * INPUT_SIZE_X * INPUT_SIZE_Y);

    output[output_linear_id] = (input[input_linear_id] - mean_val) / sqrt(variance_val + EPSILON);
}