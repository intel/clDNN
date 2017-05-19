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

KERNEL (scale_gpu)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output, __global UNIT_TYPE* scale_input
#if BIAS_TERM
, __global UNIT_TYPE* bias)
#else
)
#endif
{
    // constexpr:
    const int input_buffer_size_x = INPUT_PADDING_LOWER_SIZE_X + INPUT_SIZE_X + INPUT_PADDING_UPPER_SIZE_X;
    const int input_buffer_size_y = INPUT_PADDING_LOWER_SIZE_Y + INPUT_SIZE_Y + INPUT_PADDING_UPPER_SIZE_Y;
    const uint output_buffer_size_x = OUTPUT_PADDING_LOWER_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PADDING_UPPER_SIZE_X;
    const uint output_buffer_size_y = OUTPUT_PADDING_LOWER_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PADDING_UPPER_SIZE_Y;

    const uint x = ((uint)get_global_id(2) % INPUT_SIZE_X);
    const uint y = ((uint)get_global_id(2) / INPUT_SIZE_X);

    if ((x >= OUTPUT_SIZE_X) || (y >= OUTPUT_SIZE_Y))
        return;

#if INPUT_BFYX_USED
    const uint input_linear_id = x + INPUT_PADDING_LOWER_SIZE_X + input_buffer_size_x * (INPUT_PADDING_LOWER_SIZE_Y + y + input_buffer_size_y * ((uint)get_global_id(1) + (uint)get_global_id(0) * INPUT_FEATURE_NUM));
    const uint output_linear_id = x + OUTPUT_PADDING_LOWER_SIZE_X + output_buffer_size_x * (OUTPUT_PADDING_LOWER_SIZE_Y + y + output_buffer_size_y * ((uint)get_global_id(1) + (uint)get_global_id(0) * OUTPUT_FEATURE_NUM));
#else
    const uint input_linear_id = (uint)get_global_id(0) + INPUT_BATCH_NUM * ((uint)get_global_id(1) + INPUT_FEATURE_NUM * (x + INPUT_PADDING_LOWER_SIZE_X + (uint)input_buffer_size_x * (INPUT_PADDING_LOWER_SIZE_Y + y)));
    const uint output_linear_id = (uint)get_global_id(0) + OUTPUT_BATCH_NUM * ((uint)get_global_id(1) + OUTPUT_FEATURE_NUM * (x + OUTPUT_PADDING_LOWER_SIZE_X + (uint)output_buffer_size_x * (OUTPUT_PADDING_LOWER_SIZE_Y + y)));
#endif

    const uint scale_batch_id = (SCALE_BATCH_NUM == 1) ? 0 : get_global_id(0);
    const uint scale_feature_id = (SCALE_FEATURE_NUM == 1) ? 0 : get_global_id(1);
    const uint scale_x = (SCALE_SIZE_X == 1) ? 0 : ((SCALE_SIZE_Y == 1) ? ((uint)get_global_id(2) % INPUT_SIZE_X) : ((uint)get_global_id(2) % SCALE_SIZE_X));
    const uint scale_y = (SCALE_SIZE_Y == 1) ? 0 : ((SCALE_SIZE_X == 1) ? ((uint)get_global_id(2) / INPUT_SIZE_X) : ((uint)get_global_id(2) / SCALE_SIZE_X));
#if SCALE_BFYX_USED
    const uint scale_linear_id = scale_x + SCALE_SIZE_X * (scale_y + SCALE_SIZE_Y * (scale_feature_id + scale_batch_id * SCALE_FEATURE_NUM));
#else
    const uint scale_linear_id = scale_batch_id + SCALE_BATCH_NUM * (scale_feature_id + SCALE_FEATURE_NUM * (scale_x + scale_y * SCALE_SIZE_X));
#endif

#if BIAS_TERM
    output[output_linear_id] = mad(input[input_linear_id], scale_input[scale_linear_id], bias[scale_linear_id]);
#else
    output[output_linear_id] = input[input_linear_id] * scale_input[scale_linear_id];
#endif
}