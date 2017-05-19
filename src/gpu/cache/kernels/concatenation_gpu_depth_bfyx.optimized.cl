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


KERNEL (concatenation_gpu_depth_bfyx)(__global UNIT_TYPE* input, __global UNIT_TYPE* output, uint depth_offset)
{
    // constexpr:
    const uint input_buffer_size_x = INPUT_PADDING_LOWER_SIZE_X + INPUT_SIZE_X + INPUT_PADDING_UPPER_SIZE_X;
    const uint input_buffer_size_y = INPUT_PADDING_LOWER_SIZE_Y + INPUT_SIZE_Y + INPUT_PADDING_UPPER_SIZE_Y;
    const uint output_buffer_size_x = OUTPUT_PADDING_LOWER_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PADDING_UPPER_SIZE_X;
    const uint output_buffer_size_y = OUTPUT_PADDING_LOWER_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PADDING_UPPER_SIZE_Y;


    uint batch_id = get_global_id(0);
    uint feature_and_y_id = get_global_id(1);
    uint feature_id = feature_and_y_id / INPUT_SIZE_Y;
    uint y = feature_and_y_id % INPUT_SIZE_Y;

    if(feature_id >= INPUT_FEATURE_NUM)
        return;

    uint input_offset = (batch_id * INPUT_FEATURE_NUM + feature_id) * input_buffer_size_x * input_buffer_size_y;
    input_offset += (INPUT_PADDING_LOWER_SIZE_Y + y) * input_buffer_size_x + INPUT_PADDING_LOWER_SIZE_X;

    uint output_offset = (batch_id * OUTPUT_FEATURE_NUM + depth_offset + feature_id) * output_buffer_size_x * output_buffer_size_y;
    output_offset += (OUTPUT_PADDING_LOWER_SIZE_Y + y) * output_buffer_size_x + OUTPUT_PADDING_LOWER_SIZE_X;

    for(uint x = 0; x < INPUT_SIZE_X; x++)
    {
        output[output_offset++] = input[input_offset++];
    }
}