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

#if (RELU || PRELU) && FP16_UNIT_USED
    #define ACTIVATION(output, input, slope) output = isinf(convert_half(slope)) ? ((input >= 0.0h) ? \
    input : -convert_half(slope)) : (max(input, 0.0h) + convert_half(slope) * min(input, 0.0h));
#elif (RELU || PRELU)
    #define ACTIVATION(output, input, slope) output = isinf(slope) ? ((input >= 0.0f) ? \
    input : -slope) : (max(input, 0.0f) + slope * min(input, 0.0f));
#else
    #define ACTIVATION(output, input) output = input;
#endif


KERNEL (relu_gpu)(const __global UNIT_TYPE* input, 
				  __global UNIT_TYPE* output
#if PRELU
				  , const __global UNIT_TYPE* slope
#endif   
				  )
{
    // constexpr:
    const uint input_buffer_size_x = INPUT_PADDING_LOWER_SIZE_X + INPUT_SIZE_X + INPUT_PADDING_UPPER_SIZE_X;
    const uint input_buffer_size_y = INPUT_PADDING_LOWER_SIZE_Y + INPUT_SIZE_Y + INPUT_PADDING_UPPER_SIZE_Y;
    const uint input_buffer_size_f = INPUT_PADDING_LOWER_FEATURE_NUM + INPUT_FEATURE_NUM + INPUT_PADDING_UPPER_FEATURE_NUM;
    const uint input_buffer_size_b = INPUT_PADDING_LOWER_BATCH_NUM + INPUT_BATCH_NUM + INPUT_PADDING_UPPER_BATCH_NUM;

    const uint output_buffer_size_x = OUTPUT_PADDING_LOWER_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PADDING_UPPER_SIZE_X;
    const uint output_buffer_size_y = OUTPUT_PADDING_LOWER_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PADDING_UPPER_SIZE_Y;
    const uint output_buffer_size_f = OUTPUT_PADDING_LOWER_FEATURE_NUM + OUTPUT_FEATURE_NUM + OUTPUT_PADDING_UPPER_FEATURE_NUM;
    const uint output_buffer_size_b = OUTPUT_PADDING_LOWER_BATCH_NUM + OUTPUT_BATCH_NUM + OUTPUT_PADDING_UPPER_BATCH_NUM;

    const uint batch_num = INPUT_BATCH_NUM;

    const uint global_id = get_global_id(0);
    const uint batch_id = global_id % batch_num;
    const uint feature_id = (global_id / batch_num) % INPUT_FEATURE_NUM;
    const uint x = ((global_id / batch_num) / INPUT_FEATURE_NUM) % INPUT_SIZE_X;
    const uint y = ((global_id / batch_num) / INPUT_FEATURE_NUM) / INPUT_SIZE_X;

    uint input_id = INPUT_PADDING_LOWER_BATCH_NUM + batch_id +
        input_buffer_size_b * (INPUT_PADDING_LOWER_FEATURE_NUM + feature_id +
        input_buffer_size_f * (INPUT_PADDING_LOWER_SIZE_X + x + 
        input_buffer_size_x * (INPUT_PADDING_LOWER_SIZE_Y + y)));

    uint output_id = OUTPUT_PADDING_LOWER_BATCH_NUM + batch_id + 
        output_buffer_size_b * (OUTPUT_PADDING_LOWER_FEATURE_NUM + feature_id +
        output_buffer_size_f * (OUTPUT_PADDING_LOWER_SIZE_X + x +
        output_buffer_size_x * (OUTPUT_PADDING_LOWER_SIZE_Y + y)));

#if RELU
	ACTIVATION(output[output_id], input[input_id], NEGATIVE_SLOPE);
#elif PRELU
	ACTIVATION(output[output_id], input[input_id], slope[feature_id]);
#else
	ACTIVATION(output[output_id], input[input_id]);
#endif
}

#undef ACTIVATION
