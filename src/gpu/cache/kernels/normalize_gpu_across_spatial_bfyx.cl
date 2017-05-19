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


KERNEL (normalize_gpu_across_spatial_bfyx)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output, const __global UNIT_TYPE* scale_input)
{
	const uint b = get_global_id(0);

	float norm = EPSILON;

	uint input_first = b * INPUT_FEATURE_NUM * INPUT_BUFFER_SIZE_Y * INPUT_BUFFER_SIZE_X;

	// Compute norm
	uint input_idx = input_first;
	for (int f = 0; f < INPUT_FEATURE_NUM; f++)
	{
		input_idx += INPUT_PADDING_LOWER_SIZE_Y * INPUT_BUFFER_SIZE_X;
		for (int y = 0; y < INPUT_SIZE_Y; y++)
		{
			input_idx += INPUT_PADDING_LOWER_SIZE_X;
			for (int x = 0; x < INPUT_SIZE_X; x++)
			{
				float value = (float)input[input_idx];
				norm = mad(value, value, norm);
				input_idx++;
			}
			input_idx += INPUT_PADDING_UPPER_SIZE_X;
		}
		input_idx += INPUT_PADDING_UPPER_SIZE_Y * INPUT_BUFFER_SIZE_X;
	}
	norm = native_powr(norm, -0.5f);

	uint output_idx = b * OUTPUT_FEATURE_NUM * OUTPUT_BUFFER_SIZE_X * OUTPUT_BUFFER_SIZE_Y;

	// Scale the input
	input_idx = input_first;
	for (int f = 0; f < INPUT_FEATURE_NUM; f++)
	{
		input_idx += INPUT_PADDING_LOWER_SIZE_Y * INPUT_BUFFER_SIZE_X;
		output_idx += OUTPUT_PADDING_LOWER_SIZE_Y * OUTPUT_BUFFER_SIZE_X;
		for (int y = 0; y < INPUT_SIZE_Y; y++)
		{
			input_idx += INPUT_PADDING_LOWER_SIZE_X;
			output_idx += OUTPUT_PADDING_LOWER_SIZE_X;
			for (int x = 0; x < INPUT_SIZE_X; x++)
			{
				output[output_idx] = UNIT_CVT_FUNC(norm) * input[input_idx] * scale_input[SCALE_INDEX];
				input_idx++;
				output_idx++;
			}
			input_idx += INPUT_PADDING_UPPER_SIZE_X;
			output_idx += OUTPUT_PADDING_UPPER_SIZE_X;
		}
		input_idx += INPUT_PADDING_UPPER_SIZE_Y * INPUT_BUFFER_SIZE_X;
		output_idx += OUTPUT_PADDING_UPPER_SIZE_Y * OUTPUT_BUFFER_SIZE_X;
	}
}


#undef UNIT_CVT_FUNC
