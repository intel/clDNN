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


KERNEL (normalize_gpu_within_spatial_yxfb)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output, const __global UNIT_TYPE* scale_input)
{
	const uint x = get_global_id(0);
	if (x >= INPUT_SIZE_X)
		return;
	const uint y = get_global_id(1);
	const uint b = get_global_id(2);

	uint input_first = b + INPUT_BATCH_NUM * INPUT_FEATURE_NUM * ((INPUT_PADDING_LOWER_SIZE_Y + y) * INPUT_BUFFER_SIZE_X + INPUT_PADDING_LOWER_SIZE_X + x);

	// Compute norm
	uint input_idx = input_first;
	float norm = EPSILON;
	for (int i = 0; i < INPUT_FEATURE_NUM; i++)
	{
		float value = (float)input[input_idx];
		norm = mad(value, value, norm);
		input_idx += INPUT_BATCH_NUM; // skip to the next feature
	}
	norm = native_powr(norm, -0.5f);

	uint output_idx = b + OUTPUT_BATCH_NUM * OUTPUT_FEATURE_NUM * ((OUTPUT_PADDING_LOWER_SIZE_Y + y) * OUTPUT_BUFFER_SIZE_X + OUTPUT_PADDING_LOWER_SIZE_X + x);

	// Scale the input
	input_idx = input_first;
	for (int f = 0; f < INPUT_FEATURE_NUM; f++)
	{
		output[output_idx] = UNIT_CVT_FUNC(norm) * input[input_idx] * scale_input[SCALE_INDEX];
		output_idx += OUTPUT_BATCH_NUM;
		input_idx += INPUT_BATCH_NUM;
	}
}


#undef UNIT_CVT_FUNC
