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


KERNEL (lrn_gpu)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    const uint global_id = get_global_id(0);
    const uint element_offset = (uint)get_global_id(1) * INPUT_BATCH_NUM * INPUT_FEATURE_NUM;
    const uint linear_id = global_id + element_offset;
    int input_offset_f = global_id + HELP_INPUT_OFFSET * INPUT_BATCH_NUM;

    const uint batch_num = INPUT_BATCH_NUM;
    const uint feature_id = global_id / batch_num;
    const uint batch_id = linear_id % batch_num;
    const uint x = ((linear_id / batch_num) / INPUT_FEATURE_NUM) % INPUT_SIZE_X;
    const uint y = ((linear_id / batch_num) / INPUT_FEATURE_NUM) / INPUT_SIZE_X;
    int input_id = batch_id + batch_num * (feature_id + INPUT_FEATURE_NUM * ((INPUT_PADDING_LOWER_SIZE_Y + y) * INPUT_BUFFER_SIZE_X + INPUT_PADDING_LOWER_SIZE_X + x));
    
    const uint first_element_offset = INPUT_FEATURE_NUM * INPUT_BATCH_NUM * INPUT_PADDING_LOWER_SIZE_X + INPUT_PADDING_LOWER_SIZE_Y * INPUT_BUFFER_SIZE_X * INPUT_FEATURE_NUM * INPUT_BATCH_NUM;
    const uint element_offset_y = y * INPUT_FEATURE_NUM * INPUT_BATCH_NUM * (INPUT_PADDING_LOWER_SIZE_X + INPUT_PADDING_UPPER_SIZE_X);
    int input_idx = input_offset_f + first_element_offset +element_offset + element_offset_y;

    UNIT_TYPE acc = UNIT_VAL_ZERO;
    for (int i = 0; i < P_SIZE; i++)
    {
        bool zero = input_offset_f < 0 || input_offset_f >= INPUT_FEATURE_NUM * INPUT_BATCH_NUM;
        UNIT_TYPE value = zero ? UNIT_VAL_ZERO : UNIT_CVT_FUNC(ALPHA_VAL_FACTOR_DIV_BY_SIZE) * input[input_idx];
        acc = mad(value, value, acc);

        input_offset_f += INPUT_BATCH_NUM;
        input_idx += INPUT_BATCH_NUM;
    }
    acc = mad(acc, UNIT_CVT_FUNC(ALPHA_DIV_BY_SIZE), UNIT_CVT_FUNC(K));
    acc = native_powr(acc, -UNIT_CVT_FUNC(BETA));

    const uint out_y_offset = OUTPUT_FEATURE_NUM * OUTPUT_BATCH_NUM * OUTPUT_BUFFER_SIZE_X * (OUTPUT_PADDING_LOWER_SIZE_Y + y);
    const uint out_x_offset = OUTPUT_FEATURE_NUM * OUTPUT_BATCH_NUM * (OUTPUT_PADDING_LOWER_SIZE_X + x);
    const uint out_bf_offset = feature_id * OUTPUT_BATCH_NUM + batch_id;
    const uint output_idx = out_y_offset + out_x_offset + out_bf_offset;
    output[output_idx] =acc * input[input_id];
}

#undef UNIT_CVT_FUNC
