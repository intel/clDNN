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

#if RELU && FP16_UNIT_USED
    #define ACTIVATION(output, input) output = isinf(convert_half(NEGATIVE_SLOPE)) ? ((input >= 0.0h) ? \
    input : -convert_half(NEGATIVE_SLOPE)) : (max(input, 0.0h) + convert_half(NEGATIVE_SLOPE) * min(input, 0.0h));
#elif RELU
    #define ACTIVATION(output, input) output = isinf(NEGATIVE_SLOPE) ? ((input >= 0.0f) ? \
    input : -NEGATIVE_SLOPE) : (max(input, 0.0f) + NEGATIVE_SLOPE * min(input, 0.0f));
#else
    #define ACTIVATION(output, input) output = input;
#endif

KERNEL(deconvolution_gpu_bfyx_oiyx)(
    const __global UNIT_TYPE* input,
    __global UNIT_TYPE* output,
    const __global UNIT_TYPE* filter,
#if BIAS_TERM
    const __global UNIT_TYPE* bias,
#endif
    uint split_idx)
{
    const int batch_num = INPUT_BATCH_NUM;
    const uint batch_id = get_global_id(0) % INPUT_BATCH_NUM;
    const uint feature_id = get_global_id(0) / INPUT_BATCH_NUM;
    const uint out_x = get_global_id(1);
    const uint out_y = get_global_id(2);

    const uint linear_id = out_x + OUTPUT_SIZE_X * (out_y + OUTPUT_SIZE_Y * (feature_id + OUTPUT_FEATURE_NUM * batch_id));
    const int bifn_num = batch_num * OUTPUT_SIZE_X * OUTPUT_SIZE_Y * FILTER_OUTPUT_FEATURE_NUM;
    int global_id = linear_id % bifn_num + (linear_id / bifn_num) * bifn_num * FILTER_ARRAY_NUM + split_idx * bifn_num;

    const int ofm_offset = (global_id / batch_num) % FILTER_OUTPUT_FEATURE_NUM;

#if BIAS_TERM
    UNIT_TYPE result = bias[ofm_offset];
#else
    UNIT_TYPE result = UNIT_VAL_ZERO;
#endif

    bool finish = false;

    finish = out_x >= OUTPUT_LIMIT_SIZE_X || out_x < OUTPUT_PADDING_LOWER_SIZE_X;
    finish = (out_y >= OUTPUT_LIMIT_SIZE_Y || out_y < OUTPUT_PADDING_LOWER_SIZE_Y) ? true : finish;

    if(!finish)
    {
        const int batch_offset = global_id / (OUTPUT_FEATURE_NUM * OUTPUT_SIZE_X * OUTPUT_SIZE_Y);

        const int f_ofm_offset = ofm_offset * FILTER_SIZE_Y * FILTER_SIZE_X * FILTER_INPUT_FEATURE_NUM;

        const int x = out_x - INPUT_OFFSET_SIZE_X - (FILTER_SIZE_X - 1);
        const int y = out_y - INPUT_OFFSET_SIZE_Y - (FILTER_SIZE_Y - 1);

        for (uint i = 0; i < FILTER_SIZE_Y; i++)
        {
            int input_offset_y = y + i;
            bool zero_y = (input_offset_y >= INPUT_SIZE_Y * STRIDE_SIZE_Y) || (input_offset_y < 0) || ((input_offset_y % STRIDE_SIZE_Y) != 0);

            if(!zero_y)
            {
                for (uint j = 0; j < FILTER_SIZE_X; j++)
                {
                    int input_offset_x = x + j;
                    bool zero = (input_offset_x >= INPUT_SIZE_X * STRIDE_SIZE_X) || (input_offset_x < 0) || ((input_offset_x % STRIDE_SIZE_X) != 0);

                    if(!zero)
                    {
                        int input_idx = (input_offset_x / STRIDE_SIZE_X + (input_offset_y * INPUT_SIZE_X / STRIDE_SIZE_Y));
                        input_idx += split_idx * FILTER_INPUT_FEATURE_NUM * INPUT_SIZE_X * INPUT_SIZE_Y;
                        input_idx += (feature_id - ofm_offset) * INPUT_SIZE_X * INPUT_SIZE_Y;
                        input_idx += batch_offset * FILTER_OUTPUT_FEATURE_NUM * INPUT_SIZE_X * INPUT_SIZE_Y;

                        uint filter_idx = ((FILTER_SIZE_X * FILTER_SIZE_Y - 1) - (i * FILTER_SIZE_X + j)) + f_ofm_offset;

                        for (uint h = 0; h < FILTER_INPUT_FEATURE_NUM; h++)
                        {
#if FP16_UNIT_USED
                            result = fma(input[input_idx], filter[filter_idx], result);
#else
                            result = mad(input[input_idx], filter[filter_idx], result);
#endif
                            filter_idx += FILTER_SIZE_Y * FILTER_SIZE_X;
                            input_idx += INPUT_SIZE_X * INPUT_SIZE_Y;
                        }
                    }
                }
            }
        }
    }
    ACTIVATION(output[global_id], result);
}

#undef ACTIVATION