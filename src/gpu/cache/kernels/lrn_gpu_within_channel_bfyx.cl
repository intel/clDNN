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


KERNEL (lrn_gpu_within_channel_bfyx)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    const uint output_buffer_size_x = OUTPUT_PADDING_LOWER_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PADDING_UPPER_SIZE_X;
    const uint output_buffer_size_y = OUTPUT_PADDING_LOWER_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PADDING_UPPER_SIZE_Y;

    for (uint index = get_global_id(0) ; index < COUNT ; index += get_global_size(0))
    {
        const uint batch_num = INPUT_BATCH_NUM;
        const uint global_id = index;
        const uint batch_id = index / (COUNT / batch_num);
        const uint feature_id = (index % (COUNT / batch_num)) / (INPUT_SIZE_X*INPUT_SIZE_Y);
        const uint x = index % INPUT_SIZE_X;
        const uint y = (index / INPUT_SIZE_X) % INPUT_SIZE_Y;
        const uint first_index_in_batch = (batch_id * INPUT_FEATURE_NUM + feature_id) * INPUT_BUFFER_SIZE_X * INPUT_BUFFER_SIZE_Y;
        const uint input_id = first_index_in_batch + (INPUT_PADDING_LOWER_SIZE_Y + y) * INPUT_BUFFER_SIZE_X + INPUT_PADDING_LOWER_SIZE_X + x;

        int wstart = x - PAD;
        int hstart = y - PAD;
        int hend = min(hstart + P_SIZE, INPUT_SIZE_Y + PAD);
        int wend = min(wstart + P_SIZE, INPUT_SIZE_X + PAD);
        const int pool_size = (hend - hstart) * (wend - wstart);
        hstart = max(hstart, (int)0);
        wstart = max(wstart, (int)0);
        hend = min(hend, INPUT_SIZE_Y);
        wend = min(wend, INPUT_SIZE_X);
        UNIT_TYPE aveval = 0;
        const uint first_element_offset = INPUT_PADDING_LOWER_SIZE_X + INPUT_BUFFER_SIZE_X * INPUT_PADDING_LOWER_SIZE_Y;
        __global const UNIT_TYPE* bottom_slice = input + first_index_in_batch + first_element_offset;
        for (int h = hstart; h < hend; ++h)
        {
            for (int w = wstart; w < wend; ++w)
            {
                UNIT_TYPE tmp_val = bottom_slice[h * INPUT_BUFFER_SIZE_X + w] * UNIT_CVT_FUNC(ALPHA_VAL_FACTOR);
                aveval += (tmp_val * tmp_val);
            }
        }

        UNIT_TYPE acc = aveval / pool_size;
        acc = mad(acc, UNIT_CVT_FUNC(ALPHA), UNIT_CVT_FUNC(K));
        acc = native_powr(acc, -UNIT_CVT_FUNC(BETA));

        uint output_pos = (batch_id * OUTPUT_FEATURE_NUM + feature_id) * output_buffer_size_x * output_buffer_size_y;
        output_pos += (OUTPUT_PADDING_LOWER_SIZE_Y + y) * output_buffer_size_x + OUTPUT_PADDING_LOWER_SIZE_X + x;

        output[output_pos] = acc * input[input_id];
    }
}