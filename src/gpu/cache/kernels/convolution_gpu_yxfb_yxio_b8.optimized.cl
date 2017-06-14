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


#if RELU
    #define ACTIVATION(output, input) output = isinf(NEGATIVE_SLOPE) ? ((input >= 0.0f) ? \
    input : -NEGATIVE_SLOPE) : (max(input, 0.0f) + NEGATIVE_SLOPE * min(input, 0.0f));
#else
    #define ACTIVATION(output, input) output = input;
#endif

__attribute__((reqd_work_group_size(LOCAL_WORK_GROUP_SIZE, 1, 1)))
KERNEL(convolution_gpu_yxfb_yxio_b8)(
    const __global float* input,
    __global float* output,
    const __global float* filter,
#if BIAS_TERM
    const __global float* bias,
#endif
    uint split_idx)
{
    const uint batch_num = INPUT_BATCH_NUM;

    const uint linear_id_xy = get_global_id(1) + get_global_size(1) * get_global_id(2);
    // we're computing 8 OUTPUT_FEATURE_MAP so we must divide by 8, but we got 8 batches, so no division is needed.
    uint global_id = ((uint)get_global_id(0) / batch_num) * batch_num + (linear_id_xy * FILTER_ARRAY_NUM + split_idx) * (FILTER_OUTPUT_FEATURE_NUM / OFM_PER_WORK_ITEM) * batch_num;

    const uint out_batch_id = get_local_id(0);
    const uint out_x = get_global_id(1);
    const uint out_y = get_global_id(2);

    const uint out_id = (global_id / batch_num) * OFM_PER_WORK_ITEM * batch_num + out_batch_id;

    const uint ofm_offset = (global_id * OFM_PER_WORK_ITEM) / batch_num % FILTER_OUTPUT_FEATURE_NUM;

    bool finish = out_x >= OUTPUT_LIMIT_SIZE_X || out_x < OUTPUT_PADDING_LOWER_SIZE_X
               || out_y >= OUTPUT_LIMIT_SIZE_Y || out_y < OUTPUT_PADDING_LOWER_SIZE_Y;

    const uint sub_group_id = get_local_id(0);

    float8 _data0 = 0.f;
#if OFM_PER_WORK_ITEM == 16
    float8 _data1 = 0.f;
#endif

    if(!finish)
    {
        const int x = out_x * STRIDE_SIZE_X + INPUT_OFFSET_SIZE_X;
        const int y = out_y * STRIDE_SIZE_Y + INPUT_OFFSET_SIZE_Y;

        for (uint i = 0; i < FILTER_SIZE_Y; i++)
        {
            int input_offset_y = y + i * DILATION_SIZE_Y;
            bool zero_y = input_offset_y >= INPUT_SIZE_Y || input_offset_y < 0;

            if(!zero_y)
            {
                for (uint j = 0; j < FILTER_SIZE_X; j++)
                {
                    int input_offset_x = x + j * DILATION_SIZE_X;

                    bool zero = input_offset_x >= INPUT_SIZE_X || input_offset_x < 0;

                    if(!zero)
                    {
                        uint input_idx = (input_offset_x + (input_offset_y * INPUT_SIZE_X)) * INPUT_FEATURE_NUM * INPUT_BATCH_NUM;
                        input_idx += split_idx * FILTER_INPUT_FEATURE_NUM * INPUT_BATCH_NUM;
                        input_idx += out_batch_id;

                        //sub_group_id used as offset to make each workitem load different filter, and then shuffle it
                        uint filter_idx = ofm_offset + sub_group_id + FILTER_INPUT_FEATURE_NUM * (FILTER_OUTPUT_FEATURE_NUM * (i * FILTER_SIZE_X + j));
#if OFM_PER_WORK_ITEM == 16
                        uint filter_idx2 = filter_idx + 8;
#endif
                        for (uint h = 0; h < FILTER_INPUT_FEATURE_NUM / 8; h++)
                        {
                            float8 _input = as_float8(intel_sub_group_block_read8((const __global uint*)input + input_idx));

                            DOT_PRODUCT_8(_data0, _input.s0, filter[filter_idx]) filter_idx += FILTER_OUTPUT_FEATURE_NUM;
#if OFM_PER_WORK_ITEM == 16
                            DOT_PRODUCT_8(_data1, _input.s0, filter[filter_idx2]) filter_idx2 += FILTER_OUTPUT_FEATURE_NUM;
#endif
                            DOT_PRODUCT_8(_data0, _input.s1, filter[filter_idx]) filter_idx += FILTER_OUTPUT_FEATURE_NUM;
#if OFM_PER_WORK_ITEM == 16
                            DOT_PRODUCT_8(_data1, _input.s1, filter[filter_idx2]) filter_idx2 += FILTER_OUTPUT_FEATURE_NUM;
#endif
                            DOT_PRODUCT_8(_data0, _input.s2, filter[filter_idx]) filter_idx += FILTER_OUTPUT_FEATURE_NUM;
#if OFM_PER_WORK_ITEM == 16
                            DOT_PRODUCT_8(_data1, _input.s2, filter[filter_idx2]) filter_idx2 += FILTER_OUTPUT_FEATURE_NUM;
#endif
                            DOT_PRODUCT_8(_data0, _input.s3, filter[filter_idx]) filter_idx += FILTER_OUTPUT_FEATURE_NUM;
#if OFM_PER_WORK_ITEM == 16
                            DOT_PRODUCT_8(_data1, _input.s3, filter[filter_idx2]) filter_idx2 += FILTER_OUTPUT_FEATURE_NUM;
#endif
                            DOT_PRODUCT_8(_data0, _input.s4, filter[filter_idx]) filter_idx += FILTER_OUTPUT_FEATURE_NUM;
#if OFM_PER_WORK_ITEM == 16
                            DOT_PRODUCT_8(_data1, _input.s4, filter[filter_idx2]) filter_idx2 += FILTER_OUTPUT_FEATURE_NUM;
#endif
                            DOT_PRODUCT_8(_data0, _input.s5, filter[filter_idx]) filter_idx += FILTER_OUTPUT_FEATURE_NUM;
#if OFM_PER_WORK_ITEM == 16
                            DOT_PRODUCT_8(_data1, _input.s5, filter[filter_idx2]) filter_idx2 += FILTER_OUTPUT_FEATURE_NUM;
#endif
                            DOT_PRODUCT_8(_data0, _input.s6, filter[filter_idx]) filter_idx += FILTER_OUTPUT_FEATURE_NUM;
#if OFM_PER_WORK_ITEM == 16
                            DOT_PRODUCT_8(_data1, _input.s6, filter[filter_idx2]) filter_idx2 += FILTER_OUTPUT_FEATURE_NUM;
#endif
                            DOT_PRODUCT_8(_data0, _input.s7, filter[filter_idx]) filter_idx += FILTER_OUTPUT_FEATURE_NUM;
#if OFM_PER_WORK_ITEM == 16
                            DOT_PRODUCT_8(_data1, _input.s7, filter[filter_idx2]) filter_idx2 += FILTER_OUTPUT_FEATURE_NUM;
#endif
                            input_idx += 8 * INPUT_BATCH_NUM;
                        }
                        for (uint h = FILTER_INPUT_FEATURE_NUM - (FILTER_INPUT_FEATURE_NUM % 8); h < FILTER_INPUT_FEATURE_NUM; h++)
                        {
                            float8 _filter = TRANSPOSE_BLOCK_8(filter[filter_idx]); filter_idx += FILTER_OUTPUT_FEATURE_NUM;
                            _data0 = mad(input[input_idx], _filter, _data0);
#if OFM_PER_WORK_ITEM == 16
                            float8 _filter2 = TRANSPOSE_BLOCK_8(filter[filter_idx2]); filter_idx2 += FILTER_OUTPUT_FEATURE_NUM;
                            _data1 = mad(input[input_idx], _filter2, _data1);
#endif
                            input_idx += INPUT_BATCH_NUM;
                        }
                    }
                }
            }
        }
    }
#if BIAS_TERM
    ADD_BIAS_8(_data0, bias[ofm_offset + sub_group_id]);
#if OFM_PER_WORK_ITEM == 16
    ADD_BIAS_8(_data1, bias[ofm_offset + sub_group_id + 8]);
#endif
#endif // #if BIAS_TERM
    ACTIVATION(_data0, _data0);
#if OFM_PER_WORK_ITEM == 16
    ACTIVATION(_data1, _data1);
#endif

    intel_sub_group_block_write8((__global uint*)output + out_id, as_uint8(_data0));
#if OFM_PER_WORK_ITEM == 16
    intel_sub_group_block_write8((__global uint*)output + out_id + 8 * batch_num, as_uint8(_data1));
#endif
}

#undef ACTIVATION