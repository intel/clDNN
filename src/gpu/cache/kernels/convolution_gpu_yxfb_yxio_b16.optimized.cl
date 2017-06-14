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

KERNEL(convolution_gpu_yxfb_yxio_b16)(
    const __global float* input,
    __global float* output,
    const __global float* filter,
#if BIAS_TERM
    const __global float* bias,
#endif
    uint split_idx)
{
    const uint linear_id_xy = get_global_id(1) + get_global_size(1) * get_global_id(2);
    // we're computing 8 OUTPUT_FEATURE_MAP so we must divide by 8, but we got 8 batches, so no division is needed.
    uint global_id = (((uint)get_global_id(0) / WORK_ITEMS_PER_SINGLE_BATCHES_ELEMENTS) + (linear_id_xy * FILTER_ARRAY_NUM + split_idx) * (FILTER_OUTPUT_FEATURE_NUM / OFM_PER_WORK_ITEM)) * WORK_ITEMS_PER_SINGLE_BATCHES_ELEMENTS;

    const uint out_batch_id = (uint)get_local_id(0) + LOCAL_WORK_GROUP_SIZE * BATCHES_PER_WORK_ITEM * ((uint)get_group_id(0) % LOCAL_WORK_GROUPS_PER_SINGLE_BATCHES_ELEMENTS);
    const uint out_x = get_global_id(1);
    const uint out_y = get_global_id(2);

    const uint out_id = (global_id / WORK_ITEMS_PER_SINGLE_BATCHES_ELEMENTS) * OFM_PER_WORK_ITEM * INPUT_BATCH_NUM + out_batch_id;

    const uint ofm_offset = ((global_id * OFM_PER_WORK_ITEM) / WORK_ITEMS_PER_SINGLE_BATCHES_ELEMENTS) % FILTER_OUTPUT_FEATURE_NUM;

    bool finish = false;

    finish = out_x >= OUTPUT_LIMIT_SIZE_X || out_x < OUTPUT_PADDING_LOWER_SIZE_X;
    finish = (out_y >= OUTPUT_LIMIT_SIZE_Y || out_y < OUTPUT_PADDING_LOWER_SIZE_Y) ? true : finish;

    const uint sub_group_id = get_local_id(0);

    float8 _data[BATCHES_PER_WORK_ITEM];
    for(uint i = 0; i < BATCHES_PER_WORK_ITEM; i++)
    {
        _data[i] = 0.f;
    }
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
                        int input_idx = (input_offset_x + (input_offset_y * INPUT_SIZE_X)) * INPUT_FEATURE_NUM * INPUT_BATCH_NUM;
                        input_idx += split_idx * FILTER_INPUT_FEATURE_NUM * INPUT_BATCH_NUM;
                        input_idx += out_batch_id;

                        //sub_group_id used as offset to make each workitem load different filter, and then shuffle it
                        int filter_idx = ofm_offset + sub_group_id + FILTER_INPUT_FEATURE_NUM * (FILTER_OUTPUT_FEATURE_NUM * (i * FILTER_SIZE_X + j));

                        for (uint h = 0; h < FILTER_INPUT_FEATURE_NUM; h++)
                        {
#ifdef USE_BLOCK_READ_2
                            float2 _input = as_float2(intel_sub_group_block_read2((const __global uint*)input + input_idx));
                            float8 _filter = TRANSPOSE_BLOCK_8(filter[filter_idx]);
                            _data[0] = mad(_input.s0, _filter, _data[0]);
                            _data[1] = mad(_input.s1, _filter, _data[1]);
                            input_idx += INPUT_BATCH_NUM;
#else
                            float8 _filter = TRANSPOSE_BLOCK_8(filter[filter_idx]);
                            for(uint s = 0; s < BATCHES_PER_WORK_ITEM; s++)
                            {
                                _data[s] = mad(input[input_idx], _filter, _data[s]);
                                input_idx += LOCAL_WORK_GROUP_SIZE;
                            }
                            input_idx += INPUT_BATCH_NUM - BATCHES_PER_WORK_ITEM * LOCAL_WORK_GROUP_SIZE;
#endif
                            filter_idx += FILTER_OUTPUT_FEATURE_NUM;
                        }
                    }
                }
            }
        }
    }
#if BIAS_TERM
    float bias_val = bias[ofm_offset + sub_group_id];
    for(uint s = 0; s < BATCHES_PER_WORK_ITEM; s++)
    {
        ADD_BIAS_8(_data[s], bias_val);
    }
#endif
    for(uint s = 0; s < BATCHES_PER_WORK_ITEM; s++)
    {
        ACTIVATION(_data[s], _data[s]);
    }

    for(uint s = 0; s < BATCHES_PER_WORK_ITEM; s++)
    {
        int _out_id = out_id + s * LOCAL_WORK_GROUP_SIZE;
        output[_out_id] = _data[s].s0; _out_id += INPUT_BATCH_NUM;
        output[_out_id] = _data[s].s1; _out_id += INPUT_BATCH_NUM;
        output[_out_id] = _data[s].s2; _out_id += INPUT_BATCH_NUM;
        output[_out_id] = _data[s].s3; _out_id += INPUT_BATCH_NUM;
        output[_out_id] = _data[s].s4; _out_id += INPUT_BATCH_NUM;
        output[_out_id] = _data[s].s5; _out_id += INPUT_BATCH_NUM;
        output[_out_id] = _data[s].s6; _out_id += INPUT_BATCH_NUM;
        output[_out_id] = _data[s].s7; _out_id += INPUT_BATCH_NUM;
    }
}

#undef ACTIVATION