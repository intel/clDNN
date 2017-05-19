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

// input format: bx
// output format: bx
// input-output mapping: each input element has exactly one output element: y_i = exp(x_i - x_max) / sum_for_all_x(exp(x - x_max))
// WI mapping: each WI calculates ITEMS_NUM outputs, each WI operates only in his own batch
// gws: 0 - work items per batch (INPUT_SIZE_X / ITEMS_NUM), 1 - batch numbers
// lws: all work items assigned to one batch should be in one local group (gws0 == lws0) - one lw per batch
//
// overview (description per batch):
//  All WI find maximum value in batch:
//    - firstly each WI finds maximum from its chunk of data (ITEMS_NUM elements) and stores it in local memory
//    - lately single WI finds maximum from all partial maximums found in step 1
//  This value is lately used by all WI to calculate exp(x_i - x_max) and analogical operation is performed to calculate sum of these values.
//  Finaly, calculated sum is used to scale each value: exp(x_i - x_max) / sum.
__attribute__((intel_reqd_sub_group_size(LWS)))
__attribute__((reqd_work_group_size(LWS, 1, 1)))
KERNEL (softmax_gpu_batches_bx)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    const uint batch_idx = get_global_id(1);        //in processing of which batch this WI participates?
    const uint workers_per_batch = LWS;             //how many WI participates in processing of one batch
    const uint in_batch_idx = get_global_id(0);     //this WI's id in group of items processing single batch

    const uint batch_offset = batch_idx * INPUT_SIZE_X;
    const uint my_data_offset = batch_offset + in_batch_idx;

    UNIT_TYPE my_chunk[ITEMS_NUM + 1];
    UNIT_TYPE my_maximum = -UNIT_VAL_MAX;
    UNIT_TYPE my_sum = UNIT_VAL_ZERO;
    UNIT_TYPE tmp;
    __local UNIT_TYPE lw_storage[LWS];

    //each WI reads ITEMS_NUM consecutive items from batch
    for (uint i=0; i<ITEMS_NUM; ++i)
    {
        tmp = input[my_data_offset + i * LWS];
        my_maximum = max(my_maximum, tmp);
        my_chunk[i] = tmp;
    }

    if (in_batch_idx < LEFTOVERS)
    {
        tmp = input[batch_offset + LWS * ITEMS_NUM + in_batch_idx];
        my_maximum = max(my_maximum, tmp);
        my_chunk[ITEMS_NUM] = tmp;
    }

    lw_storage[in_batch_idx] = my_maximum;

    barrier(CLK_LOCAL_MEM_FENCE);
    if (in_batch_idx == 0)
    {
        for (uint i=1; i<LWS; ++i)
            my_maximum = max(my_maximum, lw_storage[i]);

        lw_storage[0] = my_maximum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    //my_maximum from this point is in fact global maximum
    my_maximum = lw_storage[0];

    for (uint i=0; i<ITEMS_NUM; ++i)
    {
        tmp = native_exp(my_chunk[i] - my_maximum);
        my_sum += tmp;
        my_chunk[i] = tmp;
    }

    if (in_batch_idx < LEFTOVERS)
    {
        tmp = native_exp(my_chunk[ITEMS_NUM] - my_maximum);
        my_sum += tmp;
        my_chunk[ITEMS_NUM] = tmp;
    }

    lw_storage[in_batch_idx] = my_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (in_batch_idx == 0)
    {
        for (uint i=1; i<LWS; ++i)
            my_sum += lw_storage[i];

        lw_storage[0] = my_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    my_sum = lw_storage[0];

    for (uint i=0; i<ITEMS_NUM; ++i)
        output[my_data_offset + i * LWS] = my_chunk[i] / my_sum;
    if (in_batch_idx < LEFTOVERS)
        output[batch_offset + LWS * ITEMS_NUM + in_batch_idx] = my_chunk[ITEMS_NUM] / my_sum;
}
