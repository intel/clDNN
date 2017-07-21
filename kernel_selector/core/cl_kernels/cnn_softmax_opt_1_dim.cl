/*
// Copyright (c) 2016 Intel Corporation
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
*/



#include "include/cnn_common.cl"


#if FP16_SUPPORTED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif


DATA_TYPE FUNC(find_max_value)(__local DATA_TYPE* partial_max, const int idx, const __global DATA_TYPE* input)
{
    DATA_TYPE value = -DATA_TYPE_MAX;
    for(int i = 0; i < ITEMS_NUM; i++)
    {
        value = fmax(value, input[LWS * i + idx]);
    }
    value = fmax(value, idx < LEFTOVERS? input[LWS * ITEMS_NUM + idx] : -DATA_TYPE_MAX);
    partial_max[idx] = value;

    barrier(CLK_LOCAL_MEM_FENCE);
    if(idx == 0)
    {
        for(int i = 1; i < LWS; i++)
        {
            partial_max[0] = fmax(partial_max[0], partial_max[i]);
        };
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    return partial_max[0];
}

KERNEL(softmax)(const __global DATA_TYPE* input, __global DATA_TYPE* output)
{
    const int idx = get_local_id(0);

    __local DATA_TYPE partial_max[LWS];
    const DATA_TYPE max_value = FUNC_CALL(find_max_value)(partial_max, idx, input);
    
    DATA_TYPE tmp_vals[ITEMS_NUM + 1];
    for(int i = 0; i < ITEMS_NUM; i++)
    {
        tmp_vals[i] = native_exp(input[LWS * i + idx] - max_value);
    }
    tmp_vals[ITEMS_NUM] = idx < LEFTOVERS ? native_exp(input[LWS * ITEMS_NUM + idx] - max_value) : DATA_TYPE_ZERO;

    // accumulate all values;
    __local DATA_TYPE partial_acc[LWS]; // all values accumulated;
    partial_acc[idx] = DATA_TYPE_ZERO;
    for(int i = 0; i < ITEMS_NUM + 1; i++)
    {
        partial_acc[idx] += tmp_vals[i];
    }

    barrier(CLK_LOCAL_MEM_FENCE); // we must be sure that all threads calculated max of elements(we can remove it if simd32 and GWS <= 32
    if(idx == 0)
    {
        for(int i = 1; i < LWS; i++)
        {
            partial_acc[0] += partial_acc[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = 0; i < ITEMS_NUM; i++)
    {
        output[LWS * i + idx] = tmp_vals[i] / partial_acc[0];
    }
    if(idx < LEFTOVERS)
        output[LWS * ITEMS_NUM + idx] = tmp_vals[ITEMS_NUM] / partial_acc[0];
}
