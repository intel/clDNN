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

KERNEL (softmax_gpu_batches_bfyx)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{

    const uint element_id = get_global_id(0); // flatten indexes in batch
    const uint batch_id = get_global_id(1); // index of batch
    const uint in_batch_offset = INPUT_SIZE_Y*INPUT_FEATURE_NUM;
    const uint batch_offset = INPUT_SIZE_X*INPUT_SIZE_Y*INPUT_FEATURE_NUM;
    const uint global_id = element_id + batch_id*batch_offset;

    if (element_id >= ELEMENTS_NUM)
        return;

    UNIT_TYPE tmp_vals = UNIT_VAL_ZERO;
    UNIT_TYPE feature_maximum = -UNIT_VAL_MAX;
    UNIT_TYPE feature_sum = UNIT_VAL_ZERO;
    UNIT_TYPE vals[ITEMS_NUM];


    //find max and allocate inputs to vals
    for (uint i = 0; i<ITEMS_NUM; ++i)
    {
        tmp_vals = input[element_id + i*in_batch_offset+batch_id*batch_offset];
        feature_maximum = max(feature_maximum, tmp_vals);
        vals[i] = tmp_vals;
    }

    //calculate native_exp and sum
    for (uint i = 0; i<ITEMS_NUM; ++i)
    {
        tmp_vals = native_exp(vals[i] - feature_maximum);
        feature_sum += tmp_vals;
        vals[i] = tmp_vals;
    }


    for (uint i = 0; i<ITEMS_NUM; ++i)
    {
        output[global_id + i*in_batch_offset] = vals[i] / feature_sum;
    }

}
