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

KERNEL (softmax_gpu_generic)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    const uint data_set_idx1 = get_global_id(0);
    const uint data_set_idx2 = get_global_id(1);
    
    const uint input_offset = data_set_idx1 + data_set_idx2*DATA_SET_SIZE*INPUT_STRIDE;
    const uint output_offset = data_set_idx1 + data_set_idx2*DATA_SET_SIZE*OUTPUT_STRIDE;
    
    UNIT_TYPE data[DATA_SET_SIZE];
    
    UNIT_TYPE maximum = -UNIT_VAL_MAX;
    for (uint i=0; i < DATA_SET_SIZE; ++i)
    {
        UNIT_TYPE in = input[input_offset + i * INPUT_STRIDE];
        maximum = max(maximum, in);
        data[i] = in;
    }
    
    UNIT_TYPE sum = UNIT_VAL_ZERO;
    for (uint i=0; i < DATA_SET_SIZE; ++i)
    {
        data[i] = native_exp(data[i] - maximum);
        sum += data[i];
    }
    
    for (uint i=0; i < DATA_SET_SIZE; ++i)
        output[output_offset + i * OUTPUT_STRIDE] = data[i] / sum;
}
