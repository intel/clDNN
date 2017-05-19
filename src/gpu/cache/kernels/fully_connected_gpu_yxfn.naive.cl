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

#if RELU && FP16_UNIT_USED
    #define ACTIVATION(output, input) output = isinf(convert_half(NEGATIVE_SLOPE)) ? ((input >= 0.0h) ? \
    input : -convert_half(NEGATIVE_SLOPE)) : (max(input, 0.0h) + convert_half(NEGATIVE_SLOPE) * min(input, 0.0h));
#elif RELU
    #define ACTIVATION(output, input) output = isinf(NEGATIVE_SLOPE) ? ((input >= 0.0f) ? \
    input : -NEGATIVE_SLOPE) : (max(input, 0.0f) + NEGATIVE_SLOPE * min(input, 0.0f));
#else
    #define ACTIVATION(output, input) output = input;
#endif

// Required JIT constants:
//  - FP16_SUPPORTED       - [0/1] Value indicating whether device supports FP16 OpenCL extension (cl_khr_fp16).
//  - FP16_UNIT_USED       - [0/1] Value indicating that current kernel should use FP16.
//  - UNIT_TYPE            - Type of unit of input/output/weight/bias.
//  - UNIT_VAL_ZERO        - Literal of current UNIT_TYPE that represents 0.
//  - INPUT_BATCH_NUM      - [int] Number of elements from single spatial and single feature that are grouped in single batch in input.
//  - INPUT_ELEMENTS_COUNT - [int] Cumulative number of elements from input that are processed in single batch.
//  - WEIGHTS_BATCH_NUM    - [int] Cumulative number of elements that are outputted in single batch.
//  - RELU                 - [0/1] Indicates that ReLU activation function should be used on output.
//  - NEGATIVE_SLOPE       - [float] Factor for negative output values (required when ReLU is specified).


KERNEL (fully_connected_gpu_yxfn)(
    const __global UNIT_TYPE* input,
    __global UNIT_TYPE* output,
    const __global UNIT_TYPE* weight
#if BIAS_TERM
    , __global UNIT_TYPE* bias)
#else
    )
#endif
{
    const uint x = get_global_id(0);
    const uint batch_id = x % INPUT_BATCH_NUM;
    const uint neuronIdx = x / INPUT_BATCH_NUM;
#if BIAS_TERM
    UNIT_TYPE result = bias[neuronIdx];
#else 
UNIT_TYPE result = UNIT_VAL_ZERO;
#endif
    uint weight_offset = neuronIdx * INPUT_FEATURE_NUM * INPUT_SIZE_Y * INPUT_SIZE_X;
    for (uint k = 0; k < INPUT_FEATURE_NUM; k++)
    {
        for (uint j = 0; j < INPUT_SIZE_Y; j++)
        {
            for(uint i = 0; i < INPUT_SIZE_X; i++)
            {
                result += input[(k + INPUT_FEATURE_NUM * (i + j * INPUT_SIZE_X)) * INPUT_BATCH_NUM + batch_id] * weight[weight_offset++];
            }
        }
    }
    ACTIVATION(output[x], result);
}

#undef ACTIVATION
