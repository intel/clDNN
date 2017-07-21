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


#include "include/common.cl"

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

inline uint3 FUNC(reshape)(uint i, uint y, uint x)
{
#if (FILTER_DIMS == 4)
    return (uint3)(i,y,x);
#elif (FILTER_DIMS == 2)
    uint _i = i*INPUT_SIZE_Y*INPUT_SIZE_X + y*INPUT_SIZE_X + x;
    return (uint3)(_i,0,0);
#else
#error
#endif
}

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

    UNIT_TYPE result = UNIT_VAL_ZERO;

    uint weight_offset = neuronIdx * FILTER_OFM_PITCH;
    for (uint k = 0; k < INPUT_FEATURE_NUM; k++)
    {
        for (uint j = 0; j < INPUT_SIZE_Y; j++)
        {
            for(uint i = 0; i < INPUT_SIZE_X; i++)
            {
                uint3 widx = FUNC_CALL(reshape)(k,j,i);
                uint weight_idx = weight_offset + widx[0]*FILTER_IFM_PITCH + widx[1]*FILTER_Y_PITCH + widx[2]*FILTER_X_PITCH;
                uint input_idx = INPUT_OFFSET + k*INPUT_FEATURE_PITCH + j*INPUT_Y_PITCH + i*INPUT_X_PITCH + batch_id*INPUT_BATCH_PITCH;
                result += input[input_idx] * weight[weight_idx];
            }
        }
    }
    uint output_idx = OUTPUT_OFFSET + batch_id*OUTPUT_BATCH_PITCH + neuronIdx*OUTPUT_FEATURE_PITCH;

#if BIAS_TERM
    result += bias[neuronIdx];
#endif
    ACTIVATION(output[output_idx], result);
}

#undef ACTIVATION
