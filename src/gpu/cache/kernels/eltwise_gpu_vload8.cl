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

#define CONCAT_TOKEN_HANDLER1(prefix, suffix) prefix##suffix

// Expands and concatenates two tokens into one.
#define CONCAT_TOKEN(prefix, suffix) CONCAT_TOKEN_HANDLER1(prefix, suffix)

// Creates vector type.
#define MAKE_VECTOR_TYPE(elem_type, size) CONCAT_TOKEN(elem_type, size)

KERNEL (eltwise_gpu_vload8)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output, const __global UNIT_TYPE* input2)
{
    const uint global_id = get_global_id(0);

    const MAKE_VECTOR_TYPE(UNIT_TYPE, 8) in1 = vload8(global_id, input);
    const MAKE_VECTOR_TYPE(UNIT_TYPE, 8) in2 = vload8(global_id, input2);

    MAKE_VECTOR_TYPE(UNIT_TYPE, 8) result;
#if MAX_MODE_USED
    result = (in1 > in2 ? in1 : in2);
#elif PROD_MODE_USED
    result = in1 * in2;
#elif SUB_MODE_USED
    result = in1 - in2;
#else
    result = in1 + in2;
#endif
   
    ACTIVATION(result, result);

    vstore8(result, global_id, output);

}

#undef ACTIVATION
