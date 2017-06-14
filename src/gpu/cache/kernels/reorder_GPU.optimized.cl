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

#define TYPE_CVT_FUNC3(val, type) convert_##type(val)
#define TYPE_CVT_FUNC2(val, type) TYPE_CVT_FUNC3(val, type)
#if SRC_DEST_TYPE_CVT
    #define SRC_DEST_TYPE_CVT_FUNC(val) TYPE_CVT_FUNC2(val, DEST_TYPE)
#else
    #define SRC_DEST_TYPE_CVT_FUNC(val) val
#endif

uint FUNC(OUT_FORMAT)(uint size[DIMENSIONS], uint pos[DIMENSIONS], uint lpad[DIMENSIONS], uint upad[DIMENSIONS]) {
    OUT_FORMAT_IMPLEMENTATION
}
KERNEL (reorder_GPU)(const __global SRC_TYPE* input, __global DEST_TYPE* output)
{
    const uint global_id_0 = get_global_id(0);
    const uint dim0 = global_id_0 % SIZE[CALCULATION_ORDER[0]];
    const uint dim1 = global_id_0 / SIZE[CALCULATION_ORDER[0]];
    const uint dim2 = get_global_id(1);
    const uint dim3 = get_global_id(2);

    const uint dim_size0 = INPUT_LOWER_PADDING[CALCULATION_ORDER[0]] + SIZE[CALCULATION_ORDER[0]] + INPUT_UPPER_PADDING[CALCULATION_ORDER[0]];
    const uint dim_size1 = INPUT_LOWER_PADDING[CALCULATION_ORDER[1]] + SIZE[CALCULATION_ORDER[1]] + INPUT_UPPER_PADDING[CALCULATION_ORDER[1]];
    const uint dim_size2 = INPUT_LOWER_PADDING[CALCULATION_ORDER[2]] + get_global_size(1) + INPUT_UPPER_PADDING[CALCULATION_ORDER[2]];

    uint pos[DIMENSIONS]; // position in each of dimensions
    pos[CALCULATION_ORDER[DIMENSIONS-1]] = dim3;
    pos[CALCULATION_ORDER[DIMENSIONS-2]] = dim2;
    uint pos1D = global_id_0;
    for(uint i = 0; i < DIMENSIONS-2; i++)
    {
        uint order_idx = CALCULATION_ORDER[i];
        pos[order_idx] = pos1D % SIZE[order_idx];
        pos1D /= SIZE[order_idx];
    }

    uint output_pos = FUNC_CALL(OUT_FORMAT)(SIZE, pos, OUTPUT_LOWER_PADDING, OUTPUT_UPPER_PADDING);
    uint input_idx = (((dim3 + INPUT_LOWER_PADDING[CALCULATION_ORDER[3]]) * dim_size2 + (dim2 + INPUT_LOWER_PADDING[CALCULATION_ORDER[2]])) * dim_size1 +
                     (dim1 + INPUT_LOWER_PADDING[CALCULATION_ORDER[1]])) * dim_size0 + dim0 + INPUT_LOWER_PADDING[CALCULATION_ORDER[0]];
    output[output_pos] = SRC_DEST_TYPE_CVT_FUNC(input[input_idx]);
}

#undef SRC_DEST_TYPE_CVT_FUNC
#undef TYPE_CVT_FUNC2
#undef TYPE_CVT_FUNC3