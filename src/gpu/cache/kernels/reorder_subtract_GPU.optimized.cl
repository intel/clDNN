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

#if SUBTRACT_SRC_TYPE_CVT
    #define SUBTRACT_SRC_TYPE_CVT_FUNC(val) TYPE_CVT_FUNC2(val, SRC_TYPE)
#else
    #define SUBTRACT_SRC_TYPE_CVT_FUNC(val) val
#endif

uint FUNC(OUT_FORMAT)(uint size[DIMENSIONS], uint pos[DIMENSIONS], uint lpad[DIMENSIONS], uint upad[DIMENSIONS]) {
    OUT_FORMAT_IMPLEMENTATION
}
uint FUNC(SUBTRACT_FORMAT)(uint size[DIMENSIONS], uint pos[DIMENSIONS], uint lpad[DIMENSIONS], uint upad[DIMENSIONS]) {

    SUBTRACT_FORMAT_IMPLEMENTATION
}
KERNEL (reorder_subtract_GPU)(const __global SRC_TYPE* input, __global DEST_TYPE* output, const __global SUBTRACT_TYPE* subtract)
{
    const uint global_id_0 = get_global_id(0);
    const uint global_id_1 = get_global_id(1);
    const uint global_id_2 = get_global_id(2);
    const uint global_size_1 = get_global_size(1);
    const uint global_size_0 = get_global_size(0);

    uint pos[DIMENSIONS]; // position in each of dimensions
    pos[CALCULATION_ORDER[DIMENSIONS-1]] = global_id_2;
    pos[CALCULATION_ORDER[DIMENSIONS-2]] = global_id_1;
    uint pos1D = global_id_0;
    for(uint i = 0; i < DIMENSIONS-2; i++)
    {
        uint order_idx = CALCULATION_ORDER[i];
        pos[order_idx] = pos1D % SIZE[order_idx];
        pos1D /= SIZE[order_idx];
    }

    uint output_pos = FUNC_CALL(OUT_FORMAT)(SIZE, pos, LOWER_PADDING, UPPER_PADDING);
    // We set it to 0 because we subtract the same values from every input batch
    pos[0] = 0;
    uint subtract_pos = FUNC_CALL(SUBTRACT_FORMAT)(SIZE, pos, SUBTRACT_LOWER_PADDING, SUBTRACT_UPPER_PADDING);
    uint input_idx = (global_id_2 * global_size_1 + global_id_1) * global_size_0 + global_id_0;
    output[output_pos] = SRC_DEST_TYPE_CVT_FUNC(input[input_idx] - SUBTRACT_SRC_TYPE_CVT_FUNC(subtract[subtract_pos]));
}

#undef SUBTRACT_SRC_TYPE_CVT_FUNC
#undef SRC_DEST_TYPE_CVT_FUNC
#undef TYPE_CVT_FUNC2
#undef TYPE_CVT_FUNC3