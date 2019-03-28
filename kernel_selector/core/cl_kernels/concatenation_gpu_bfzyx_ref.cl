// Copyright (c) 2017 Intel Corporation
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


#include "include/include_all.cl"

KERNEL (concatenation_gpu_ref)(__global UNIT_TYPE* input, __global UNIT_TYPE* output, uint output_offset_in_concat_axis)
{
    const uint x = get_global_id(0) % INPUT0_SIZE_X;
    const uint y = get_global_id(0) / INPUT0_SIZE_X;
    const uint z = get_global_id(1);
    const uint f = get_global_id(2) % INPUT0_FEATURE_NUM;
    const uint b = get_global_id(2) / INPUT0_FEATURE_NUM;
    
    uint input_offset  = GET_3D_DATA_INDEX(INPUT0, b, f, z, y, x);
    uint output_offset = GET_3D_DATA_INDEX(OUTPUT, b, f, z, y, x) + output_offset_in_concat_axis*OUTPUT_PITCHES[CONCAT_AXIS_INDEX];

    output[output_offset] = ACTIVATION(input[input_offset], NL_M, NL_N);
}
