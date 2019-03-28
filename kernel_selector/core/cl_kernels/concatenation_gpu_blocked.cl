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


#include "include/include_all.cl"

//
// In this kernel we are processing "fyx" as flatten 1D "elements".
// As long as we can we use block read/write.
// For last SIMD in which we have to write only partial data we use normal read/write to buffer.
//

// must be 8 as long as we use block_read8/write8
#define ELEMENTS_PER_WORK_ITEM 8
#define WORK_GROUP_SIZE 16
#define INPUT0_ELEMENTS_COUNT (INPUT0_LENGTH/INPUT0_BATCH_NUM)
#define IC_BLOCK 16

#if FP16_UNIT_USED
    #define ALIGNED_BLOCK_READ(ptr, byte_offset) as_half(intel_sub_group_block_read_us((const __global ushort*)(ptr) + (byte_offset)))
    #define ALIGNED_BLOCK_READ8(ptr, byte_offset) as_half8(intel_sub_group_block_read_us8((const __global ushort*)(ptr) + (byte_offset)))
    #define ALIGNED_BLOCK_WRITE(ptr, byte_offset, val) intel_sub_group_block_write_us((__global ushort*)(ptr) + (byte_offset), as_ushort(val))
    #define ALIGNED_BLOCK_WRITE8(ptr, byte_offset, val) intel_sub_group_block_write_us8((__global ushort*)(ptr) + (byte_offset), as_ushort8(val))
#else
    #define ALIGNED_BLOCK_READ(ptr, byte_offset) as_float(intel_sub_group_block_read((const __global uint*)(ptr) + (byte_offset)))
    #define ALIGNED_BLOCK_READ8(ptr, byte_offset) as_float8(intel_sub_group_block_read8((const __global uint*)(ptr) + (byte_offset)))
    #define ALIGNED_BLOCK_WRITE(ptr, byte_offset, val) intel_sub_group_block_write((__global uint*)(ptr) + (byte_offset), as_uint(val))
    #define ALIGNED_BLOCK_WRITE8(ptr, byte_offset, val) intel_sub_group_block_write8((__global uint*)(ptr) + (byte_offset), as_uint8(val))
#endif
    
__attribute__((reqd_work_group_size(1, WORK_GROUP_SIZE, 1)))
__attribute__((intel_reqd_sub_group_size(WORK_GROUP_SIZE)))
KERNEL (concatenation_gpu_blocked)(__global UNIT_TYPE* input, __global UNIT_TYPE* output, uint output_offset_in_concat_axis)
{
    const int b = get_global_id(0);
    const int f_block = get_group_id(1);
    const int xy = get_global_id(2);
    const int lid = get_sub_group_local_id();

    const int x = xy % OUTPUT_SIZE_X;
    const int y = xy / OUTPUT_SIZE_X;
    const uint input_offset = b*INPUT0_BATCH_PITCH*IC_BLOCK + INPUT0_OFFSET*IC_BLOCK + IC_BLOCK*f_block*INPUT0_FEATURE_PITCH + y*INPUT0_Y_PITCH*IC_BLOCK + x*INPUT0_X_PITCH*IC_BLOCK;

    const uint dst_index = b*OUTPUT_BATCH_PITCH*IC_BLOCK + OUTPUT_OFFSET*IC_BLOCK + (f_block*IC_BLOCK + output_offset_in_concat_axis)*OUTPUT_FEATURE_PITCH
             + y*OUTPUT_Y_PITCH*IC_BLOCK + x*OUTPUT_X_PITCH*IC_BLOCK;

    UNIT_TYPE src = ALIGNED_BLOCK_READ(input, input_offset);
    src = ACTIVATION(src, NL_M, NL_N);
    ALIGNED_BLOCK_WRITE(output, dst_index, src);
}

#undef INPUT0_ELEMENTS_COUNT
#undef WORK_GROUP_SIZE
#undef ELEMENTS_PER_WORK_ITEM
#undef ALIGNED_BLOCK_READ8
#undef ALIGNED_BLOCK_WRITE8
