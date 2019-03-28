// Copyright (c) 2018 Intel Corporation
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

#pragma OPENCL EXTENSION  cl_intel_subgroups : enable
#pragma OPENCL EXTENSION  cl_intel_subgroups_short : enable

    // Block read - currently block is 4 bytes aligned.
#if FP16_UNIT_USED == 1

#define GET_SRC(data, id) as_half(intel_sub_group_shuffle(as_ushort(src), id))
#define GET_SRC8(data, id) as_half8(intel_sub_group_shuffle(as_ushort8(src), id))
#define ALIGNED_BLOCK_READ(ptr, byte_offset) as_half(intel_sub_group_block_read_us((const __global ushort*)(ptr) + (byte_offset)))
#define ALIGNED_BLOCK_WRITE(ptr, byte_offset, val) intel_sub_group_block_write_us((__global ushort*)(ptr) + (byte_offset), as_ushort(val))

#define ALIGNED_BLOCK_READ2(ptr, byte_offset) as_half2(intel_sub_group_block_read_us2((const __global ushort*)(ptr) + (byte_offset)))
#define ALIGNED_BLOCK_READ8(ptr, byte_offset) as_half8(intel_sub_group_block_read_us8((const __global ushort*)(ptr) + (byte_offset)))
#define ALIGNED_BLOCK_WRITE8(ptr, byte_offset, val) intel_sub_group_block_write_us8((__global ushort*)(ptr) + (byte_offset), as_ushort8(val))
#else
#define GET_SRC(data, id) intel_sub_group_shuffle(src, id)
#define GET_SRC8(data, id) intel_sub_group_shuffle(src, id)
#define ALIGNED_BLOCK_READ(ptr, byte_offset) as_float(intel_sub_group_block_read((const __global uint*)(ptr) + (byte_offset)))
#define ALIGNED_BLOCK_WRITE(ptr, byte_offset, val) intel_sub_group_block_write((__global uint*)(ptr) + (byte_offset), as_uint(val))

#define ALIGNED_BLOCK_READ2(ptr, byte_offset) as_float2(intel_sub_group_block_read2((const __global uint*)(ptr) + (byte_offset)))
#define ALIGNED_BLOCK_READ8(ptr, byte_offset) as_float8(intel_sub_group_block_read8((const __global uint*)(ptr) + (byte_offset)))
#define ALIGNED_BLOCK_WRITE8(ptr, byte_offset, val) intel_sub_group_block_write8((__global uint*)(ptr) + (byte_offset), as_uint8(val))
#endif

#define vec_t MAKE_VECTOR_TYPE(UNIT_TYPE, X_BLOCK_SIZE)

#if   defined MAX_POOLING
    #define UNIT_INIT_VAL UNIT_VAL_MIN
#elif defined AVG_POOLING
    #define UNIT_INIT_VAL UNIT_VAL_ZERO
#else
#error
#endif

KERNEL(pooling_gpu_blocked)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{

    const int lid = get_sub_group_local_id();
    const int f_block = get_group_id(1);
    const int b = get_global_id(2);

    const int xy = get_global_id(0);
    const int x = (xy % X_BLOCKS) * X_BLOCK_SIZE;
    const int y = xy / X_BLOCKS;

    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;
    const int input_offset = b*INPUT0_BATCH_PITCH*IC_BLOCK + INPUT0_OFFSET*IC_BLOCK + input_y*IC_BLOCK*INPUT0_Y_PITCH + input_x*IC_BLOCK + f_block*IC_BLOCK*INPUT0_FEATURE_PITCH;

    const int dst_index = OUTPUT_OFFSET*IC_BLOCK + f_block*IC_BLOCK*OUTPUT_FEATURE_PITCH;

    vec_t dst = (vec_t)UNIT_INIT_VAL;

#if AVG_POOLING && DYNAMIC_KERNEL_DIVIDER
    UNIT_TYPE count;
    if (lid < X_BLOCK_SIZE)
    {
        int x_min = max(0, input_x + lid);
        int x_max = min(input_x + lid + POOL_SIZE_X, INPUT0_SIZE_X);
        int y_min = max(0, input_y);
        int y_max = min(input_y + POOL_SIZE_Y, INPUT0_SIZE_Y);

        count = (UNIT_TYPE)(1.f / (float)((y_max - y_min) * (x_max - x_min)));
    }

    vec_t scale = (vec_t)(intel_sub_group_shuffle(count, 0),
                          intel_sub_group_shuffle(count, 1),
                          intel_sub_group_shuffle(count, 2),
                          intel_sub_group_shuffle(count, 3),
                          intel_sub_group_shuffle(count, 4),
                          intel_sub_group_shuffle(count, 5),
                          intel_sub_group_shuffle(count, 6),
                          intel_sub_group_shuffle(count, 7));
#endif

    for (int kh = 0; kh < POOL_SIZE_Y; kh++)
    {
        if (input_y + kh < 0 || input_y + kh >= INPUT0_SIZE_Y)
            continue;

        UNIT_TYPE line_cache[INPUT_LINE_SIZE];
        for (int i = 0; i < INPUT_LINE_SIZE; i++)
        {
            if ((input_x + i) >= 0 && (input_x + i) < INPUT0_SIZE_X)
                line_cache[i] = ALIGNED_BLOCK_READ(input, input_offset + kh*IC_BLOCK*INPUT0_Y_PITCH + i*IC_BLOCK);
            else
                line_cache[i] = UNIT_INIT_VAL;
        }

        __attribute__((opencl_unroll_hint(POOL_SIZE_X)))
        for (int kw = 0; kw < POOL_SIZE_X; kw++)
        {
            vec_t src;
            for (int i = 0; i < X_BLOCK_SIZE; i++) {
                src[i] = line_cache[kw + STRIDE_SIZE_X*i];
            }


#if defined MAX_POOLING
            dst = max(dst, src);
#elif defined AVG_POOLING
            dst += src;
#endif
        }
    }

#if defined MAX_POOLING
    dst = ACTIVATION(dst, NL_M, NL_N);
#elif defined AVG_POOLING && DYNAMIC_KERNEL_DIVIDER
    dst = ACTIVATION((dst*scale), NL_M ,NL_N);
#elif defined AVG_POOLING && FIXED_KERNEL_DIVIDER
    dst = ACTIVATION((dst/(POOL_SIZE_X*POOL_SIZE_Y)), NL_M ,NL_N);
#endif
    if (x + X_BLOCK_SIZE <= OUTPUT_SIZE_X)
    {
        ALIGNED_BLOCK_WRITE8(output, dst_index + y*OUTPUT_Y_PITCH*IC_BLOCK + x*IC_BLOCK, dst);
    }
    else
    {
        const int x_tail = OUTPUT_SIZE_X - x;
        for (int i = 0; i < x_tail; i++)
            ALIGNED_BLOCK_WRITE(output, dst_index + (x+i)*IC_BLOCK + y*IC_BLOCK*OUTPUT_Y_PITCH, dst[i]);
    }


}

#undef UNIT_INIT_VAL
