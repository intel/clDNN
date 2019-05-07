// Copyright (c) 2018-2019 Intel Corporation
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
#include "include/unit_type.cl"

#define vec_t MAKE_VECTOR_TYPE(UNIT_TYPE, X_BLOCK_SIZE)
#define uint_vec_t MAKE_VECTOR_TYPE(uint, X_BLOCK_SIZE)

#if   defined MAX_POOLING
    #define UNIT_INIT_VAL UNIT_VAL_MIN
#elif defined AVG_POOLING
    #define UNIT_INIT_VAL UNIT_VAL_ZERO
#else
    #error pooling_gpu_blocked.cl - Unsupported pooling mode.
#endif

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
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

    // Input offset calculations:
    const int input_x_pitch = IC_BLOCK;
    const int input_y_pitch = input_x_pitch * (INPUT0_PAD_BEFORE_SIZE_X +  INPUT0_SIZE_X + INPUT0_PAD_AFTER_SIZE_X);
    const int input_fs_pitch = input_y_pitch * (INPUT0_PAD_BEFORE_SIZE_Y +  INPUT0_SIZE_Y + INPUT0_PAD_AFTER_SIZE_Y);
    const int input_total_f_size = INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM;
    const int input_b_pitch = input_fs_pitch * ((input_total_f_size + IC_BLOCK - 1) / IC_BLOCK);

    const uint input_fs_pad_before = INPUT0_PAD_BEFORE_FEATURE_NUM / IC_BLOCK;

    const int input_offset = b * input_b_pitch +
                             (input_y + INPUT0_PAD_BEFORE_SIZE_Y) * input_y_pitch +
                             (input_x + INPUT0_PAD_BEFORE_SIZE_X) * input_x_pitch +
                             (f_block + input_fs_pad_before) * input_fs_pitch;

    // Output offset calculations:
    const uint output_x_pitch = IC_BLOCK;
    const uint output_y_pitch = output_x_pitch * (OUTPUT_PAD_BEFORE_SIZE_X +  OUTPUT_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X);
    const uint output_total_f_size = OUTPUT_PAD_BEFORE_FEATURE_NUM + OUTPUT_FEATURE_NUM + OUTPUT_PAD_AFTER_FEATURE_NUM;
    const uint output_fs_pitch = output_y_pitch * (OUTPUT_PAD_BEFORE_SIZE_Y +  OUTPUT_SIZE_Y + OUTPUT_PAD_AFTER_SIZE_Y);
    const uint output_b_pitch = output_fs_pitch * ((output_total_f_size + IC_BLOCK - 1) / IC_BLOCK);

    const uint output_fs_pad_before = OUTPUT_PAD_BEFORE_FEATURE_NUM / IC_BLOCK;

    const int dst_index = b * output_b_pitch +
                          INPUT0_PAD_BEFORE_SIZE_Y * input_y_pitch +
                          INPUT0_PAD_BEFORE_SIZE_X * input_x_pitch +
                          (f_block + output_fs_pad_before) * output_fs_pitch;

    vec_t dst = (vec_t)UNIT_INIT_VAL;

#if AVG_POOLING && DYNAMIC_KERNEL_DIVIDER
    uint count;
    if (lid < X_BLOCK_SIZE)
    {
        int x_min = max(0, input_x + lid);
        int x_max = min(input_x + lid + POOL_SIZE_X, INPUT0_SIZE_X);
        int y_min = max(0, input_y);
        int y_max = min(input_y + POOL_SIZE_Y, INPUT0_SIZE_Y);

        count = (y_max - y_min) * (x_max - x_min);
    }

    uint_vec_t scale = (uint_vec_t)(intel_sub_group_shuffle(count, 0),
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
                line_cache[i] = UNIT_BLOCK_READ(input, input_offset + kh * input_y_pitch + i * IC_BLOCK);
            else
                line_cache[i] = UNIT_INIT_VAL;
        }

        __attribute__((opencl_unroll_hint(POOL_SIZE_X)))
        for (int kw = 0; kw < POOL_SIZE_X; kw++)
        {
            vec_t src;
            for (int i = 0; i < X_BLOCK_SIZE; i++) {
                src[i] = line_cache[kw + STRIDE_SIZE_X * i];
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
    dst = ACTIVATION((dst / scale), NL_M ,NL_N);
#elif defined AVG_POOLING && FIXED_KERNEL_DIVIDER
    dst = ACTIVATION((dst / (POOL_SIZE_X * POOL_SIZE_Y)), NL_M ,NL_N);
#endif

    if (x + X_BLOCK_SIZE <= OUTPUT_SIZE_X)
    {
        UNIT_BLOCK_WRITE8(output, dst_index + y * output_y_pitch + x * output_x_pitch, dst);
    }
    else
    {
        // TODO Add case for not full feature slice
        const int x_tail = OUTPUT_SIZE_X - x;
        for (int i = 0; i < x_tail; i++)
            UNIT_BLOCK_WRITE(output, dst_index + (x+i) * output_x_pitch + y * output_y_pitch, dst[i]);
    }


}

#undef UNIT_INIT_VAL
