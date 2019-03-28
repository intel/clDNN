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

#define FEATURE_SLICE_SIZE 16
#define XY_BLOCK 8

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(1, SUB_GROUP_SIZE, 1)))
KERNEL(convolution_depthwise)(
    __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global FILTER_TYPE* weights,
#if BIAS_TERM
    __global BIAS_TYPE* biases,
#endif
    uint split_idx)
{
    const uint yx = get_global_id(0);
    const uint x = (yx % XY_BLOCKS) * XY_BLOCK;
    const uint y = (yx / XY_BLOCKS);
    const uint f_block = get_group_id(1);
    const int lid = get_local_id(1);
    const uint b = get_global_id(2);

    const uint filter_offset = f_block * IC_BLOCK * FILTER_OFM_PITCH;

    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    const uint input_offset = b*INPUT0_BATCH_PITCH*IC_BLOCK + INPUT0_OFFSET*IC_BLOCK + f_block*IC_BLOCK*INPUT0_FEATURE_PITCH;

#if BIAS_TERM
    UNIT_TYPE8 blockC00 = (UNIT_TYPE8)(UNIT_BLOCK_READ(biases, f_block * FEATURE_SLICE_SIZE));
#else
    UNIT_TYPE8 blockC00 = (UNIT_TYPE8)(UNIT_VAL_ZERO);
 #endif

    UNIT_TYPE wei_00 = UNIT_BLOCK_READ(weights, filter_offset + 0*FILTER_Y_PITCH*IC_BLOCK + 0*IC_BLOCK);
    UNIT_TYPE wei_01 = UNIT_BLOCK_READ(weights, filter_offset + 0*FILTER_Y_PITCH*IC_BLOCK + 1*IC_BLOCK);
    UNIT_TYPE wei_02 = UNIT_BLOCK_READ(weights, filter_offset + 0*FILTER_Y_PITCH*IC_BLOCK + 2*IC_BLOCK);
    UNIT_TYPE wei_10 = UNIT_BLOCK_READ(weights, filter_offset + 1*FILTER_Y_PITCH*IC_BLOCK + 0*IC_BLOCK);
    UNIT_TYPE wei_11 = UNIT_BLOCK_READ(weights, filter_offset + 1*FILTER_Y_PITCH*IC_BLOCK + 1*IC_BLOCK);
    UNIT_TYPE wei_12 = UNIT_BLOCK_READ(weights, filter_offset + 1*FILTER_Y_PITCH*IC_BLOCK + 2*IC_BLOCK);
    UNIT_TYPE wei_20 = UNIT_BLOCK_READ(weights, filter_offset + 2*FILTER_Y_PITCH*IC_BLOCK + 0*IC_BLOCK);
    UNIT_TYPE wei_21 = UNIT_BLOCK_READ(weights, filter_offset + 2*FILTER_Y_PITCH*IC_BLOCK + 1*IC_BLOCK);
    UNIT_TYPE wei_22 = UNIT_BLOCK_READ(weights, filter_offset + 2*FILTER_Y_PITCH*IC_BLOCK + 2*IC_BLOCK);

#if STRIDE_SIZE_X == 1
    UNIT_TYPE8 src_block_0 = UNIT_BLOCK_READ8(input, input_offset + (input_y + 0)*IC_BLOCK*INPUT0_Y_PITCH + (input_x)*IC_BLOCK);
    UNIT_TYPE8 src_block_1 = UNIT_BLOCK_READ8(input, input_offset + (input_y + 1)*IC_BLOCK*INPUT0_Y_PITCH + (input_x)*IC_BLOCK);
    UNIT_TYPE8 src_block_2 = UNIT_BLOCK_READ8(input, input_offset + (input_y + 2)*IC_BLOCK*INPUT0_Y_PITCH + (input_x)*IC_BLOCK);
    UNIT_TYPE src_tail_00 = UNIT_BLOCK_READ(input, input_offset + (input_y + 0)*IC_BLOCK*INPUT0_Y_PITCH + (input_x + 8)*IC_BLOCK);
    UNIT_TYPE src_tail_01 = UNIT_BLOCK_READ(input, input_offset + (input_y + 0)*IC_BLOCK*INPUT0_Y_PITCH + (input_x + 9)*IC_BLOCK);
    UNIT_TYPE src_tail_10 = UNIT_BLOCK_READ(input, input_offset + (input_y + 1)*IC_BLOCK*INPUT0_Y_PITCH + (input_x + 8)*IC_BLOCK);
    UNIT_TYPE src_tail_11 = UNIT_BLOCK_READ(input, input_offset + (input_y + 1)*IC_BLOCK*INPUT0_Y_PITCH + (input_x + 9)*IC_BLOCK);
    UNIT_TYPE src_tail_20 = UNIT_BLOCK_READ(input, input_offset + (input_y + 2)*IC_BLOCK*INPUT0_Y_PITCH + (input_x + 8)*IC_BLOCK);
    UNIT_TYPE src_tail_21 = UNIT_BLOCK_READ(input, input_offset + (input_y + 2)*IC_BLOCK*INPUT0_Y_PITCH + (input_x + 9)*IC_BLOCK);

    for (int i = 0; i < XY_BLOCK - 2; i++)
    {
        blockC00[i] = mad(src_block_0[i + 0], wei_00,blockC00[i]);
        blockC00[i] = mad(src_block_0[i + 1], wei_01,blockC00[i]);
        blockC00[i] = mad(src_block_0[i + 2], wei_02,blockC00[i]);

        blockC00[i] = mad(src_block_1[i + 0], wei_10,blockC00[i]);
        blockC00[i] = mad(src_block_1[i + 1], wei_11,blockC00[i]);
        blockC00[i] = mad(src_block_1[i + 2], wei_12,blockC00[i]);

        blockC00[i] = mad(src_block_2[i + 0], wei_20,blockC00[i]);
        blockC00[i] = mad(src_block_2[i + 1], wei_21,blockC00[i]);
        blockC00[i] = mad(src_block_2[i + 2], wei_22,blockC00[i]);
    }
    {
        blockC00[6] = mad(src_block_0[6], wei_00,blockC00[6]);
        blockC00[6] = mad(src_block_0[7], wei_01,blockC00[6]);
        blockC00[6] = mad(src_tail_00, wei_02,blockC00[6]);

        blockC00[6] = mad(src_block_1[6], wei_10,blockC00[6]);
        blockC00[6] = mad(src_block_1[7], wei_11,blockC00[6]);
        blockC00[6] = mad(src_tail_10, wei_12,blockC00[6]);

        blockC00[6] = mad(src_block_2[6], wei_20,blockC00[6]);
        blockC00[6] = mad(src_block_2[7], wei_21,blockC00[6]);
        blockC00[6] = mad(src_tail_20, wei_22,blockC00[6]);
    }

    {
        blockC00[7] = mad(src_block_0[7], wei_00, blockC00[7]);
        blockC00[7] = mad(src_tail_00, wei_01, blockC00[7]);
        blockC00[7] = mad(src_tail_01, wei_02, blockC00[7]);

        blockC00[7] = mad(src_block_1[7], wei_10, blockC00[7]);
        blockC00[7] = mad(src_tail_10, wei_11, blockC00[7]);
        blockC00[7] = mad(src_tail_11, wei_12, blockC00[7]);

        blockC00[7] = mad(src_block_2[7], wei_20, blockC00[7]);
        blockC00[7] = mad(src_tail_20, wei_21, blockC00[7]);
        blockC00[7] = mad(src_tail_21, wei_22, blockC00[7]);
    }

#else
    UNIT_TYPE8 src_block_00 = UNIT_BLOCK_READ8(input, input_offset + (input_y + 0) * IC_BLOCK * INPUT0_Y_PITCH + (input_x)*IC_BLOCK);
    UNIT_TYPE8 src_block_01 = UNIT_BLOCK_READ8(input, input_offset + (input_y + 0) * IC_BLOCK * INPUT0_Y_PITCH + (input_x + 8)*IC_BLOCK);
    UNIT_TYPE8 src_block_10 = UNIT_BLOCK_READ8(input, input_offset + (input_y + 1) * IC_BLOCK * INPUT0_Y_PITCH + (input_x)*IC_BLOCK);
    UNIT_TYPE8 src_block_11 = UNIT_BLOCK_READ8(input, input_offset + (input_y + 1) * IC_BLOCK * INPUT0_Y_PITCH + (input_x + 8)*IC_BLOCK);
    UNIT_TYPE8 src_block_20 = UNIT_BLOCK_READ8(input, input_offset + (input_y + 2) * IC_BLOCK * INPUT0_Y_PITCH + (input_x)*IC_BLOCK);
    UNIT_TYPE8 src_block_21 = UNIT_BLOCK_READ8(input, input_offset + (input_y + 2) * IC_BLOCK * INPUT0_Y_PITCH + (input_x + 8)*IC_BLOCK);
    UNIT_TYPE src_tail_0 = UNIT_BLOCK_READ(input, input_offset + (input_y + 0)*IC_BLOCK*INPUT0_Y_PITCH + (input_x + 16)*IC_BLOCK);
    UNIT_TYPE src_tail_1 = UNIT_BLOCK_READ(input, input_offset + (input_y + 1)*IC_BLOCK*INPUT0_Y_PITCH + (input_x + 16)*IC_BLOCK);
    UNIT_TYPE src_tail_2 = UNIT_BLOCK_READ(input, input_offset + (input_y + 2)*IC_BLOCK*INPUT0_Y_PITCH + (input_x + 16)*IC_BLOCK);

    for (int i = 0; i < 3; i++)
    {
        blockC00[i] = mad(src_block_00[2*i + 0], wei_00,blockC00[i]);
        blockC00[i] = mad(src_block_00[2*i + 1], wei_01,blockC00[i]);
        blockC00[i] = mad(src_block_00[2*i + 2], wei_02,blockC00[i]);
        blockC00[i] = mad(src_block_10[2*i + 0], wei_10,blockC00[i]);
        blockC00[i] = mad(src_block_10[2*i + 1], wei_11,blockC00[i]);
        blockC00[i] = mad(src_block_10[2*i + 2], wei_12,blockC00[i]);
        blockC00[i] = mad(src_block_20[2*i + 0], wei_20,blockC00[i]);
        blockC00[i] = mad(src_block_20[2*i + 1], wei_21,blockC00[i]);
        blockC00[i] = mad(src_block_20[2*i + 2], wei_22,blockC00[i]);
    }
    {
        blockC00[3] = mad(src_block_00[6], wei_00,blockC00[3]);
        blockC00[3] = mad(src_block_00[7], wei_01,blockC00[3]);
        blockC00[3] = mad(src_block_01[0], wei_02,blockC00[3]);

        blockC00[3] = mad(src_block_10[6], wei_10,blockC00[3]);
        blockC00[3] = mad(src_block_10[7], wei_11,blockC00[3]);
        blockC00[3] = mad(src_block_11[0], wei_12,blockC00[3]);

        blockC00[3] = mad(src_block_20[6], wei_20,blockC00[3]);
        blockC00[3] = mad(src_block_20[7], wei_21,blockC00[3]);
        blockC00[3] = mad(src_block_21[0], wei_22,blockC00[3]);
    }

    for (int i = 0; i < 3; i++)
    {
        blockC00[4+i] = mad(src_block_01[2*i + 0], wei_00,blockC00[4+i]);
        blockC00[4+i] = mad(src_block_01[2*i + 1], wei_01,blockC00[4+i]);
        blockC00[4+i] = mad(src_block_01[2*i + 2], wei_02,blockC00[4+i]);

        blockC00[4+i] = mad(src_block_11[2*i + 0], wei_10,blockC00[4+i]);
        blockC00[4+i] = mad(src_block_11[2*i + 1], wei_11,blockC00[4+i]);
        blockC00[4+i] = mad(src_block_11[2*i + 2], wei_12,blockC00[4+i]);

        blockC00[4+i] = mad(src_block_21[2*i + 0], wei_20,blockC00[4+i]);
        blockC00[4+i] = mad(src_block_21[2*i + 1], wei_21,blockC00[4+i]);
        blockC00[4+i] = mad(src_block_21[2*i + 2], wei_22,blockC00[4+i]);
    }
    {
        blockC00[7] = mad(src_block_01[6], wei_00,blockC00[7]);
        blockC00[7] = mad(src_block_01[7], wei_01,blockC00[7]);
        blockC00[7] = mad(src_tail_0, wei_02,blockC00[7]);

        blockC00[7] = mad(src_block_11[6], wei_10,blockC00[7]);
        blockC00[7] = mad(src_block_11[7], wei_11,blockC00[7]);
        blockC00[7] = mad(src_tail_1, wei_12,blockC00[7]);

        blockC00[7] = mad(src_block_21[6], wei_20,blockC00[7]);
        blockC00[7] = mad(src_block_21[7], wei_21,blockC00[7]);
        blockC00[7] = mad(src_tail_2, wei_22,blockC00[7]);
    }
#endif // STRIDE_SIZE_X == 1

    blockC00 = ACTIVATION(blockC00, NL_M, NL_N);
    const uint dst_index = f_block * IC_BLOCK * OUTPUT_FEATURE_PITCH + y * IC_BLOCK * OUTPUT_Y_PITCH;

    if (x + XY_BLOCK <= OUTPUT_SIZE_X)
    {
        UNIT_BLOCK_WRITE8(output, dst_index + x*IC_BLOCK, blockC00);
    }
    else
    {
        for (int i = 0; i < (OUTPUT_SIZE_X - x); i++)
        {
            UNIT_BLOCK_WRITE(output, dst_index + (x+i)*IC_BLOCK, blockC00[i]);
        }
    }
}

#undef FEATURE_SLICE_SIZE
#undef XY_BLOCK
