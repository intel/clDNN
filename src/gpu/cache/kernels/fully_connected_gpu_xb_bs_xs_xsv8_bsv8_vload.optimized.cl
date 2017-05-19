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


#if RELU
    #define ACTIVATION(output, input) output = isinf(NEGATIVE_SLOPE) ? ((input >= 0.0f) ? \
    input : -NEGATIVE_SLOPE) : (max(input, 0.0f) + NEGATIVE_SLOPE * min(input, 0.0f));
#else
    #define ACTIVATION(output, input) output = input;
#endif

#define MULTIPLY_BLOCKS_8x8(_result, _blockA, _blockB)  \
{   \
    const float8 acol0 = TRANSPOSE_BLOCK_8( _blockA.s0 ); \
    const float8 acol1 = TRANSPOSE_BLOCK_8( _blockA.s1 ); \
    const float8 acol2 = TRANSPOSE_BLOCK_8( _blockA.s2 ); \
    const float8 acol3 = TRANSPOSE_BLOCK_8( _blockA.s3 ); \
    const float8 acol4 = TRANSPOSE_BLOCK_8( _blockA.s4 ); \
    const float8 acol5 = TRANSPOSE_BLOCK_8( _blockA.s5 ); \
    const float8 acol6 = TRANSPOSE_BLOCK_8( _blockA.s6 ); \
    const float8 acol7 = TRANSPOSE_BLOCK_8( _blockA.s7 ); \
    _result = mad( _blockB.s0, acol0, _result ); \
    _result = mad( _blockB.s1, acol1, _result ); \
    _result = mad( _blockB.s2, acol2, _result ); \
    _result = mad( _blockB.s3, acol3, _result ); \
    _result = mad( _blockB.s4, acol4, _result ); \
    _result = mad( _blockB.s5, acol5, _result ); \
    _result = mad( _blockB.s6, acol6, _result ); \
    _result = mad( _blockB.s7, acol7, _result ); \
}

#define SUB_GROUP_SIZE 8

__attribute__((reqd_work_group_size(SUB_GROUP_SIZE, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
KERNEL (fully_connected_gpu_xb_bs_xs_xsv8_bsv8_vload)(
    const __global float* input,
    __global float* output,
    const __global float* weight
#if BIAS_TERM
    , __global UNIT_TYPE* bias)
#else
    )
#endif
{
    const uint global_id = get_global_id(0);
    const uint group_id = get_group_id(0);
    const uint batch_group_id = get_global_id(1); // which part of batches we are computing, for example for batch 64 we compute batches 0..31 for batch_group_id == 0 and batches 32..65 for batch_group_id == 1
    const uint id_in_sub_group = get_sub_group_local_id();

    const uint out_id = (id_in_sub_group * BATCHES_PER_WORK_ITEM * (uint)get_global_size(1)) / SUB_GROUP_SIZE + group_id * BATCHES_PER_WORK_ITEM * NEURONS_PER_WORK_ITEM * (uint)get_global_size(1) + (BATCHES_PER_WORK_ITEM * batch_group_id) / SUB_GROUP_SIZE;

    uint neuronIdx = id_in_sub_group + group_id * SUB_GROUP_SIZE * NEURONS_PER_WORK_ITEM;

    float8 blockC00 = 0.f;
    float8 blockC10 = 0.f;

#if BATCHES_PER_WORK_ITEM >= 16
    float8 blockC01 = 0.f;
    float8 blockC11 = 0.f;
#if BATCHES_PER_WORK_ITEM >= 32
    float8 blockC02 = 0.f;
    float8 blockC12 = 0.f;
    float8 blockC03 = 0.f;
    float8 blockC13 = 0.f;
#endif
#endif

    uint weight_offset = id_in_sub_group + SUB_GROUP_SIZE * group_id * NEURONS_PER_WORK_ITEM * INPUT_ELEMENTS_COUNT;
#if NEURONS_PER_WORK_ITEM > 1

    uint weight_offset2 = weight_offset + SUB_GROUP_SIZE * INPUT_ELEMENTS_COUNT;

#endif // #if NEURONS_PER_WORK_ITEM > 1

    uint input_idx = id_in_sub_group + batch_group_id * BATCHES_PER_WORK_ITEM * INPUT_ELEMENTS_COUNT;
    for(uint h = 0; h < INPUT_ELEMENTS_COUNT / 8; h++)
    {
        // read input data in blocks ( 8 batch * 8 x )
        float8 blockA00 = as_float8(intel_sub_group_block_read8((const __global uint*)input + input_idx));
#if BATCHES_PER_WORK_ITEM >= 16
        float8 blockA01 = as_float8(intel_sub_group_block_read8((const __global uint*)input + (input_idx + (INPUT_ELEMENTS_COUNT*8))));
#if BATCHES_PER_WORK_ITEM >= 32
        float8 blockA02 = as_float8(intel_sub_group_block_read8((const __global uint*)input + (input_idx + (INPUT_ELEMENTS_COUNT*16))));
        float8 blockA03 = as_float8(intel_sub_group_block_read8((const __global uint*)input + (input_idx + (INPUT_ELEMENTS_COUNT*24))));
#endif
#endif

        float8 blockB00 = as_float8(intel_sub_group_block_read8((const __global uint*)weight + weight_offset)); weight_offset += 64;

        MULTIPLY_BLOCKS_8x8(blockC00, blockA00, blockB00)
#if BATCHES_PER_WORK_ITEM >= 16
        MULTIPLY_BLOCKS_8x8(blockC01, blockA01, blockB00)
#if BATCHES_PER_WORK_ITEM >= 32
        MULTIPLY_BLOCKS_8x8(blockC02, blockA02, blockB00)
        MULTIPLY_BLOCKS_8x8(blockC03, blockA03, blockB00)
#endif
#endif

#if NEURONS_PER_WORK_ITEM > 1

        float8 blockB10 = as_float8(intel_sub_group_block_read8((const __global uint*)weight + weight_offset2)); weight_offset2 += 64;

        MULTIPLY_BLOCKS_8x8(blockC10, blockA00, blockB10)
#if BATCHES_PER_WORK_ITEM >= 16
        MULTIPLY_BLOCKS_8x8(blockC11, blockA01, blockB10)
#if BATCHES_PER_WORK_ITEM >= 32
        MULTIPLY_BLOCKS_8x8(blockC12, blockA02, blockB10)
        MULTIPLY_BLOCKS_8x8(blockC13, blockA03, blockB10)
#endif
#endif

#endif // #if NEURONS_PER_WORK_ITEM > 1
        input_idx += 64; // 64 because of input format which have blocks of 64 elements
    }

#if BIAS_TERM
    blockC00 += bias[neuronIdx];
#if BATCHES_PER_WORK_ITEM >= 16
    blockC01 += bias[neuronIdx];
#if BATCHES_PER_WORK_ITEM >= 32
    blockC02 += bias[neuronIdx];
    blockC03 += bias[neuronIdx];
#endif
#endif

#if NEURONS_PER_WORK_ITEM > 1

    blockC10 += bias[neuronIdx+8];
#if BATCHES_PER_WORK_ITEM >= 16
    blockC11 += bias[neuronIdx+8];
#if BATCHES_PER_WORK_ITEM >= 32
    blockC12 += bias[neuronIdx+8];
    blockC13 += bias[neuronIdx+8];
#endif
#endif

#endif // #if NEURONS_PER_WORK_ITEM > 1
#endif // #if BIAS_TERM
    ACTIVATION(blockC00, blockC00);
#if BATCHES_PER_WORK_ITEM >= 16
    ACTIVATION(blockC01, blockC01);
#if BATCHES_PER_WORK_ITEM >= 32
    ACTIVATION(blockC02, blockC02);
    ACTIVATION(blockC03, blockC03);
#endif
#endif

#if NEURONS_PER_WORK_ITEM > 1

    ACTIVATION(blockC10, blockC10);
#if BATCHES_PER_WORK_ITEM >= 16
    ACTIVATION(blockC11, blockC11);
#if BATCHES_PER_WORK_ITEM >= 32
    ACTIVATION(blockC12, blockC12);
    ACTIVATION(blockC13, blockC13);
#endif
#endif

#endif // #if NEURONS_PER_WORK_ITEM > 1

    if(neuronIdx >= OUTPUT_ELEMENTS_COUNT)
        return;

    vstore8(blockC00, out_id, output);
#if BATCHES_PER_WORK_ITEM >= 16
    vstore8(blockC01, out_id + 1, output);
#if BATCHES_PER_WORK_ITEM >= 32
    vstore8(blockC02, out_id + 2, output);
    vstore8(blockC03, out_id + 3, output);
#endif
#endif

#if NEURONS_PER_WORK_ITEM > 1

    if(neuronIdx + 8 >= OUTPUT_ELEMENTS_COUNT)
        return;

    vstore8(blockC10, out_id+INPUT_BATCH_NUM, output);
#if BATCHES_PER_WORK_ITEM >= 16
    vstore8(blockC11, out_id+INPUT_BATCH_NUM+1, output);
#if BATCHES_PER_WORK_ITEM >= 32
    vstore8(blockC12, out_id+INPUT_BATCH_NUM+2, output);
    vstore8(blockC13, out_id+INPUT_BATCH_NUM+3, output);
#endif
#endif

#endif // #if NEURONS_PER_WORK_ITEM > 1
}

#undef SUB_GROUP_SIZE
#undef MULTIPLY_BLOCKS_8x8
#undef ACTIVATION
