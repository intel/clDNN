/*
// Copyright (c) 2016 Intel Corporation
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
*/

#include "include/cnn_common.cl"

#define TILE_M          2
#define TILE_K          KERNEL_WIDTH
#define TILE_N          32

__attribute__((intel_reqd_sub_group_size(8)))
KERNEL(convolution_f32)(
    const __global float *src0,
    __global float *dst,
    const __global float *src1,
#ifdef OUTPUT_BIASED
    const __global float *bias,
#endif
    uint split_idx)
{
#include "include/cnn_common_data_types.cl"

    const unsigned group_x = get_group_id(0);
    const unsigned group_y = get_group_id(1);
    const unsigned global_x = get_global_id(0);
    const unsigned global_y = get_global_id(1);
    const unsigned global_z = get_global_id(2);

    unsigned interleaved_y;
    unsigned kernel_y;
    unsigned kernel_idx;

    // Result ctile (*dst) is M rows x N columns
    // LWG size is 1x8.  Thus each thread calculates 8*M rows x N cols of ctile.
    float8  blockC00 = 0.f;
    float8  blockC10 = 0.f;
    float8  blockC20 = 0.f;
    float8  blockC30 = 0.f;
    float8  blockC01 = 0.f;
    float8  blockC11 = 0.f;
    float8  blockC21 = 0.f;
    float8  blockC31 = 0.f;

    const uint in_split_offset = split_idx * INPUT_FEATURE_PITCH * INPUT_FEATURE_NUM;
    // Src0 (patch input) is directly used as atile.
    // Each work item points to the start of a different patch.
    // atile is M rows x K columns.
    int src0_read_offset0 = INPUT_OFFEST_FOR_PADDED_PART + in_split_offset
     + INPUT_BATCH_PITCH * global_z                                                 // batch offset
     + ( ( ( global_y * TILE_M + 0 ) / OUTPUT_SIZE_X ) * STRIDE_Y * INPUT_Y_PITCH )   // y offset
     + ( ( ( global_y * TILE_M + 0 ) % OUTPUT_SIZE_X ) * STRIDE_X );                    // x offset
    int src0_read_offset1 = INPUT_OFFEST_FOR_PADDED_PART + in_split_offset
     + INPUT_BATCH_PITCH * global_z                                                 // batch offset
     + ( ( ( global_y * TILE_M + 1 ) / OUTPUT_SIZE_X ) * STRIDE_Y * INPUT_Y_PITCH )   // y offset
     + ( ( ( global_y * TILE_M + 1 ) % OUTPUT_SIZE_X ) * STRIDE_X );                    // x offset

    // Src1 (filter) is directly used as btile.
    // It starts at the top of src1 and walks down.
    // btile is K rows x N columns.
    const __global float *src1_read = src1 + ( global_x * TILE_N * 2);

#define DOT_PRODUCT_8( _result, _rowA, colB )    \
    {   \
        _result.s0 = mad( _rowA, sub_group_broadcast( colB,  0 ), _result.s0 );  \
        _result.s1 = mad( _rowA, sub_group_broadcast( colB,  1 ), _result.s1 );  \
        _result.s2 = mad( _rowA, sub_group_broadcast( colB,  2 ), _result.s2 );  \
        _result.s3 = mad( _rowA, sub_group_broadcast( colB,  3 ), _result.s3 );  \
        _result.s4 = mad( _rowA, sub_group_broadcast( colB,  4 ), _result.s4 );  \
        _result.s5 = mad( _rowA, sub_group_broadcast( colB,  5 ), _result.s5 );  \
        _result.s6 = mad( _rowA, sub_group_broadcast( colB,  6 ), _result.s6 );  \
        _result.s7 = mad( _rowA, sub_group_broadcast( colB,  7 ), _result.s7 );  \
    }
    typedef CAT( float, KERNEL_WIDTH ) float_t;

    // Walk DOWN src0 (patch 0, 1, 2, ...) and DOWN src1.
    // Inner loop loads and FMADs one row (KERNEL_WIDTH) of each input patch
    // and KERNEL_WIDTH/2 rows of interleaved filter.
    unsigned patch_depth = 0;
    do
    {
        unsigned patch_row = 0;
        do
        {
            // Load atile and btile.
            // Kernel data is partially interleaved.  Every 2 rows are interleaved at float8 granularity.
            // The exception is that if KERNEL_WIDTH is odd the last row is not interleaved.  The non
            // interleaved row is padded with zero to ensure same size as interleaved rows. This
            // interleaving is done to ensure 0% GDR bank conflicts.  For example, this is how the
            // kernel data would be arranged before/after interleaving for KERNEL_WIDTH=3.
            // (0, 0) (8, 0) (16, 0) (24, 0) ...       (0, 0) (0, 1) (8, 0) (0, 1) (16, 0) (0, 1) (24, 0) ..
            // (0, 1) (8, 1) (16, 1) (24, 1) ... =>    (0, 2) (8, 2) (16, 2) (24, 2) ...
            // (0, 2) (8, 2) (16, 2) (24, 2) ...       ...
            // ...
            const bool kernel_width_is_odd = KERNEL_WIDTH % 2 == 1;

            const __global float *src0_read0 = src0 + src0_read_offset0;
            const __global float *src0_read1 = src0 + src0_read_offset1;
            
            float blockA00[KERNEL_WIDTH];
            float blockA01[KERNEL_WIDTH];
            
            // in case the data is not aligned to sizeof(T)*KERNEL_WIDTH we need to use vload or set the data in a loop
            {
                unsigned i = 0;
                LOOP(KERNEL_WIDTH, i, 
                {
                    blockA00[i] = src0_read0[i];
                    blockA01[i] = src0_read1[i];
                } )
            }

            float*  pblockA00 = (float*)(&blockA00);
            float*  pblockA01 = (float*)(&blockA01);

            src0_read_offset0 += INPUT_Y_PITCH;
            src0_read_offset1 += INPUT_Y_PITCH;


            float blockB00[KERNEL_WIDTH*4];
            float8* p8BlockB00 = (float8*)blockB00;
            float4* p4BlockB00 = (float4*)blockB00;
            float*  pBlockB00 =  (float* )blockB00;

            interleaved_y = 0;
            LOOP(KERNEL_WIDTH_DIV2, interleaved_y,
            {
                p8BlockB00[interleaved_y] = as_float8( intel_sub_group_block_read8( (const __global uint*)src1_read ) );
                src1_read += ALIGNED_OFM * 2;
            } )
            if ( kernel_width_is_odd )
            {
                p4BlockB00[KERNEL_WIDTH - 1] = as_float4( intel_sub_group_block_read4( (const __global uint*)src1_read ) );
                src1_read += ALIGNED_OFM * 2;
            }

            // Perform MADs
            kernel_idx = 0;
            interleaved_y = 0;
            LOOP(KERNEL_WIDTH_DIV2, interleaved_y,
            {
                kernel_y = interleaved_y * 2;
                DOT_PRODUCT_8( blockC00, pblockA00[kernel_y    ], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC01, pblockA01[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC00, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC01, pblockA01[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC10, pblockA00[kernel_y    ], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC11, pblockA01[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC10, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC11, pblockA01[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC20, pblockA00[kernel_y    ], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC21, pblockA01[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC20, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC21, pblockA01[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC30, pblockA00[kernel_y    ], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC31, pblockA01[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC30, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC31, pblockA01[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
            } )
            if ( kernel_width_is_odd )
            {
                kernel_y = interleaved_y * 2;
                DOT_PRODUCT_8( blockC00, pblockA00[kernel_y], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC01, pblockA01[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC10, pblockA00[kernel_y], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC11, pblockA01[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC20, pblockA00[kernel_y], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC21, pblockA01[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC30, pblockA00[kernel_y], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC31, pblockA01[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
            }
        }

        //while( ++patch_row < 1 ); //debug
        while( ++patch_row < KERNEL_HEIGHT );

        src0_read_offset0 += INPUT_FEATURE_PITCH - ( KERNEL_HEIGHT * INPUT_Y_PITCH ); // reset to start of next slice of patch
        src0_read_offset1 += INPUT_FEATURE_PITCH - ( KERNEL_HEIGHT * INPUT_Y_PITCH ); // reset to start of next slice of patch
    }
    //while ( ++patch_depth < 1 );  //debug
    while ( ++patch_depth < INPUT_FEATURE_NUM );

    const uint out_split_offset = split_idx * OUTPUT_FEATURE_PITCH * OUTPUT_FEATURE_NUM;
    // Dst resembles a cube of width x height x (output channel * batches).  Each tile writes:
    // (SIMD * TILE_M) x 1 x TILE_N.  Partial writes most likely generated if padding used.
    __global float *out0 = dst + OUTPUT_OFFSET + out_split_offset
     + global_z * OUTPUT_BATCH_PITCH                                                       // batch offset
     + ( group_x * TILE_N ) * OUTPUT_FEATURE_PITCH                                           // channel offset
     + ( ( global_y * TILE_M + 0 ) / OUTPUT_SIZE_X ) * OUTPUT_Y_PITCH // y offset
     + ( ( global_y * TILE_M + 0 ) % OUTPUT_SIZE_X );               // x offset
    __global float *out1 = dst + OUTPUT_OFFSET + out_split_offset
     + global_z * OUTPUT_BATCH_PITCH                                                       // batch offset
     + ( group_x * TILE_N ) * OUTPUT_FEATURE_PITCH                                           // channel offset
     + ( ( global_y * TILE_M + 1 ) / OUTPUT_SIZE_X ) * OUTPUT_Y_PITCH // y offset
     + ( ( global_y * TILE_M + 1 ) % OUTPUT_SIZE_X );               // x offset

    #ifdef OUTPUT_BIASED
    __global float8* biasPtr = (__global float8*) (bias + group_x * TILE_N);
    #endif
    
    if( global_y * TILE_M < OUTPUT_SIZE_X * OUTPUT_SIZE_Y )
    {
        if ( ( OUTPUT_FEATURE_NUM % TILE_N ) == 0 )
        {
            #ifdef OUTPUT_BIASED
            blockC00 += *biasPtr;
            blockC10 += *(biasPtr + 1);
            blockC20 += *(biasPtr + 2);
            blockC30 += *(biasPtr + 3);
            #endif

            blockC00 = FUNC_CALL(activation_function_float8)(blockC00, NL_M, NL_N);
            blockC10 = FUNC_CALL(activation_function_float8)(blockC10, NL_M, NL_N);
            blockC20 = FUNC_CALL(activation_function_float8)(blockC20, NL_M, NL_N);
            blockC30 = FUNC_CALL(activation_function_float8)(blockC30, NL_M, NL_N);

            for( unsigned i = 0; i < 8; i++ )
            {
                out0[( 0+i) * OUTPUT_FEATURE_PITCH] = blockC00[i];
                out0[( 8+i) * OUTPUT_FEATURE_PITCH] = blockC10[i];
                out0[(16+i) * OUTPUT_FEATURE_PITCH] = blockC20[i];
                out0[(24+i) * OUTPUT_FEATURE_PITCH] = blockC30[i];
            }
        }
        else
        {
            if ( ( global_x + 1 ) < get_global_size(0) )
            {
                #ifdef OUTPUT_BIASED
                blockC00 += *biasPtr;
                blockC10 += *(biasPtr + 1);
                blockC20 += *(biasPtr + 2);
                blockC30 += *(biasPtr + 3);
                #endif

                blockC00 = FUNC_CALL(activation_function_float8)(blockC00, NL_M, NL_N);
                blockC10 = FUNC_CALL(activation_function_float8)(blockC10, NL_M, NL_N);
                blockC20 = FUNC_CALL(activation_function_float8)(blockC20, NL_M, NL_N);
                blockC30 = FUNC_CALL(activation_function_float8)(blockC30, NL_M, NL_N);

                for ( unsigned i = 0; i < 8; i++ )
                {
                    out0[( 0+i) * OUTPUT_FEATURE_PITCH] = blockC00[i];
                    out0[( 8+i) * OUTPUT_FEATURE_PITCH] = blockC10[i];
                    out0[(16+i) * OUTPUT_FEATURE_PITCH] = blockC20[i];
                    out0[(24+i) * OUTPUT_FEATURE_PITCH] = blockC30[i];
                }
            }
            else
            {
                if ( ( OUTPUT_FEATURE_NUM % TILE_N ) >= 24 )
                {
                    #ifdef OUTPUT_BIASED
                    blockC00 += *biasPtr;
                    blockC10 += *(biasPtr + 1);
                    blockC20 += *(biasPtr + 2);
                    if (( OUTPUT_FEATURE_NUM % TILE_N) > 24 ) blockC30 += *(biasPtr + 3);
                    #endif

                    blockC00 = FUNC_CALL(activation_function_float8)(blockC00, NL_M, NL_N);
                    blockC10 = FUNC_CALL(activation_function_float8)(blockC10, NL_M, NL_N);
                    blockC20 = FUNC_CALL(activation_function_float8)(blockC20, NL_M, NL_N);

                    for (unsigned i = 0; i < 8; i++)
                    {
                        out0[( 0+i) * OUTPUT_FEATURE_PITCH] = blockC00[i];
                        out0[( 8+i) * OUTPUT_FEATURE_PITCH] = blockC10[i];
                        out0[(16+i) * OUTPUT_FEATURE_PITCH] = blockC20[i];
                    }

                    // remaining output channels
                    for (unsigned i = 0; i < OUTPUT_FEATURE_NUM % 8; i++)
                    {
                        out0[(24+i) * OUTPUT_FEATURE_PITCH] = FUNC_CALL(activation_function)(blockC30[i], NL_M, NL_N);
                    }
                }
                else if ( ( OUTPUT_FEATURE_NUM % TILE_N ) >= 16 )
                {
                    #ifdef OUTPUT_BIASED
                    blockC00 += *biasPtr;
                    blockC10 += *(biasPtr + 1);
                    if (( OUTPUT_FEATURE_NUM % TILE_N) > 16 )
                        blockC20 += *(biasPtr + 2);
                    #endif

                    blockC00 = FUNC_CALL(activation_function_float8)(blockC00, NL_M, NL_N);
                    blockC10 = FUNC_CALL(activation_function_float8)(blockC10, NL_M, NL_N);

                    for (unsigned i = 0; i < 8; i++)
                    {
                        out0[( 0+i) * OUTPUT_FEATURE_PITCH] = blockC00[i];
                        out0[( 8+i) * OUTPUT_FEATURE_PITCH] = blockC10[i];
                    }

                    for (unsigned i = 0; i < OUTPUT_FEATURE_NUM % 8; i++)
                    {
                        out0[(16+i) * OUTPUT_FEATURE_PITCH] = FUNC_CALL(activation_function)(blockC20[i], NL_M, NL_N);

                    }
                }
                else if ( ( OUTPUT_FEATURE_NUM % TILE_N ) >= 8 )
                {
                    #ifdef OUTPUT_BIASED
                    blockC00 += *biasPtr;
                    if (( OUTPUT_FEATURE_NUM % TILE_N) > 8 )
                        blockC10 += *(biasPtr + 1);
                    #endif

                    blockC00 = FUNC_CALL(activation_function_float8)(blockC00, NL_M, NL_N);

                    for (unsigned i = 0; i < 8; i++)
                    {
                        out0[( 0+i) * OUTPUT_FEATURE_PITCH] = blockC00[i];
                    }

                    for (unsigned i = 0; i < OUTPUT_FEATURE_NUM % 8; i++)
                    {
                        out0[(8+i) * OUTPUT_FEATURE_PITCH] = FUNC_CALL(activation_function)(blockC10[i], NL_M, NL_N);
                    }
                }
                else
                {
                    #ifdef OUTPUT_BIASED
                    blockC00 += *biasPtr;
                    #endif
                    for (unsigned i = 0; i < OUTPUT_FEATURE_NUM % 8; i++)
                    {
                        out0[( 0+i) * OUTPUT_FEATURE_PITCH] = FUNC_CALL(activation_function)(blockC00[i], NL_M, NL_N);
                    }
                }
            }
        }
    }

    if ((global_y * TILE_M + 1) < OUTPUT_SIZE_X * OUTPUT_SIZE_Y )
    {
        if ( ( OUTPUT_FEATURE_NUM % TILE_N ) == 0 )
        {
            #ifdef OUTPUT_BIASED
            blockC01 += *biasPtr;
            blockC11 += *(biasPtr + 1);
            blockC21 += *(biasPtr + 2);
            blockC31 += *(biasPtr + 3);
            #endif

            blockC01 = FUNC_CALL(activation_function_float8)(blockC01, NL_M, NL_N);
            blockC11 = FUNC_CALL(activation_function_float8)(blockC11, NL_M, NL_N);
            blockC21 = FUNC_CALL(activation_function_float8)(blockC21, NL_M, NL_N);
            blockC31 = FUNC_CALL(activation_function_float8)(blockC31, NL_M, NL_N);

            for( unsigned i = 0; i < 8; i++ )
            {
                out1[( 0+i) * OUTPUT_FEATURE_PITCH] = blockC01[i];
                out1[( 8+i) * OUTPUT_FEATURE_PITCH] = blockC11[i];
                out1[(16+i) * OUTPUT_FEATURE_PITCH] = blockC21[i];
                out1[(24+i) * OUTPUT_FEATURE_PITCH] = blockC31[i];
            }
        }
        else
        {
            if ( ( global_x + 1 ) < get_global_size(0) )
            {
                #ifdef OUTPUT_BIASED
                blockC01 += *biasPtr;
                blockC11 += *(biasPtr + 1);
                blockC21 += *(biasPtr + 2);
                blockC31 += *(biasPtr + 3);
                #endif

                blockC01 = FUNC_CALL(activation_function_float8)(blockC01, NL_M, NL_N);
                blockC11 = FUNC_CALL(activation_function_float8)(blockC11, NL_M, NL_N);
                blockC21 = FUNC_CALL(activation_function_float8)(blockC21, NL_M, NL_N);
                blockC31 = FUNC_CALL(activation_function_float8)(blockC31, NL_M, NL_N);

                for ( unsigned i = 0; i < 8; i++ )
                {
                    out1[( 0+i) * OUTPUT_FEATURE_PITCH] = blockC01[i];
                    out1[( 8+i) * OUTPUT_FEATURE_PITCH] = blockC11[i];
                    out1[(16+i) * OUTPUT_FEATURE_PITCH] = blockC21[i];
                    out1[(24+i) * OUTPUT_FEATURE_PITCH] = blockC31[i];
                }
            }
            else
            {
                if ( ( OUTPUT_FEATURE_NUM % TILE_N ) >= 24 )
                {
                    #ifdef OUTPUT_BIASED
                    blockC01 += *biasPtr;
                    blockC11 += *(biasPtr + 1);
                    blockC21 += *(biasPtr + 2);
                    if ( ( OUTPUT_FEATURE_NUM % TILE_N ) > 24 ) blockC31 += *(biasPtr + 3);
                    #endif

                    blockC01 = FUNC_CALL(activation_function_float8)(blockC01, NL_M, NL_N);
                    blockC11 = FUNC_CALL(activation_function_float8)(blockC11, NL_M, NL_N);
                    blockC21 = FUNC_CALL(activation_function_float8)(blockC21, NL_M, NL_N);

                    for (unsigned i = 0; i < 8; i++)
                    {
                        out1[( 0+i) * OUTPUT_FEATURE_PITCH] = blockC01[i];
                        out1[( 8+i) * OUTPUT_FEATURE_PITCH] = blockC11[i];
                        out1[(16+i) * OUTPUT_FEATURE_PITCH] = blockC21[i];
                    }

                    // Remaining channels
                    for (unsigned i = 0; i < OUTPUT_FEATURE_NUM % 8; i++)
                    {
                        out1[(24+i) * OUTPUT_FEATURE_PITCH] = FUNC_CALL(activation_function)(blockC31[i], NL_M, NL_N);
                    }
                }
                else if ( ( OUTPUT_FEATURE_NUM % TILE_N ) >= 16 )
                {
                    #ifdef OUTPUT_BIASED
                    blockC01 += *biasPtr;
                    blockC11 += *(biasPtr + 1);
                    if ( ( OUTPUT_FEATURE_NUM % TILE_N ) > 16 ) blockC21 += *(biasPtr + 2);
                    #endif

                    blockC01 = FUNC_CALL(activation_function_float8)(blockC01, NL_M, NL_N);
                    blockC11 = FUNC_CALL(activation_function_float8)(blockC11, NL_M, NL_N);

                    for (unsigned i = 0; i < 8; i++)
                    {
                        out1[( 0+i) * OUTPUT_FEATURE_PITCH] = blockC01[i];
                        out1[( 8+i) * OUTPUT_FEATURE_PITCH] = blockC11[i];
                    }

                    for (unsigned i = 0; i < OUTPUT_FEATURE_NUM % 8; i++)
                    {
                        out1[(16+i) * OUTPUT_FEATURE_PITCH] = FUNC_CALL(activation_function)(blockC21[i], NL_M, NL_N);
                    }
                }
                else if ( ( OUTPUT_FEATURE_NUM % TILE_N ) >= 8 )
                {
                    #ifdef OUTPUT_BIASED
                    blockC01 += *biasPtr;
                    if ( ( OUTPUT_FEATURE_NUM % TILE_N ) > 8 ) blockC11 += *(biasPtr + 1);
                    #endif

                    blockC01 = FUNC_CALL(activation_function_float8)(blockC01, NL_M, NL_N);

                    for (unsigned i = 0; i < 8; i++)
                    {
                        out1[( 0+i) * OUTPUT_FEATURE_PITCH] = blockC01[i];
                    }

                    for (unsigned i = 0; i < OUTPUT_FEATURE_NUM % 8; i++)
                    {
                        out1[(8+i) * OUTPUT_FEATURE_PITCH] = FUNC_CALL(activation_function)(blockC11[i], NL_M, NL_N);
                    }
                }
                else
                {
                    #ifdef OUTPUT_BIASED
                    blockC01 += *biasPtr;
                    #endif

                    for (unsigned i = 0; i < OUTPUT_FEATURE_NUM % 8; i++)
                    {
                        out1[( 0+i) * OUTPUT_FEATURE_PITCH] = FUNC_CALL(activation_function)(blockC01[i], NL_M, NL_N);
                    }
                }
            }
        }
    }
}
