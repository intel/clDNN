/*
// Copyright (c) 2016-2019 Intel Corporation
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

#include "include/include_all.cl"

#if OUTPUT_DIMS == 5 // 3D spatial
#if !ELTWISE_BROADCAST
#ifdef INPUT_STRIDED

#define GET_INDEX(prefix, num) \
    CAT(CAT(prefix, num), _OFFSET) + \
    ((d1 * CAT(CAT(prefix, num), _STRIDE_X)) % CAT(CAT(prefix, num), _SIZE_X))*CAT(CAT(prefix, num), _X_PITCH) +\
    ((d2 * CAT(CAT(prefix, num), _STRIDE_Y)) % CAT(CAT(prefix, num), _SIZE_Y))*CAT(CAT(prefix, num), _Y_PITCH) +\
    ((d3 * CAT(CAT(prefix, num), _STRIDE_Z)) % CAT(CAT(prefix, num), _SIZE_Z))*CAT(CAT(prefix, num), _Z_PITCH) +\
    (d4 % CAT(CAT(prefix, num), _FEATURE_NUM))*CAT(CAT(prefix, num), _FEATURE_PITCH) + \
    (d5 % CAT(CAT(prefix, num), _BATCH_NUM  ))*CAT(CAT(prefix, num), _BATCH_PITCH)

#else

#if ELTWISE_LAYOUT_BASED || QUANTIZATION_TERM

#define GET_INDEX(prefix, num)                                                          \
    CAT(CAT(prefix, num), _OFFSET) +                                                    \
    (d1 % CAT(CAT(prefix, num), _SIZE_X     ))*CAT(CAT(prefix, num), _X_PITCH) +        \
    (d2 % CAT(CAT(prefix, num), _SIZE_Y     ))*CAT(CAT(prefix, num), _Y_PITCH) +        \
    (d3 % CAT(CAT(prefix, num), _SIZE_Z     ))*CAT(CAT(prefix, num), _Z_PITCH) +        \
    (d4 % CAT(CAT(prefix, num), _FEATURE_NUM))*CAT(CAT(prefix, num), _FEATURE_PITCH) +  \
    (d5 % CAT(CAT(prefix, num), _BATCH_NUM  ))*CAT(CAT(prefix, num), _BATCH_PITCH)

#elif ELTWISE_NO_PITCH_SAME_DIMS
#define GET_INDEX(prefix, num)                                                      \
    CAT(CAT(prefix, num), _OFFSET) + d1

#else

#define GET_INDEX(prefix, num)                                                      \
    CAT(CAT(prefix, num), _OFFSET) +                                                \
    (d1 % CAT(CAT(prefix, num), _SIZES)[0])*CAT(CAT(prefix, num), _PITCHES)[0] +    \
    (d2 % CAT(CAT(prefix, num), _SIZES)[1])*CAT(CAT(prefix, num), _PITCHES)[1] +    \
    (d3 % CAT(CAT(prefix, num), _SIZES)[2])*CAT(CAT(prefix, num), _PITCHES)[2] +    \
    (d4 % CAT(CAT(prefix, num), _SIZES)[3])*CAT(CAT(prefix, num), _PITCHES)[3] +    \
    (d5 % CAT(CAT(prefix, num), _SIZES)[4])*CAT(CAT(prefix, num), _PITCHES)[4]

#endif

#endif
#else // ELTWISE_BROADCAST
#ifdef INPUT_STRIDED

#define GET_INDEX(prefix, num) \
    CAT(CAT(prefix, num), _OFFSET) + \
    ((CAT(d1_in, num) * CAT(CAT(prefix, num), _STRIDE_X)) % CAT(CAT(prefix, num), _SIZE_X))*CAT(CAT(prefix, num), _X_PITCH) +\
    ((CAT(d2_in, num) * CAT(CAT(prefix, num), _STRIDE_Y)) % CAT(CAT(prefix, num), _SIZE_Y))*CAT(CAT(prefix, num), _Y_PITCH) +\
    ((CAT(d3_in, num) * CAT(CAT(prefix, num), _STRIDE_Z)) % CAT(CAT(prefix, num), _SIZE_Z))*CAT(CAT(prefix, num), _Z_PITCH) +\
    (CAT(d4_in, num) % CAT(CAT(prefix, num), _FEATURE_NUM))*CAT(CAT(prefix, num), _FEATURE_PITCH) + \
    (CAT(d5_in, num) % CAT(CAT(prefix, num), _BATCH_NUM  ))*CAT(CAT(prefix, num), _BATCH_PITCH)

#else

#if ELTWISE_LAYOUT_BASED || QUANTIZATION_TERM

#define GET_INDEX(prefix, num)                                                          \
    CAT(CAT(prefix, num), _OFFSET) +                                                    \
    (CAT(d1_in, num) % CAT(CAT(prefix, num), _SIZE_X     ))*CAT(CAT(prefix, num), _X_PITCH) +        \
    (CAT(d2_in, num) % CAT(CAT(prefix, num), _SIZE_Y     ))*CAT(CAT(prefix, num), _Y_PITCH) +        \
    (CAT(d3_in, num) % CAT(CAT(prefix, num), _SIZE_Z     ))*CAT(CAT(prefix, num), _Z_PITCH) +        \
    (CAT(d4_in, num) % CAT(CAT(prefix, num), _FEATURE_NUM))*CAT(CAT(prefix, num), _FEATURE_PITCH) +  \
    (CAT(d5_in, num) % CAT(CAT(prefix, num), _BATCH_NUM  ))*CAT(CAT(prefix, num), _BATCH_PITCH)

#elif ELTWISE_NO_PITCH_SAME_DIMS
#define GET_INDEX(prefix, num)                                                      \
    CAT(CAT(prefix, num), _OFFSET) + CAT(d1_in, num)

#else

#define GET_INDEX(prefix, num)                                                      \
    CAT(CAT(prefix, num), _OFFSET) +                                                \
    (CAT(d1_in, num) % CAT(CAT(prefix, num), _SIZES)[0])*CAT(CAT(prefix, num), _PITCHES)[0] +    \
    (CAT(d2_in, num) % CAT(CAT(prefix, num), _SIZES)[1])*CAT(CAT(prefix, num), _PITCHES)[1] +    \
    (CAT(d3_in, num) % CAT(CAT(prefix, num), _SIZES)[2])*CAT(CAT(prefix, num), _PITCHES)[2] +    \
    (CAT(d4_in, num) % CAT(CAT(prefix, num), _SIZES)[3])*CAT(CAT(prefix, num), _PITCHES)[3] +    \
    (CAT(d5_in, num) % CAT(CAT(prefix, num), _SIZES)[4])*CAT(CAT(prefix, num), _PITCHES)[4]

#endif

#endif
#endif // ELTWISE_BROADCAST
#else // 2D spatial
#if !ELTWISE_BROADCAST
#ifdef INPUT_STRIDED

#define GET_INDEX(prefix, num) \
    CAT(CAT(prefix, num), _OFFSET) + \
    ((d1 * CAT(CAT(prefix, num), _STRIDE_X)) % CAT(CAT(prefix, num), _SIZE_X))*CAT(CAT(prefix, num), _X_PITCH) +\
    ((d2 * CAT(CAT(prefix, num), _STRIDE_Y)) % CAT(CAT(prefix, num), _SIZE_Y))*CAT(CAT(prefix, num), _Y_PITCH) +\
    (d3 % CAT(CAT(prefix, num), _FEATURE_NUM))*CAT(CAT(prefix, num), _FEATURE_PITCH) + \
    (d4 % CAT(CAT(prefix, num), _BATCH_NUM  ))*CAT(CAT(prefix, num), _BATCH_PITCH)

#else

#if ELTWISE_LAYOUT_BASED || QUANTIZATION_TERM

#define GET_INDEX(prefix, num)                                                          \
    CAT(CAT(prefix, num), _OFFSET) +                                                    \
    (d1 % CAT(CAT(prefix, num), _SIZE_X     ))*CAT(CAT(prefix, num), _X_PITCH) +        \
    (d2 % CAT(CAT(prefix, num), _SIZE_Y     ))*CAT(CAT(prefix, num), _Y_PITCH) +        \
    (d3 % CAT(CAT(prefix, num), _FEATURE_NUM))*CAT(CAT(prefix, num), _FEATURE_PITCH) +  \
    (d4 % CAT(CAT(prefix, num), _BATCH_NUM  ))*CAT(CAT(prefix, num), _BATCH_PITCH)

#elif ELTWISE_NO_PITCH_SAME_DIMS
#define GET_INDEX(prefix, num)                                                      \
    CAT(CAT(prefix, num), _OFFSET) + d1

#else

#define GET_INDEX(prefix, num)                                                      \
    CAT(CAT(prefix, num), _OFFSET) +                                                \
    (d1 % CAT(CAT(prefix, num), _SIZES)[0])*CAT(CAT(prefix, num), _PITCHES)[0] +    \
    (d2 % CAT(CAT(prefix, num), _SIZES)[1])*CAT(CAT(prefix, num), _PITCHES)[1] +    \
    (d3 % CAT(CAT(prefix, num), _SIZES)[2])*CAT(CAT(prefix, num), _PITCHES)[2] +    \
    (d4 % CAT(CAT(prefix, num), _SIZES)[3])*CAT(CAT(prefix, num), _PITCHES)[3]

#endif

#endif
#else
#ifdef INPUT_STRIDED

#define GET_INDEX(prefix, num) \
    CAT(CAT(prefix, num), _OFFSET) + \
    ((CAT(d1_in, num) * CAT(CAT(prefix, num), _STRIDE_X)) % CAT(CAT(prefix, num), _SIZE_X))*CAT(CAT(prefix, num), _X_PITCH) +\
    ((CAT(d2_in, num) * CAT(CAT(prefix, num), _STRIDE_Y)) % CAT(CAT(prefix, num), _SIZE_Y))*CAT(CAT(prefix, num), _Y_PITCH) +\
    (CAT(d3_in, num) % CAT(CAT(prefix, num), _FEATURE_NUM))*CAT(CAT(prefix, num), _FEATURE_PITCH) + \
    (CAT(d4_in, num) % CAT(CAT(prefix, num), _BATCH_NUM  ))*CAT(CAT(prefix, num), _BATCH_PITCH)

#else

#if ELTWISE_LAYOUT_BASED || QUANTIZATION_TERM

#define GET_INDEX(prefix, num)                                                          \
    CAT(CAT(prefix, num), _OFFSET) +                                                    \
    (CAT(d1_in, num) % CAT(CAT(prefix, num), _SIZE_X     ))*CAT(CAT(prefix, num), _X_PITCH) +        \
    (CAT(d2_in, num) % CAT(CAT(prefix, num), _SIZE_Y     ))*CAT(CAT(prefix, num), _Y_PITCH) +        \
    (CAT(d3_in, num) % CAT(CAT(prefix, num), _FEATURE_NUM))*CAT(CAT(prefix, num), _FEATURE_PITCH) +  \
    (CAT(d4_in, num) % CAT(CAT(prefix, num), _BATCH_NUM  ))*CAT(CAT(prefix, num), _BATCH_PITCH)

#elif ELTWISE_NO_PITCH_SAME_DIMS
#define GET_INDEX(prefix, num)                                                      \
    CAT(CAT(prefix, num), _OFFSET) + CAT(d1_in, num)

#else

#define GET_INDEX(prefix, num)                                                      \
    CAT(CAT(prefix, num), _OFFSET) +                                                \
    (CAT(d1_in, num) % CAT(CAT(prefix, num), _SIZES)[0])*CAT(CAT(prefix, num), _PITCHES)[0] +    \
    (CAT(d2_in, num) % CAT(CAT(prefix, num), _SIZES)[1])*CAT(CAT(prefix, num), _PITCHES)[1] +    \
    (CAT(d3_in, num) % CAT(CAT(prefix, num), _SIZES)[2])*CAT(CAT(prefix, num), _PITCHES)[2] +    \
    (CAT(d4_in, num) % CAT(CAT(prefix, num), _SIZES)[3])*CAT(CAT(prefix, num), _PITCHES)[3]

#endif

#endif
#endif
#endif // 3D/2D spatial

#if OUTPUT_LAYOUT_BFYX_F16
#   define GET_OUTPUT_INDEX(b, f, y, x)                        \
        GET_DATA_BFYX_F16_INDEX(OUTPUT, b, f, y, x)

#endif

#if !ELTWISE_NO_PITCH_SAME_DIMS
#   define GET_INPUT_INDEX_BFYX_F16(num)                                            \
    GET_DATA_BFYX_F16_INDEX(                                                        \
            CAT(INPUT, num),                                                        \
            d4 % CAT(CAT(INPUT, num), _BATCH_NUM),                                  \
            d3 % CAT(CAT(INPUT, num), _FEATURE_NUM),                                \
            d2 * CAT(CAT(INPUT, num), _STRIDE_Y) % CAT(CAT(INPUT, num), _SIZE_Y),   \
            d1 * CAT(CAT(INPUT, num), _STRIDE_X) % CAT(CAT(INPUT, num), _SIZE_X))

#else
#   define GET_INPUT_INDEX_BFYX_F16(num) GET_INDEX(INPUT, num)
#endif

KERNEL(eltwise)(
    INPUTS_DECLS
    __global UNIT_TYPE* output
#if CALIBRATION_TERM
    , const __global float* calibrations
#endif
    )
{
#if OUTPUT_DIMS == 5 // 3D spatial
#if ELTWISE_LAYOUT_BASED || QUANTIZATION_TERM

    uint data_idx = get_global_id(GWS_YX);
    const uint d1 = data_idx % OUTPUT_SIZE_X; // X
    data_idx = data_idx / OUTPUT_SIZE_X;

    const uint d2 = data_idx % OUTPUT_SIZE_Y; // Y
    data_idx = data_idx / OUTPUT_SIZE_Y;

    const uint d3 = data_idx % OUTPUT_SIZE_Z; // Z

    const uint d4 = get_global_id(GWS_FEATURE);             // Feature
    const uint d5 = get_global_id(GWS_BATCH);               // Batch

    uint output_offset = OUTPUT_OFFSET +
                         d1*OUTPUT_X_PITCH +
                         d2*OUTPUT_Y_PITCH +
                         d3*OUTPUT_Z_PITCH +
                         d4*OUTPUT_FEATURE_PITCH +
                         d5*OUTPUT_BATCH_PITCH;

#elif ELTWISE_NO_PITCH_SAME_DIMS
    const uint d1 = get_global_id(0);
    uint output_offset = OUTPUT_OFFSET + d1;
#else
    const uint d1 = get_global_id(0);
    const uint d2 = get_global_id(1) % OUTPUT_SIZES[2];
    const uint d3 = get_global_id(1) / OUTPUT_SIZES[2];
    const uint d4 = get_global_id(2) % OUTPUT_SIZES[3];
    const uint d5 = get_global_id(2) / OUTPUT_SIZES[3];

    uint output_offset = OUTPUT_OFFSET +
                         d1*OUTPUT_PITCHES[0] +
                         d2*OUTPUT_PITCHES[1] +
                         d3*OUTPUT_PITCHES[2] +
                         d4*OUTPUT_PITCHES[3] +
                         d5*OUTPUT_PITCHES[4];
#endif

#else // 2D spatial

#if ELTWISE_LAYOUT_BASED || QUANTIZATION_TERM
    const uint d1 = get_global_id(GWS_YX) % OUTPUT_SIZE_X;  // X
    const uint d2 = get_global_id(GWS_YX) / OUTPUT_SIZE_X;  // Y
    const uint d3 = get_global_id(GWS_FEATURE);             // Feature
    const uint d4 = get_global_id(GWS_BATCH);               // Batch

#if OUTPUT_LAYOUT_BFYX_F16
    uint output_offset = GET_OUTPUT_INDEX(d4, d3, d2, d1);
#else
    uint output_offset = OUTPUT_OFFSET +
                         d1*OUTPUT_X_PITCH +
                         d2*OUTPUT_Y_PITCH +
                         d3*OUTPUT_FEATURE_PITCH +
                         d4*OUTPUT_BATCH_PITCH;
#endif
#elif ELTWISE_NO_PITCH_SAME_DIMS
    const uint d1 = get_global_id(0);
    uint output_offset = OUTPUT_OFFSET + d1;
#else
    const uint d1 = get_global_id(0);
    const uint d2 = get_global_id(1);
    const uint d3 = get_global_id(2) % OUTPUT_SIZES[2];
    const uint d4 = get_global_id(2) / OUTPUT_SIZES[2];

#if OUTPUT_LAYOUT_BFYX_F16
    uint output_offset = GET_OUTPUT_INDEX(d4, d3, d2, d1);
#else  
    uint output_offset = OUTPUT_OFFSET +
                         d1*OUTPUT_PITCHES[0] +
                         d2*OUTPUT_PITCHES[1] +
                         d3*OUTPUT_PITCHES[2] +
                         d4*OUTPUT_PITCHES[3];
#endif
#endif
#endif

#if ELTWISE_BROADCAST
    const uint d1_in0 = d1 % INPUT0_SIZE_X;
#if !ELTWISE_NO_PITCH_SAME_DIMS
    const uint d2_in0 = d2 % INPUT0_SIZE_Y;
#if OUTPUT_DIMS == 5 // 3D spatial
    const uint d3_in0 = d3 % INPUT0_SIZE_Z;
    const uint d4_in0 = d4 % INPUT0_FEATURE_NUM;
    const uint d5_in0 = d5 % INPUT0_BATCH_NUM;
#else // 2D spatial
    const uint d3_in0 = d3 % INPUT0_FEATURE_NUM;
    const uint d4_in0 = d4 % INPUT0_BATCH_NUM;
#endif // 3D/2D spatial
#endif
    const uint d1_in1 = d1 % INPUT1_SIZE_X;
#if !ELTWISE_NO_PITCH_SAME_DIMS
    const uint d2_in1 = d2 % INPUT1_SIZE_Y;
#if OUTPUT_DIMS == 5 // 3D spatial
    const uint d3_in1 = d3 % INPUT1_SIZE_Z;
    const uint d4_in1 = d4 % INPUT1_FEATURE_NUM;
    const uint d5_in1 = d5 % INPUT1_BATCH_NUM;
#else // 2D spatial
    const uint d3_in1 = d3 % INPUT1_FEATURE_NUM;
    const uint d4_in1 = d4 % INPUT1_BATCH_NUM;
#endif // 3D/2D spatial
#endif
#endif

#if QUANTIZATION_TERM
    int res;
#else
    UNIT_TYPE res;
#endif
    
    DO_ELTWISE;

#if QUANTIZATION_TERM
#if CALIBRATION_TERM
    res = (int)round(((float)res) * calibrations[d3]);
#else  // CALIBRATION_TERM
    res = (int)round(((float)res) * O_QF);
#endif // CALIBRATION_TERM
#endif // QUANTIZATION_TERM

#if QUANTIZATION_TERM
    output[output_offset] = ACTIVATION(convert_char_sat(res), NL_M, NL_N);
#else
    output[output_offset] = ACTIVATION(res, NL_M, NL_N);
#endif
}
