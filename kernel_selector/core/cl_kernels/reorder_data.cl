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


#include "include/common.cl"

///////////////////////// subtruct Index /////////////////////////
#ifdef MEAN_SUBTRUCT_IN_BUFFER
#if MEAN_SUBTRUCT_SIMPLE
inline uint FUNC(get_mean_index)(uint b, uint f, uint y, uint x)
{ 
   return MEAN_SUBTRUCT_OFFSET + b*MEAN_SUBTRUCT_BATCH_PITCH + f*MEAN_SUBTRUCT_FEATURE_PITCH + y*MEAN_SUBTRUCT_Y_PITCH + x*MEAN_SUBTRUCT_X_PITCH;
}
#else
#error - not supported
#endif

inline uint4 FUNC(reshape_mean)(uint b, uint f, uint y, uint x)
{
#if (INPUT_DIMS == MEAN_SUBTRUCT_DIMS)
    return (uint4)(b,f,y,x);
#elif (INPUT_DIMS == 2 && MEAN_SUBTRUCT_DIMS == 4)
    uint _f  = f / (INPUT_SIZE_Y*INPUT_SIZE_X);
    uint _yx = f % (INPUT_SIZE_Y*INPUT_SIZE_X);
    uint _y = _yx / INPUT_SIZE_X;
    uint _x = _yx % INPUT_SIZE_X;
    return (uint4)(b,_f,_y,_x);
#elif (INPUT_DIMS == 4 && MEAN_SUBTRUCT_DIMS == 2)
    uint _f = f*INPUT_SIZE_Y*INPUT_SIZE_X + y*INPUT_SIZE_X + x;
    return (uint4)(b,_f,0,0);
#else
#error
#endif
}

#endif

///////////////////////// Input Index /////////////////////////
#if INPUT_SIMPLE
inline uint FUNC(get_input_index)(uint b, uint f, uint y, uint x)
{ 
   return INPUT_OFFSET + b*INPUT_BATCH_PITCH + f*INPUT_FEATURE_PITCH + y*INPUT_Y_PITCH + x*INPUT_X_PITCH;
}
#else
#error - not supported
#endif

///////////////////////// Output Index /////////////////////////

#if OUTPUT_SIMPLE
inline uint FUNC(get_output_index)(uint b, uint f, uint y, uint x)
{ 
    return OUTPUT_OFFSET + b*OUTPUT_BATCH_PITCH + f*OUTPUT_FEATURE_PITCH + y*OUTPUT_Y_PITCH + x*OUTPUT_X_PITCH;
}
#else

#if defined OUTPUT_LAYOUT_OS_I_OSV16 || defined OUTPUT_LAYOUT_BS_F_BSV8__AF8 || defined OUTPUT_LAYOUT_BS_F_BSV16__AF8
inline uint FUNC(get_output_index)(uint b, uint f, uint y, uint x)
{ 
    const uint slice_id = b / SUB_GROUP_SIZE;
    const uint id_in_slice = b % SUB_GROUP_SIZE;
    const size_t output_idx = OUTPUT_OFFSET + id_in_slice + SUB_GROUP_SIZE * (f*OUTPUT_FEATURE_PITCH + slice_id*OUTPUT_BATCH_PITCH);
    return output_idx;
}
#endif
#endif

inline uint4 FUNC(reshape)(uint b, uint f, uint y, uint x)
{
#if (INPUT_DIMS == OUTPUT_DIMS)
    return (uint4)(b,f,y,x);
#elif (INPUT_DIMS == 2 && OUTPUT_DIMS == 4)
    uint _f  = f / (INPUT_SIZE_Y*INPUT_SIZE_X);
    uint _yx = f % (INPUT_SIZE_Y*INPUT_SIZE_X);
    uint _y = _yx / INPUT_SIZE_X;
    uint _x = _yx % INPUT_SIZE_X;
    return (uint4)(b,_f,_y,_x);
#elif (INPUT_DIMS == 4 && OUTPUT_DIMS == 2)
    uint _f = f*INPUT_SIZE_Y*INPUT_SIZE_X + y*INPUT_SIZE_X + x;
    return (uint4)(b,_f,0,0);
#else
#error
#endif
}


KERNEL (reorder_weights)(
    const __global INPUT_TYPE* input, 
    __global OUTPUT_TYPE* output
#ifdef MEAN_SUBTRUCT_IN_BUFFER
    , __global MEAN_SUBTRUCT_TYPE* mean_subtruct
#endif
    )
{
    const unsigned b = get_global_id(GWS_BATCH);
    const unsigned f = get_global_id(GWS_FEATURE);
#if   INPUT_DIMS == 2
    const unsigned y = 0;
    const unsigned x = 0;
#elif INPUT_DIMS == 4
    const unsigned y = get_global_id(GWS_YX) / INPUT_SIZE_X;
    const unsigned x = get_global_id(GWS_YX) % INPUT_SIZE_X;
#endif

    uint4 ov = FUNC_CALL(reshape)(b,f,y,x);
    const uint input_idx  = FUNC_CALL(get_input_index)(b, f, y, x);
    const uint output_idx = FUNC_CALL(get_output_index)(ov[0],ov[1],ov[2],ov[3]);
    CALC_TYPE res = TO_CALC_TYPE(input[input_idx]);
    
#if   defined MEAN_SUBTRUCT_INSIDE_PARAMS
    res -= TO_CALC_TYPE(VALUE_TO_SUBTRACT[f % VALUE_TO_SUBTRACT_SIZE]);
#elif defined MEAN_SUBTRUCT_IN_BUFFER
    uint4 msv = FUNC_CALL(reshape_mean)(b,f,y,x);
    res -= TO_CALC_TYPE(mean_subtruct[FUNC_CALL(get_mean_index)(
        msv[0] % MEAN_SUBTRUCT_BATCH_NUM,
        msv[1] % MEAN_SUBTRUCT_FEATURE_NUM,
        msv[2] % MEAN_SUBTRUCT_SIZE_Y,
        msv[3] % MEAN_SUBTRUCT_SIZE_X)]);
#endif

    output[output_idx] = TO_OUTPUT_TYPE(res);
}