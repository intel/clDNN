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

#if defined(cl_intel_subgroups)
#pragma OPENCL EXTENSION  cl_intel_subgroups : enable
#endif

#if defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)

#define __CAT_FUNC(x, y) FUNC(x##y)
#define CAT_FUNC(x, y) __CAT_FUNC(x, y)

#define __CAT_FUNC_CALL(x, y) FUNC_CALL(x##y)
#define CAT_FUNC_CALL(x, y) __CAT_FUNC_CALL(x, y)

#define LOOP0(VAR, STMT) 
#define LOOP1(VAR, STMT) (STMT); (VAR)++;
#define LOOP2(VAR, STMT) LOOP1(VAR, STMT); (STMT); (VAR)++;
#define LOOP3(VAR, STMT) LOOP2(VAR, STMT); (STMT); (VAR)++;
#define LOOP4(VAR, STMT) LOOP3(VAR, STMT); (STMT); (VAR)++;
#define LOOP5(VAR, STMT) LOOP4(VAR, STMT); (STMT); (VAR)++;
#define LOOP6(VAR, STMT) LOOP5(VAR, STMT); (STMT); (VAR)++;
#define LOOP7(VAR, STMT) LOOP6(VAR, STMT); (STMT); (VAR)++;
#define LOOP8(VAR, STMT) LOOP7(VAR, STMT); (STMT); (VAR)++;
#define LOOP9(VAR, STMT) LOOP8(VAR, STMT); (STMT); (VAR)++;
#define LOOP10(VAR, STMT) LOOP9(VAR, STMT); (STMT); (VAR)++;
#define LOOP11(VAR, STMT) LOOP10(VAR, STMT); (STMT); (VAR)++;
#define LOOP12(VAR, STMT) LOOP11(VAR, STMT); (STMT); (VAR)++;
#define LOOP13(VAR, STMT) LOOP12(VAR, STMT); (STMT); (VAR)++;
#define LOOP14(VAR, STMT) LOOP13(VAR, STMT); (STMT); (VAR)++;
#define LOOP15(VAR, STMT) LOOP14(VAR, STMT); (STMT); (VAR)++;
#define LOOP16(VAR, STMT) LOOP15(VAR, STMT); (STMT); (VAR)++;
#define LOOP(N, VAR, STMT) CAT(LOOP, N)((VAR), (STMT))

#undef COUNTER_TYPE

#if defined(COUNTER_TYPE_F16)
     #define COUNTER_TYPE half
#elif defined(COUNTER_TYPE_F32)
     #define COUNTER_TYPE float
#endif

#if defined TYPE_F16
    #define DATA_TYPE half

    // TODO: currently we calculate on float32 because it's lot of "add" operation and it stuck on the value "8192.0f"
    #if !defined(COUNTER_TYPE)
        #define COUNTER_TYPE_F32
        #define COUNTER_TYPE float
    #endif
    
    #define DATA_TYPE_MAX HALF_MAX
    #define DATA_TYPE_MIN -HALF_MAX
    #define DATA_TYPE_ZERO 0.0h
#elif defined TYPE_F32
    #define DATA_TYPE float

    #if !defined(COUNTER_TYPE)
        #define COUNTER_TYPE_F32
        #define COUNTER_TYPE float
    #endif
    
    #define DATA_TYPE_MAX FLT_MAX
    #define DATA_TYPE_MIN -FLT_MAX
    #define DATA_TYPE_ZERO 0.0f
#endif

#if defined ACTIVATION_FUNCTION_LOGISTIC
#define ACTIVATION_FUNCTION(TYPE_T) \
inline TYPE_T CAT_FUNC(activation_function_, TYPE_T)(TYPE_T value, float m, float n)\
    { return (TYPE_T)(1.0) / ((TYPE_T)(1.0) + exp(-value)); }

#elif defined ACTIVATION_FUNCTION_HYPERBOLIC_TAN
#define ACTIVATION_FUNCTION(TYPE_T) \
inline TYPE_T CAT_FUNC(activation_function_, TYPE_T)(TYPE_T value, float m, float n)\
    { return tanh(value); }

#elif defined ACTIVATION_FUNCTION_RELU
#define ACTIVATION_FUNCTION(TYPE_T) \
inline TYPE_T CAT_FUNC(activation_function_, TYPE_T)(TYPE_T value, float m, float n)\
    { return fmax(value, (TYPE_T)(0)); }

#elif defined ACTIVATION_FUNCTION_SOFTRELU
#define ACTIVATION_FUNCTION(TYPE_T) \
inline TYPE_T CAT_FUNC(activation_function_, TYPE_T)(TYPE_T value, float m, float n)\
    { return log( (TYPE_T)(1) + exp(value)); }
    
#elif defined ACTIVATION_FUNCTION_RELU_NEGATIVE_SLOPE
#define ACTIVATION_FUNCTION(TYPE_T) \
inline TYPE_T CAT_FUNC(activation_function_, TYPE_T)(TYPE_T value, float m, float n)\
    { return isinf((TYPE_T)m) ? ((value >= (TYPE_T)0) ? value : -(TYPE_T)m) : (fmax(value, (TYPE_T)0) + (TYPE_T)m * fmin(value, (TYPE_T)0)); }

#elif defined ACTIVATION_FUNCTION_ABS
#define ACTIVATION_FUNCTION(TYPE_T) \
inline TYPE_T CAT_FUNC(activation_function_, TYPE_T)(TYPE_T value, float m, float n)\
    { return fabs(value); }

#elif defined ACTIVATION_FUNCTION_SQUARE
#define ACTIVATION_FUNCTION(TYPE_T) \
inline TYPE_T CAT_FUNC(activation_function_, TYPE_T)(TYPE_T value, float m, float n)\
    { return value * value; }

#elif defined ACTIVATION_FUNCTION_SQRT
#define ACTIVATION_FUNCTION(TYPE_T) \
inline TYPE_T CAT_FUNC(activation_function_, TYPE_T)(TYPE_T value, float m, float n)\
    { return sqrt(value); }

#elif defined ACTIVATION_FUNCTION_BRELU
#define ACTIVATION_FUNCTION(TYPE_T) \
inline TYPE_T CAT_FUNC(activation_function_, TYPE_T)(TYPE_T value, float m, float n)\
    { return fmin((TYPE_T)(m), fmax((TYPE_T)(0), value)); }

#elif defined ACTIVATION_FUNCTION_LINEAR
#define ACTIVATION_FUNCTION(TYPE_T) \
inline TYPE_T CAT_FUNC(activation_function_, TYPE_T)(TYPE_T value, float m, float n)\
    { return (TYPE_T)(m) * value + (TYPE_T)(n); }

#else
#define ACTIVATION_FUNCTION(TYPE_T) \
inline TYPE_T CAT_FUNC(activation_function_, TYPE_T)(TYPE_T value, float m, float n)\
    { return value; }

#endif

ACTIVATION_FUNCTION(half)
ACTIVATION_FUNCTION(half2)
ACTIVATION_FUNCTION(half3)
ACTIVATION_FUNCTION(half4)
//ACTIVATION_FUNCTION(half5)
//ACTIVATION_FUNCTION(half6)
//ACTIVATION_FUNCTION(half7)
//ACTIVATION_FUNCTION(half8)
//ACTIVATION_FUNCTION(half9)
//ACTIVATION_FUNCTION(half10)
//ACTIVATION_FUNCTION(half11)
//ACTIVATION_FUNCTION(half12)
//ACTIVATION_FUNCTION(half13)
//ACTIVATION_FUNCTION(half14)
//ACTIVATION_FUNCTION(half15)
ACTIVATION_FUNCTION(half16)

ACTIVATION_FUNCTION(float)
ACTIVATION_FUNCTION(float2)
ACTIVATION_FUNCTION(float3)
ACTIVATION_FUNCTION(float4)
//ACTIVATION_FUNCTION(float5)
//ACTIVATION_FUNCTION(float6)
//ACTIVATION_FUNCTION(float7)
ACTIVATION_FUNCTION(float8)
//ACTIVATION_FUNCTION(float9)
//ACTIVATION_FUNCTION(float10)
//ACTIVATION_FUNCTION(float11)
//ACTIVATION_FUNCTION(float12)
//ACTIVATION_FUNCTION(float13)
//ACTIVATION_FUNCTION(float14)
//ACTIVATION_FUNCTION(float15)
//ACTIVATION_FUNCTION(float16)

inline DATA_TYPE FUNC(activation_function)(DATA_TYPE in_f, float m, float n)
{
    return CAT_FUNC_CALL(activation_function_, DATA_TYPE)(in_f, m ,n);
}
