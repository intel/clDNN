/*
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
*/

inline int FUNC(imad_SW)(int acc, uchar4 input, char4 weight) __attribute__((overloadable)) {
    acc += input[0] * weight[0];
    acc += input[1] * weight[1];
    acc += input[2] * weight[2];
    acc += input[3] * weight[3];
    return acc;
}

inline int FUNC(imad_SW)(int acc, char4 input, char4 weight) __attribute__((overloadable)) {
    acc += input[0] * weight[0];
    acc += input[1] * weight[1];
    acc += input[2] * weight[2];
    acc += input[3] * weight[3];
    return acc;
}

// ## PROCESS PPC BEGIN (DPAS)

#if IMAD_SUPPORTED == 1

inline int FUNC(__imad)(int c, char4 a, char4 b) __attribute__((overloadable)) {
    int __builtin_IB_dp4a_ss(int c, int a, int b) __attribute__((const));
	return __builtin_IB_dp4a_ss(c, as_int(a), as_int(b));
}
inline int FUNC(__imad)(int c, uchar4 a, uchar4 b) __attribute__((overloadable)) {
    int __builtin_IB_dp4a_uu(int c, int a, int b) __attribute__((const));
	return __builtin_IB_dp4a_uu(c, as_int(a), as_int(b));
}
inline int FUNC(__imad)(int c, char4 a, uchar4 b) __attribute__((overloadable)) {
    int __builtin_IB_dp4a_su(int c, int a, int b) __attribute__((const));
	return __builtin_IB_dp4a_su(c, as_int(a), as_int(b));
}
inline int FUNC(__imad)(int c, uchar4 a, char4 b) __attribute__((overloadable)) {
    int __builtin_IB_dp4a_us(int c, int a, int b) __attribute__((const));
	return __builtin_IB_dp4a_us(c, as_int(a), as_int(b));
}
#define IMAD(_O, _I, _W) FUNC_CALL(__imad)(_O, _I, _W)


#else
// ## PROCESS PPC END

#define IMAD(_O, _I, _W) FUNC_CALL(imad_SW)(_O, _I, _W)

// ## PROCESS PPC BEGIN (DPAS)
#endif
// ## PROCESS PPC END