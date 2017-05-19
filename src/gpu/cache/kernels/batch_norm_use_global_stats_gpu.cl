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


#if FP16_UNIT_USED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

KERNEL (batch_norm_use_global_stats_gpu)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output, __global UNIT_TYPE* mean, __global UNIT_TYPE* variance)
{
    const uint feature_id = get_global_id(1);
    const uint linear_id = (uint)get_global_id(0) + feature_id * INPUT_BATCH_NUM + (uint)get_global_id(2) * INPUT_BATCH_NUM * INPUT_FEATURE_NUM;
    output[linear_id] = (input[linear_id] - mean[feature_id]) / (sqrt(variance[feature_id]) + EPSILON);
}