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


#if FP16_SUPPORTED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif


KERNEL (mean_subtract_gpu)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output, const __global UNIT_TYPE* mean)
{
    const uint batch_num = INPUT_BATCH_NUM;

    const uint global_id = get_global_id(0);
    const uint batch_id = global_id % batch_num;
    const uint feature_id = (global_id / batch_num) % INPUT_FEATURE_NUM;
    const uint x = ((global_id / batch_num) / INPUT_FEATURE_NUM) % INPUT_SIZE_X;
    const uint y = ((global_id / batch_num) / INPUT_FEATURE_NUM) / INPUT_SIZE_X;

#if BFYX_MEAN_FORMAT_USED
    // BFYX format of mean
    output[global_id] = input[global_id] - mean[x + MEAN_SIZE_X * (y + MEAN_SIZE_Y * (feature_id + MEAN_FEATURE_NUM * (batch_id % MEAN_BATCH_NUM)))];
#else
    // YXFB format of mean
    output[global_id] = input[global_id] - mean[(batch_id % MEAN_BATCH_NUM) + MEAN_BATCH_NUM * (feature_id + MEAN_FEATURE_NUM * (x + MEAN_SIZE_X * y))];
#endif
}