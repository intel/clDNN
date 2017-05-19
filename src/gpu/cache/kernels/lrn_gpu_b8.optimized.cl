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


__attribute__((reqd_work_group_size(8, 1, 1)))
KERNEL (lrn_gpu_b8)(const __global float* input, __global float* output)
{
    const uint global_id = get_global_id(0);
    const uint element_offset = (uint)get_global_id(1) * (INPUT_BATCH_NUM/8) * INPUT_FEATURE_NUM;
    
    const uint linear_id = global_id + element_offset;
    float8 acc = 0;

    int input_offset_f = global_id + HELP_INPUT_OFFSET * (INPUT_BATCH_NUM / 8);
    int input_idx = input_offset_f + element_offset;
    for (int i = 0; i < P_SIZE; i++)
    {
        bool zero = input_offset_f < 0 || input_offset_f >= INPUT_FEATURE_NUM * (INPUT_BATCH_NUM/8);

        if(!zero)
        {
            float8 value = vload8(input_idx, input);
            acc = mad(value, value, acc);
        }

        input_offset_f+= INPUT_BATCH_NUM/8;
        input_idx += INPUT_BATCH_NUM/8;
    }
    acc = mad(acc, ALPHA_DIV_BY_SIZE, K);
    acc = native_powr(acc, -BETA);

    float8 _in = vload8(linear_id, input);
    vstore8(acc * _in, linear_id, output);
}