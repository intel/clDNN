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


KERNEL(warm_up_gpu)(int c, int a, int b, __global int* out)
{
    int res = (get_global_id(0) * a + get_global_id(1)) * b + get_global_id(2);
    if(a >> 3)
        res += get_local_id(1);
    if(c)
        out[get_local_id(0)] = res;
}