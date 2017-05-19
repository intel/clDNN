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

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "ocl_toolkit.h"

namespace neural { namespace gpu {

cl_device_type convert_configuration_device_type(configuration::device_types device_type) {
    cl_device_type device_types[] = {
            CL_DEVICE_TYPE_DEFAULT,
            CL_DEVICE_TYPE_CPU,
            CL_DEVICE_TYPE_GPU,
            CL_DEVICE_TYPE_ACCELERATOR };
    return device_types[device_type];
}

cl::Device get_gpu_device(const configuration& config) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Device default_device;
    for (auto& p : platforms) {
        std::vector<cl::Device> devices;
        p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        for (auto& d : devices) {
            if (d.getInfo<CL_DEVICE_TYPE>() == convert_configuration_device_type(config.device_type)) {
                auto vendor_id = d.getInfo<CL_DEVICE_VENDOR_ID>();
                //set Intel GPU device default
                if (vendor_id == config.device_vendor) {
                    return d;
                }
            }
        }
    }
    throw std::runtime_error("No OpenCL GPU device found.");
}

gpu_toolkit::gpu_toolkit(const configuration& config) 
    : _configuration(config)
    , _device(get_gpu_device(config))
    , _context(_device)
    , _command_queue(_context,
                     _device,
                     config.enable_profiling
                        ? cl::QueueProperties::Profiling
                        : cl::QueueProperties::None)
    , _engine_info(*this)
    , _kernels_cache(*this)
    {}

}}
