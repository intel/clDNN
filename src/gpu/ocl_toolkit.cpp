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
#include "ocl_base_event.h"
#include "ocl_user_event.h"

#include <cassert>

namespace {
    std::string ndrange_to_string(cl::NDRange const& range)
    {
        std::string ret = "(";
        for (cl::size_type i = 0; i < range.dimensions(); ++i)
            ret += (!i ? "" : ", ") + std::to_string(range.get()[i]);

        ret += ")";
        return ret;
    }

    std::string events_list_to_string(std::vector<cldnn::event_impl::ptr> events)
    {
        std::string ret = "(";
        bool empty = true;
        for (auto& ev : events)
        {
            std::string id = "unk";
            if (auto* ocl_ev = dynamic_cast<cldnn::gpu::base_event*>(ev.get()))
                id = std::to_string(ocl_ev->get_queue_stamp());

            ret += (empty ? "" : ", ") + id;
            empty = false;
        }

        ret += ")";
        return ret;
    }
}

namespace cldnn { namespace gpu {

ocl_error::ocl_error(cl::Error const & err) : error(err.what() + std::string(", error code: ") + std::to_string(err.err()))
{
}

namespace {

    cl_device_type convert_configuration_device_type(configuration::device_types device_type)
    {
        cl_device_type device_types[] = {
                CL_DEVICE_TYPE_DEFAULT,
                CL_DEVICE_TYPE_CPU,
                CL_DEVICE_TYPE_GPU,
                CL_DEVICE_TYPE_ACCELERATOR };
        return device_types[device_type];
    }

    bool does_device_match_config(cl::Device const& dev, configuration const& config, std::list<std::string>& reasons)
    {
        auto dev_name = dev.getInfo<CL_DEVICE_NAME>();
        bool ok = true;

        auto dev_type = dev.getInfo<CL_DEVICE_TYPE>();

        if (dev_type != convert_configuration_device_type(config.device_type))
        {
            reasons.push_back(dev_name + ": invalid device type");
            ok = false;
        }

        auto vendor_id = dev.getInfo<CL_DEVICE_VENDOR_ID>();
        if (vendor_id != config.device_vendor)
        {
            reasons.push_back(dev_name + ": invalid vendor type");
            ok = false;
        }

        if (config.host_out_of_order)
        {
            auto queue_properties = dev.getInfo<CL_DEVICE_QUEUE_PROPERTIES>();
            using cmp_t = std::common_type_t<decltype(queue_properties), std::underlying_type_t<cl::QueueProperties>>;
            if (!(static_cast<cmp_t>(queue_properties) & static_cast<cmp_t>(cl::QueueProperties::OutOfOrder)))
            {
                reasons.push_back(dev_name + ": missing out of order support");
                ok = false;
            }
        }

        return ok;
    }

    bool starts_with(std::string const& str, std::string const& prefix)
    {
        if (str.size() < prefix.size())
            return false;

        return std::equal(prefix.begin(), prefix.end(), str.begin());
    }
}

cl::Device get_gpu_device(const configuration& config)
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    std::list<std::string> reasons;

    for (auto& p : platforms)
    {
        std::vector<cl::Device> devices;
        p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        for (auto& d : devices)
        {
            if (does_device_match_config(d, config, reasons))
                return d;
        }
    }

    if (reasons.empty())
        throw std::runtime_error("Could not find any OpenCL device");

    std::string error_msg = "No OpenCL device found which would match provided configuration:";
    for (const auto& reason : reasons)
        error_msg += "\n    " + reason;

    throw std::invalid_argument(std::move(error_msg));
}

std::shared_ptr<gpu_toolkit> gpu_toolkit::create(const configuration & cfg)
{
    struct make_shared_wa : public gpu_toolkit { make_shared_wa(const configuration& cfg) : gpu_toolkit(cfg) {} };
    try {
        return std::make_shared<make_shared_wa>(cfg);
    }
    catch (cl::Error const& err) {
        throw ocl_error(err);
    }
}

gpu_toolkit::gpu_toolkit(const configuration& config) 
    : _configuration(config)
    , _device(get_gpu_device(config))
    , _context(_device)
    , _command_queue(_context,
                     _device,
                     (config.enable_profiling
                        ? cl::QueueProperties::Profiling
                        : cl::QueueProperties::None) | 
                     (config.host_out_of_order
                        ? cl::QueueProperties::OutOfOrder
                        : cl::QueueProperties::None))
    , _engine_info(*this)
    , _kernels_cache(*this)
{
    _device.getInfo(CL_DEVICE_EXTENSIONS, &_extensions);
    if (logging_enabled())
    {
        open_log()
            << "Engine configuration:\n"
            << "    profiling: " + std::to_string(_configuration.enable_profiling) << "\n"
            << "    meaningful names: " + std::to_string(_configuration.enable_profiling) << "\n"
            << "    dump custom program: " + std::to_string(_configuration.enable_profiling) << "\n"
            << "    device type: " + std::to_string(_configuration.enable_profiling) << "\n"
            << "    vendor type: " + std::to_string(_configuration.enable_profiling) << "\n"
            << "    compiler options: " + _configuration.enable_profiling << "\n"
            << "    single kernel name: " + _configuration.enable_profiling << "\n"
            << "    out-of-order: " + std::to_string(_configuration.host_out_of_order) << "\n"
            << "    engine log: " + _configuration.log << "\n"
            << "    sources dumps: " + _configuration.ocl_sources_dumps_dir << "\n"
            << "\nEngine info:\n"
            << "    configuration: " + std::to_string(_engine_info.configuration) << "\n"
            << "    model: " + std::to_string(_engine_info.model) << "\n"
            << "    architecture: " + std::to_string(_engine_info.architecture) << "\n"
            << "    cores count: " + std::to_string(_engine_info.cores_count) << "\n"
            << "    core frequencey: " + std::to_string(_engine_info.core_frequency) << "\n"
            << "    max work group size: " + std::to_string(_engine_info.max_work_group_size) << "\n"
            << "    local memory size: " + std::to_string(_engine_info.max_local_mem_size) << "\n"
            << "    fp16: " + std::to_string(_engine_info.supports_fp16) << "\n"
            << "    fp16 denorms: " + std::to_string(_engine_info.supports_fp16_denorms) << "\n"
            << "    subgroups short: " + std::to_string(_engine_info.supports_subgroups_short) << "\n"
            << std::endl;
    }
}

event_impl::ptr gpu_toolkit::enqueue_kernel(cl::Kernel const& kern, cl::NDRange const& global, cl::NDRange const& local, std::vector<event_impl::ptr> const & deps)
{
    std::vector<cl::Event> dep_events;
    auto dep_events_ptr = &dep_events;
    if (!_configuration.host_out_of_order)
    {
        for (auto& dep : deps)
            if (auto ocl_ev = dynamic_cast<base_event*>(dep.get()))
                dep_events.push_back(ocl_ev->get());
    }
    else
    {
        dep_events_ptr = nullptr;
        sync_events(deps);
    }

    cl::Event ret_ev;
    try {
        _command_queue.enqueueNDRangeKernel(kern, cl::NullRange, global, local, dep_events_ptr, &ret_ev);
    }
    catch (cl::Error const& err) {
        throw ocl_error(err);
    }

    if (logging_enabled())
    {
        auto msg = kern.getInfo<CL_KERNEL_FUNCTION_NAME>() + ", gws: " + ndrange_to_string(global) + ", lws: " + ndrange_to_string(local) + ", deps: ";
        if (_configuration.host_out_of_order)
            msg += "()";
        else
            msg += events_list_to_string(deps);

        log(_queue_counter + 1, msg);
    }

    return{ new base_event(shared_from_this(), ret_ev, ++_queue_counter), false };
}

event_impl::ptr gpu_toolkit::enqueue_marker(std::vector<event_impl::ptr> const& deps)
{
    if (deps.empty())
        return{ new user_event(shared_from_this(), true), false };

    if (!_configuration.host_out_of_order)
    {
        cl::Event ret_ev;
        if (!enabled_single_kernel())
        {
            std::vector<cl::Event> dep_events;
            for (auto& dep : deps)
                if (auto ocl_ev = dynamic_cast<base_event*>(dep.get()))
                    dep_events.push_back(ocl_ev->get());

            try {
                _command_queue.enqueueMarkerWithWaitList(&dep_events, &ret_ev);
            } 
            catch (cl::Error const& err) {
                throw ocl_error(err);
            }
        }
        else
        {
            try {
                _command_queue.enqueueMarkerWithWaitList(nullptr, &ret_ev);
            }
            catch (cl::Error const& err) {
                throw ocl_error(err);
            }
        }

        if (logging_enabled())
            log(_queue_counter + 1, "Marker with dependencies: " + events_list_to_string(deps));

        return{ new base_event(shared_from_this(), ret_ev, ++_queue_counter), false };
    }
    else
    {
        sync_events(deps);
        assert(_last_barrier_ev() != nullptr);
        return{ new base_event(shared_from_this(), _last_barrier_ev, _last_barrier), false };
    }
}

void gpu_toolkit::flush()
{
    if (logging_enabled())
        log(0, "Flush");
    queue().flush();
}

void gpu_toolkit::wait_for_events(std::vector<event_impl::ptr> const & events)
{
    if (logging_enabled())
        log(0, "Wait for events: " + events_list_to_string(events));

    std::vector<cl::Event> clevents;
    for (auto& ev : events)
        if (auto ocl_ev = dynamic_cast<base_event*>(ev.get()))
            clevents.push_back(ocl_ev->get());

    try {
        cl::WaitForEvents(clevents);
    }
    catch (cl::Error const& err) {
        throw ocl_error(err);
    }
}

void gpu_toolkit::log(uint64_t id, std::string const & msg)
{
    if (_configuration.log.empty())
        return;

    open_log() << "[" << id << "] " << msg << std::endl;
}

void gpu_toolkit::sync_events(std::vector<event_impl::ptr> const & deps)
{
    if (!_configuration.host_out_of_order)
        return;

    bool needs_barrier = false;
    for (auto& dep : deps)
    {
        auto* ocl_ev = dynamic_cast<base_event*>(dep.get());
        if (ocl_ev->get_queue_stamp() > _last_barrier)
            needs_barrier = true;
    }

    if (needs_barrier)
    {
        try {
            _command_queue.enqueueBarrierWithWaitList(nullptr, &_last_barrier_ev);
        }
        catch (cl::Error const& err) {
            throw ocl_error(err);
        }

        _last_barrier = ++_queue_counter;
        if (logging_enabled())
            log(_last_barrier, "Barrier");
    }
}

std::ofstream& gpu_toolkit::open_log()
{
    if (!_log_file.is_initialized())
    {
        _log_file.emplace(_configuration.log, std::ios::out | std::ios::trunc);
        if (!_log_file.is_initialized())
            throw std::runtime_error("Could not initialize ocl_toolkit log file");
        if (!_log_file->is_open())
        {
            _log_file.reset();
            throw std::runtime_error("Could not open ocl_toolkit log file '" + _configuration.log + "' for writing");
        }
    }

    return _log_file.get();
}

}

}
