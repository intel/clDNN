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
#include "event_impl.h"
#include "engine_impl.h"

namespace cldnn
{

//class simple_user_event : public event_impl
//{
//public:
//    simple_user_event() :
//        _is_set(false)
//    {}
//
//    void wait() override
//    {
//        std::unique_lock<std::mutex> lock(_mutex);
//        if (_is_set) return;
//        _cv.wait(lock, [&] {return _is_set; });
//    }
//
//    void set() override
//    {
//        {
//            std::lock_guard<std::mutex> lock(_mutex);
//            _is_set = true;
//        }
//        _cv.notify_all();
//        for (auto& pair : _handlers)
//        {
//            pair.first(pair.second);
//        }
//    }
//
//    void add_event_handler(event::event_handler handler, void* data) override
//    {
//        if (handler == nullptr) throw std::invalid_argument("event handler");
//        _handlers.push_back({ handler, data });
//    }
//
//private:
//    bool _is_set;
//    std::mutex _mutex;
//    std::condition_variable _cv;
//    std::vector<std::pair<event::event_handler, void*>> _handlers;
//};

namespace
{
struct profiling_period_ocl_start_stop
{
    const char* name;
    cl_profiling_info start;
    cl_profiling_info stop;
};

bool is_event_profiled(const cl::Event& event)
{
    if (event() != nullptr)
    {
        auto queue = event.getInfo<CL_EVENT_COMMAND_QUEUE>();
        if(queue() != nullptr)
        {
            return (queue.getInfo<CL_QUEUE_PROPERTIES>() & CL_QUEUE_PROFILING_ENABLE) != 0;
        }
    }
    return false;
}
}

const std::vector<cldnn_profiling_interval>& event_impl::get_profiling_info()
{
    static const std::vector<profiling_period_ocl_start_stop> profiling_periods
    {
        { "submission", CL_PROFILING_COMMAND_QUEUED, CL_PROFILING_COMMAND_SUBMIT },
        { "starting",   CL_PROFILING_COMMAND_SUBMIT, CL_PROFILING_COMMAND_START },
        { "executing",  CL_PROFILING_COMMAND_START,  CL_PROFILING_COMMAND_END },
    };

    if (_profiling_info.empty())
    {
        for (auto& pp : profiling_periods)
        {
            _profiling_info.push_back({ pp.name, 0 });
        }
        if (is_event_profiled(_event))
        {
            for (size_t i = 0; i < profiling_periods.size(); i++)
            {
                cl_ulong start;
                _event.getProfilingInfo(profiling_periods[i].start, &start);
                cl_ulong end;
                _event.getProfilingInfo(profiling_periods[i].stop, &end);
                _profiling_info[i].nanoseconds = end - start;
            }
        }
    }
    return _profiling_info;
}


}
