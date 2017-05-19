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
#pragma once
#include "api_impl.h"
#include "refcounted_obj.h"
#include "gpu/ocl_toolkit.h"

namespace cldnn
{
struct event_impl: public refcounted_obj<event_impl>
{
public:
    event_impl(const cl::Event& event) : _event(event)
    {}

    void wait() const { _event.wait(); }
    virtual void set() { throw std::logic_error("cannot set OCL event"); }
    void add_event_handler(cldnn_event_handler handler, void* data)
    {
        std::lock_guard<std::mutex> lock(_handlers_mutex);
        if (_event.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>() == CL_COMPLETE)
        {
            handler(data);
        }
        else
        {
            if (_handlers.size() == 0)
            {
                _event.setCallback(CL_COMPLETE, callBack, this);
            }
            _handlers.push_back({ handler, data });
        }
    }
    cl::Event get() const { return _event; }
    const std::vector<cldnn_profiling_interval>& get_profiling_info();
protected:
    //TODO prevent long handler execution problem
    static void CL_CALLBACK callBack(cl_event, cl_int, void* me)
    {
        reinterpret_cast<event_impl*>(me)->callHandlers();
    }

    void callHandlers()
    {
        std::lock_guard<std::mutex> lock(_handlers_mutex);
        for (auto& pair : _handlers)
        {
            try
            {
                pair.first(pair.second);
            }
            catch (...) {}
        }
        _handlers.clear();
    }

    std::mutex _handlers_mutex;
    cl::Event _event;
    std::vector<std::pair<cldnn_event_handler, void*>> _handlers;
    std::vector<cldnn_profiling_interval> _profiling_info;
};

class user_event_gpu : public event_impl
{
public:
    user_event_gpu(const cl::UserEvent& event) : event_impl(event), _user_event(event) {}
    void set() override { _user_event.setStatus(CL_COMPLETE); }
private:
    cl::UserEvent _user_event;
};

}

API_CAST(::cldnn_event, cldnn::event_impl)
