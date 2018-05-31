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
#include "kernels_cache.h"
#include "ocl_toolkit.h"
#include <algorithm>
#include <cassert>
#include <sstream>
#include <fstream>
#include <set>

#include "kernel_selector_helper.h"

#define MAX_KERNELS_PER_PROGRAM 10

namespace cldnn { namespace gpu {

namespace {
    std::string get_undef_jit(kernels_cache::source_code org_source_code)
    {
        const std::string white_space_with_new_lines = " \t\r\n";
        const std::string white_space = " \t";

        size_t current_pos = 0;

        const std::string define = "define";

        std::set<std::string> to_undef;
        for (const auto& source : org_source_code)
        {
            do
            {
                size_t index_to_hash = source.find_first_not_of(white_space_with_new_lines, current_pos);
                if (index_to_hash != std::string::npos &&
                    source[index_to_hash] == '#')
                {
                    size_t index_define = source.find_first_not_of(white_space, index_to_hash + 1);

                    if (index_define != std::string::npos &&
                        !source.compare(index_define, define.size(), define))
                    {
                        size_t index_to_name = source.find_first_not_of(white_space, index_define + define.size());
                        if (index_to_name != std::string::npos)
                        {
                            size_t index_to_end_name = source.find_first_of(white_space_with_new_lines + "(", index_to_name);
                            if (index_to_end_name == std::string::npos)
                            {
                                index_to_end_name = source.size();
                            }
                            std::string name = source.substr(index_to_name, index_to_end_name - index_to_name);
                            to_undef.insert(name);
                        }
                    }
                }

                current_pos = source.find_first_of('\n', current_pos + 1);
            } while (current_pos != std::string::npos);
        }

        std::string undefs;
        for (const auto& name : to_undef)
        {
            undefs += "#ifdef " + name + "\n";
            undefs += "#undef " + name + "\n";
            undefs += "#endif\n";
        }

        return std::move(undefs);
    }

    std::string reorder_options(const std::string& org_options)
    {
        std::stringstream ss(org_options);
        std::set<std::string> sorted_options;

        while (ss.good())
        {
            std::string word;
            ss >> word;
            sorted_options.insert(word);
        }

        std::string options;

        for (const auto& o : sorted_options)
        {
            options += o + " ";
        }

        return options;
    }

    inline bool does_options_support_batch_compilation(const std::string& options)
    {
        return
            options.find("-D") == std::string::npos &&
            options.find("-I") == std::string::npos;
    }
}

kernels_cache::sorted_code kernels_cache::get_program_source(const kernels_code& kernels_source_code) const
{
    sorted_code scode;

    for (const auto& code : kernels_source_code)
    {
        const source_code   org_source_code     = { code.second.kernel_strings->jit, code.second.kernel_strings->str };
        std::string         entry_point         = code.second.kernel_strings->entry_point;
        std::string         options             = code.second.kernel_strings->options;
        bool                batch_compilation   = code.second.kernel_strings->batch_compilation;
        bool                dump_custom_program = code.second.dump_custom_program;
		bool                one_time_kernel     = code.second.one_time_kernel;

        batch_compilation &= does_options_support_batch_compilation(options);

        if (batch_compilation)
        {
            options = reorder_options(options);
        }

        std::string key = options;

        if (batch_compilation == false)
        {
            key += " __PROGRAM__" + std::to_string(scode.size());
        }

        if (dump_custom_program)
        {
            key += " __DUMP_CUSTOM_PROGRAM__"; // Adding label to key so it would be separated from other programs
        }


		if (one_time_kernel)
        {
            key += " __ONE_TIME__";
        }

        auto& current_bucket = scode[key];
        current_bucket.dump_custom_program = dump_custom_program;
		current_bucket.one_time = one_time_kernel;

        if (current_bucket.source.empty())
        {
            current_bucket.options = options;
        }

        if ((current_bucket.kernels_counter % MAX_KERNELS_PER_PROGRAM) == 0)
        {
            current_bucket.source.push_back({});
        }

        current_bucket.entry_point_to_id[entry_point] = code.second.id;

        source_code new_source_code = org_source_code;

        if (batch_compilation)
        {
            new_source_code.push_back(get_undef_jit(org_source_code));
        }

        for (auto& s : new_source_code)
        {
            current_bucket.source.back().push_back(std::move(s));
        }

        current_bucket.kernels_counter++;
    }

    return std::move(scode);
}

kernels_cache::kernels_cache(gpu_toolkit& context) : _context(context) 
{
	set_exepath();
	set_cachemode();
}

kernels_cache::kernel_id kernels_cache::set_kernel_source(const std::shared_ptr<kernel_selector::kernel_string>& kernel_string, bool dump_custom_program, bool one_time_kernel)
{
    kernels_cache::kernel_id id;

    // same kernel_string == same kernel
    const auto key = kernel_string.get()->get_hash();

    std::lock_guard<std::mutex> lock(_mutex);

    const auto it = _kernels_code.find(key);

    if (it == _kernels_code.end())
    {
        // we need unique id in order to avoid conflict across topologies.
        const auto kernel_num = _kernels.size() + _kernels_code.size();
        id = kernel_string->entry_point + "_" + std::to_string(kernel_num);
        _kernels_code[key] = { kernel_string, id, dump_custom_program, one_time_kernel };
    }
    else
    {
        id = it->second.id;
    }

    assert(_kernels.find(id) == _kernels.end());
    _pending_compilation = true;
    return id;
}

kernels_cache::kernels_map kernels_cache::build_program(const program_code& program_source,bool bfause_build) const
{
    static uint32_t current_file_index = 0;

    bool dump_sources = !_context.get_configuration().ocl_sources_dumps_dir.empty() || program_source.dump_custom_program;

    std::string dump_file_name = "";
    if (dump_sources)
    {
        dump_file_name = _context.get_configuration().ocl_sources_dumps_dir;
        if (!dump_file_name.empty() && dump_file_name.back() != '/')
            dump_file_name += '/';

        dump_file_name += "clDNN_program_" + std::to_string(current_file_index++) + "_part_";
    }

    try
    {
        kernels_map kmap;
        std::string err_log; //accumulated build log from all program's parts (only contains messages from parts which failed to compile)

        uint32_t part_idx = 0;
		uint32_t totoal_source = 0;
		uint32_t curid = 0;
		if (CLCACHE_NOCACHE != _cur_cachemode) {
			for (const auto& sources : program_source.source) {
				for (const std::string& s : sources) {
					totoal_source += uint32_t(s.length());
				}
			}
		}
        for (const auto& sources : program_source.source)
        {
            auto current_dump_file_name = dump_file_name + std::to_string(part_idx++) + ".cl";
            boost::optional<std::ofstream> dump_file;

            if (dump_sources)
            {
                dump_file.emplace(current_dump_file_name);
                for (auto& s : sources)
                    dump_file.get() << s;
            }

            try
            {
				cl::vector<cl::Kernel> kernels;
				std::string binfilename = _fmodule_path + get_clcachename(totoal_source, program_source.kernels_counter, curid++, program_source.options, ".clbin");
				std::ifstream finbin(binfilename, std::ios::binary);

				if (!finbin || CLCACHE_NOCACHE == _cur_cachemode || bfause_build) {
					cl::Program program(_context.context(), sources);
					program.build({ _context.device() }, program_source.options.c_str());

					if (dump_sources)
					{
						dump_file.get() << "\n/* Build Log:\n";
						for (auto& p : program.getBuildInfo<CL_PROGRAM_BUILD_LOG>())
							dump_file.get() << p.second << "\n";

						dump_file.get() << "*/\n";
					}

					if (CLCACHE_CACHE == _cur_cachemode) {
						//--in cache mode, save binary
						std::vector<std::vector<unsigned char>> binariesVectors;
						if (CL_SUCCESS == program.getInfo(CL_PROGRAM_BINARIES, &binariesVectors)) {
							if (binariesVectors.size() != 1)
								throw std::runtime_error("Program get fail in CL_PROGRAM_BINARIES\n");
							std::ofstream fout(binfilename, std::ios::binary);
							fout.write((const char*)binariesVectors[0].data(), binariesVectors[0].size());
							fout.close();
						}
					}
					program.createKernels(&kernels);
				}
				else {
					cl::Program::Binaries binaries;
					std::vector<unsigned char> buf;
					finbin.seekg(0, std::ios::end);
					buf.resize(finbin.tellg());
					finbin.seekg(0, std::ios::beg);
					finbin.read((char*)buf.data(), buf.size());
					binaries.push_back(buf);
					cl::Program program(_context.context(), { _context.device() }, binaries);
					program.build({ _context.device() });
					program.createKernels(&kernels);
				}

                for (auto& k : kernels)
                {
                    auto kernel_name = k.getInfo<CL_KERNEL_FUNCTION_NAME>();
                    kmap.emplace(kernel_name, k);
                }
			}
            catch (const cl::BuildError& err)
            {
                if (dump_sources)
                    dump_file.get() << "\n/* Build Log:\n";

                for (auto& p : err.getBuildLog())
                {
                    if (dump_sources)
                        dump_file.get() << p.second << "\n";
                
                    err_log += p.second + '\n';
                }

                if (dump_sources)
                    dump_file.get() << "*/\n";
            }
            
        }

        if (!err_log.empty())
            throw std::runtime_error("Program build failed:\n" + std::move(err_log));

        return kmap;
    }
    catch (const cl::Error& err)
    {
        throw ocl_error(err);
    }
}

kernels_cache::kernel_type kernels_cache::get_kernel(kernel_id id, bool one_time_kernel)
{
    build_all();
    if (one_time_kernel)
    {
        return _one_time_kernels.at(id);
    }
    else
    {
        return _kernels.at(id);
    }
}

void kernels_cache::build_all()
{
    if (!_pending_compilation)
        return;

    std::lock_guard<std::mutex> lock(_mutex);

    auto sorted_program_code = get_program_source(_kernels_code);

	_one_time_kernels.clear();
    for (auto& program : sorted_program_code)
    {
		bool bfausebuild = false;
		for (uint32_t i = 0; i < 2; i++) 
		{
			kernels_map kernels = build_program(program.second, bfausebuild);

			for (auto& k : kernels)
			{
				const auto& entry_point = k.first;
				const auto& k_id = program.second.entry_point_to_id[entry_point];   //todo: need modify inference_engine to avoid k_id =0...
				if (k_id.length()==0) {  //if cache name not unique or different sequence, different batch, here will fail and rebuild.
					bfausebuild = true;
					break;
				}
				if (program.second.one_time)
				{
					_one_time_kernels[k_id] = k.second;
				}
				else
				{
					_kernels[k_id] = k.second;
				}
			}
			if (!bfausebuild)
				break;
		}
    }

    _kernels_code.clear();
    _pending_compilation = false;
}

std::string kernels_cache::get_clcachename(uint32_t total_source, uint32_t kernels_counter, uint32_t curid, const std::string& option, const std::string& extname) const
{
	std::string binfilename=std::to_string(total_source)+'_';
	binfilename += std::to_string(kernels_counter) + '_';
	binfilename += std::to_string(curid) + '_';
	binfilename += option;
	binfilename += extname;

	if (binfilename.length() > 255)
		binfilename = binfilename.substr(0, 255);
	return binfilename;
}

/*----------------------------------------------------------------------------------------
1. coder can add openvino_clcache\nocache.flag to disable cache,  openvino_clcache will create automaticlly
2. if not, work in cahce mode, the cl kernel will build fristly in end-user pc and save the binary
----------------------------------------------------------------------------------------*/
void kernels_cache::set_cachemode()
{
	std::ifstream fin(_fmodule_path + "nocache.flag");
	if (fin) {
		_cur_cachemode = CLCACHE_NOCACHE;
		fin.close();
	}
	else 
		_cur_cachemode = CLCACHE_CACHE;
}

static int global_cache = 0;
#ifdef _WIN32
#include <Windows.h>
void kernels_cache::set_exepath()
{
	char buf[255];
	GetModuleFileName(GetModuleHandle(NULL), buf, 255);
	std::string fpath(buf);
	//std::string::size_type idx = fpath.rfind('\\', fpath.length());
	std::string::size_type idx = fpath.rfind('.', fpath.length());
	_fmodule_path = fpath.substr(0, idx) + "_openvino_clcache\\";
	CreateDirectory(_fmodule_path.c_str(), NULL);
	_fmodule_path  = _fmodule_path + std::to_string(global_cache++) + "\\";
	CreateDirectory(_fmodule_path.c_str(), NULL);
}
#else

#endif

}}
