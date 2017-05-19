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

#include "detection_output_inst.h"
#include "kernel.h"
#include "kd_selector.h"
#include "network_impl.h"
#include "implementation_map.h"

#include <algorithm>
#include <stdexcept>
#include <string>

using namespace cldnn;

namespace neural
{

struct bounding_box
{
	float xmin;
	float ymin;
	float xmax;
	float ymax;

	bounding_box() : xmin(0), ymin(0), xmax(0), ymax(0) {}

	// Computes the area of a bounding box.
	float area() const
	{
		return (xmax - xmin) * (ymax - ymin);
	}

	// Computes the intersection between 2 bounding boxes.
	bounding_box intersect(const bounding_box& other) const
	{
		bounding_box res;
		if (other.xmin > xmax || 
			other.xmax < xmin ||
			other.ymin > ymax || 
			other.ymax < ymin)
		{
			// There is no intersection.
			return res;
		}
		else
		{
			res.xmin = std::max(xmin, other.xmin);
			res.ymin = std::max(ymin, other.ymin);
			res.xmax = std::min(xmax, other.xmax);
			res.ymax = std::min(ymax, other.ymax);
			return res;
		}
	}

	// Computes the overlap between 2 bounding boxes.
	float overlap(const bounding_box& other) const
	{
		bounding_box intersect_bbox = intersect(other);
		float intersect_width = intersect_bbox.xmax - intersect_bbox.xmin;
		float intersect_height = intersect_bbox.ymax - intersect_bbox.ymin;
		if (intersect_width > 0 && intersect_height > 0)
		{
			float intersect_size = intersect_width * intersect_height;
			return intersect_size / (area() + other.area() - intersect_size);
		}
		else
		{
			return 0;
		}
	}
};

struct detection_output_gpu : typed_primitive_impl<detection_output>
{
    const detection_output_node& outer;

	detection_output_gpu(const detection_output_node& outer)
        : outer(outer)
    {}

	static void decode_bounding_box(
		const bounding_box& prior_bbox, const std::vector<float>& prior_variance,
		const cldnn::prior_box_code_type code_type, const bool variance_encoded_in_target,
		const bounding_box& bbox, bounding_box* decoded_bbox) 
	{
		switch (code_type)
		{
			case cldnn::prior_box_code_type::corner:
			{
				if (variance_encoded_in_target)
				{
					// variance is encoded in target, we simply need to add the offset predictions.
					decoded_bbox->xmin = prior_bbox.xmin + bbox.xmin;
					decoded_bbox->ymin = prior_bbox.ymin + bbox.ymin;
					decoded_bbox->xmax = prior_bbox.xmax + bbox.xmax;
					decoded_bbox->ymax = prior_bbox.ymax + bbox.ymax;
				}
				else
				{
					// variance is encoded in bbox, we need to scale the offset accordingly.
					decoded_bbox->xmin = prior_bbox.xmin + prior_variance[0] * bbox.xmin;
					decoded_bbox->ymin = prior_bbox.ymin + prior_variance[1] * bbox.ymin;
					decoded_bbox->xmax = prior_bbox.xmax + prior_variance[2] * bbox.xmax;
					decoded_bbox->ymax = prior_bbox.ymax + prior_variance[3] * bbox.ymax;
				}
				break;
			}
			case cldnn::prior_box_code_type::center_size:
			{
				float prior_width = prior_bbox.xmax - prior_bbox.xmin;
				assert(prior_width > 0);
				float prior_height = prior_bbox.ymax - prior_bbox.ymin;
				assert(prior_height > 0);
				float prior_center_x = (prior_bbox.xmin + prior_bbox.xmax) / 2.f;
				float prior_center_y = (prior_bbox.ymin + prior_bbox.ymax) / 2.f;
				float decode_bbox_center_x, decode_bbox_center_y;
				float decode_bbox_width, decode_bbox_height;
				if (variance_encoded_in_target)
				{
					// variance is encoded in target, we simply need to restore the offset predictions.
					decode_bbox_center_x = bbox.xmin * prior_width + prior_center_x;
					decode_bbox_center_y = bbox.ymin * prior_height + prior_center_y;
					decode_bbox_width = exp(bbox.xmax) * prior_width;
					decode_bbox_height = exp(bbox.ymax) * prior_height;
				}
				else
				{
					// variance is encoded in bbox, we need to scale the offset accordingly.
					decode_bbox_center_x = prior_variance[0] * bbox.xmin * prior_width + prior_center_x;
					decode_bbox_center_y = prior_variance[1] * bbox.ymin * prior_height + prior_center_y;
					decode_bbox_width = exp(prior_variance[2] * bbox.xmax) * prior_width;
					decode_bbox_height = exp(prior_variance[3] * bbox.ymax) * prior_height;
				}
				decoded_bbox->xmin = decode_bbox_center_x - decode_bbox_width / 2.f;
				decoded_bbox->ymin = decode_bbox_center_y - decode_bbox_height / 2.f;
				decoded_bbox->xmax = decode_bbox_center_x + decode_bbox_width / 2.f;
				decoded_bbox->ymax = decode_bbox_center_y + decode_bbox_height / 2.f;
				break;
			}
			case cldnn::prior_box_code_type::corner_size:
			{
				float prior_width = prior_bbox.xmax - prior_bbox.xmin;
				assert(prior_width > 0);
				float prior_height = prior_bbox.ymax - prior_bbox.ymin;
				assert(prior_height > 0);
				if (variance_encoded_in_target)
				{
					// variance is encoded in target, we simply need to add the offset predictions.
					decoded_bbox->xmin = prior_bbox.xmin + bbox.xmin * prior_width;
					decoded_bbox->ymin = prior_bbox.ymin + bbox.ymin * prior_height;
					decoded_bbox->xmax = prior_bbox.xmax + bbox.xmax * prior_width;
					decoded_bbox->ymax = prior_bbox.ymax + bbox.ymax * prior_height;
				}
				else
				{
					// variance is encoded in bbox, we need to scale the offset accordingly.
					decoded_bbox->xmin = prior_bbox.xmin + prior_variance[0] * bbox.xmin * prior_width;
					decoded_bbox->ymin = prior_bbox.ymin + prior_variance[1] * bbox.ymin * prior_height;
					decoded_bbox->xmax = prior_bbox.xmax + prior_variance[2] * bbox.xmax * prior_width;
					decoded_bbox->ymax = prior_bbox.ymax + prior_variance[3] * bbox.ymax * prior_height;
				}
				break;
			}
			default:
			{
				assert(0);
			}
		}
	}

	static void apply_nms(const std::vector<bounding_box>& bboxes,
		const std::vector<float>& scores, const float score_threshold,
		const float nms_threshold, const float eta, const int top_k,
		std::vector<int>& indices) 
	{
		assert(bboxes.size() == scores.size());

		// Get top_k scores higher than the threshold with their indices (score, index).
		std::vector<std::pair<float, int> > score_index_vec = {};
		for (int i = 0; i < (int)scores.size(); ++i)
		{
			if (scores[i] > score_threshold)
			{
				score_index_vec.push_back(std::make_pair(scores[i], i));
			}
		}
		// Sort the scores in descending order.
		std::stable_sort(score_index_vec.begin(), score_index_vec.end(), [](const std::pair<float, int>& p1, const std::pair<float, int>& p2) { return p1.first > p2.first; });
		// Keep top_k scores if needed.
		if (top_k > -1 && top_k < (int)score_index_vec.size())
		{
			score_index_vec.resize(top_k);
		}

		// NMS
		float adaptive_threshold = nms_threshold;
		indices.clear();
		while (score_index_vec.size() != 0)
		{
			const int idx = score_index_vec.front().second;
			bool keep = true;
			for (int k = 0; k < (int)indices.size(); ++k) 
			{
				if (keep) 
				{
					const int kept_idx = indices[k];
					float overlap = bboxes[idx].overlap(bboxes[kept_idx]);
					keep = (overlap <= adaptive_threshold);
				}
				else 
				{
					break;
				}
			}
			if (keep) 
			{
				indices.push_back(idx);
			}
			score_index_vec.erase(score_index_vec.begin());
			if (keep && eta < 1 && adaptive_threshold > 0.5) 
			{
				adaptive_threshold *= eta;
			}
		}
	}

	template<typename dtype>
	void generate_detections(detection_output_inst& instance, int num_of_images, const std::vector<std::map<int, std::vector<bounding_box> >>& all_bboxes, const std::vector<std::map<int, std::vector<float> > >& confidences)
	{
		cldnn::pointer<dtype> out_ptr = instance.output_memory().pointer<dtype>();
		auto& args = *outer.get_primitive();
		std::vector<std::map<int, std::vector<int> > > all_indices;
		for (int image = 0; image < num_of_images; ++image)
		{
			const std::map<int, std::vector<bounding_box> >& bboxes_per_image = all_bboxes[image];
			const std::map<int, std::vector<float> >& conf_per_image = confidences[image];
			std::map<int, std::vector<int> > indices; // class -> indices of bounding boxes
			int num_det = 0;
			for (uint32_t cls = 0; cls < args.num_classes; ++cls)
			{
				if ((int)cls == args.background_label_id)
				{
					continue; // Skip background class.
				}
				if (conf_per_image.find(cls) == conf_per_image.end())
				{
					assert(0); // No predictions for current label - shouldn't happen.
					continue;
				}
				const std::vector<float>& scores = conf_per_image.find(cls)->second;
				int label = args.share_location ? -1 : cls;
				if (bboxes_per_image.find(label) == bboxes_per_image.end())
				{
					assert(0); // No predictions for current label - shouldn't happen.
					continue;
				}
				apply_nms(bboxes_per_image.find(label)->second, scores, args.confidence_threshold, args.nms_threshold, args.eta, args.top_k, indices[cls]);
				num_det += (int)indices[cls].size();
			}
			if (num_det > args.keep_top_k)
			{
				std::vector<std::pair<float, std::pair<int, int> > > score_index_pairs;
				for (std::map<int, std::vector<int> >::iterator it = indices.begin(); it != indices.end(); ++it)
				{
					int label = it->first;
					const std::vector<int>& indices_per_label = it->second;
					if (conf_per_image.find(label) == conf_per_image.end())
					{
						assert(0); // No predictions for current label - shouldn't happen.
						continue;
					}
					const std::vector<float>& scores = conf_per_image.find(label)->second;
					for (int j = 0; j < (int)indices_per_label.size(); ++j)
					{
						int idx = indices_per_label[j];
						assert(idx < (int)scores.size());
						score_index_pairs.push_back(std::make_pair(scores[idx], std::make_pair(label, idx)));
					}
				}
				// Keep top k results per image.
				std::sort(score_index_pairs.begin(), score_index_pairs.end(), 
					[](const std::pair<float, std::pair<int, int>>& p1, const std::pair<float, std::pair<int, int>>& p2) { return p1.first > p2.first; });
				score_index_pairs.resize(args.keep_top_k);
				// Store the new indices.
				std::map<int, std::vector<int> > new_indices;
				for (int j = 0; j < (int)score_index_pairs.size(); ++j)
				{
					int label = score_index_pairs[j].second.first;
					int idx = score_index_pairs[j].second.second;
					new_indices[label].push_back(idx);
				}
				all_indices.push_back(new_indices);
			}
			else
			{
				all_indices.push_back(indices);
			}
		}

		int count = 0;
		for (int image = 0; image < num_of_images; ++image)
		{
			int saved_detections_per_image = 0;
			const std::map<int, std::vector<float> >& conf_scores = confidences[image];
			const std::map<int, std::vector<bounding_box> >& bboxes_per_image = all_bboxes[image];
			for (std::map<int, std::vector<int> >::iterator it = all_indices[image].begin(); it != all_indices[image].end(); ++it)
			{
				int label = it->first;
				if (conf_scores.find(label) == conf_scores.end())
				{
					assert(0); // No predictions for current label - shouldn't happen.
					continue;
				}
				const std::vector<float>& scores = conf_scores.find(label)->second;
				int loc_label = args.share_location ? -1 : label;
				if (bboxes_per_image.find(loc_label) == bboxes_per_image.end())
				{
					assert(0); // No predictions for current label - shouldn't happen.
					continue;
				}
				const std::vector<bounding_box>& bboxes = bboxes_per_image.find(loc_label)->second;
				std::vector<int>& indices = it->second;
				for (size_t i = 0; i < indices.size(); ++i)
				{
					int idx = indices[i];
					out_ptr[count * DETECTION_OUTPUT_ROW_SIZE] = (dtype)(float)image;
					out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 1] = (dtype)(float)label;
					out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 2] = (dtype)scores[idx];
					const bounding_box& bbox = bboxes[idx];
					out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 3] = (dtype)bbox.xmin;
					out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 4] = (dtype)bbox.ymin;
					out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 5] = (dtype)bbox.xmax;
					out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 6] = (dtype)bbox.ymax;
					++count;
					++saved_detections_per_image;
				}
			}
			//In case number of detections is smaller than keep_top_k fill the rest of the buffer with invalid image id (-1).
			for (auto j = saved_detections_per_image; j < args.keep_top_k; ++j)
			{
				out_ptr[count * DETECTION_OUTPUT_ROW_SIZE] = (dtype)-1.f;
				out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 1] = (dtype)0.f;
				out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 2] = (dtype)0.f;
				out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 3] = (dtype)0.f;
				out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 4] = (dtype)0.f;
				out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 5] = (dtype)0.f;
				out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 6] = (dtype)0.f;
				++count;
			}
		}
	}

	template<typename dtype>
	void extract_locations_per_image(detection_output_inst& instance, std::vector<std::map<int, std::vector<bounding_box> >>& locations, const int num_of_priors, const int num_loc_classes)
	{
		auto& args = *outer.get_primitive();
		const auto& input_location = instance.location_memory();
		auto location_ptr = input_location.pointer<dtype>();
		const dtype* location_data = location_ptr.data();
		const int num_of_images = (int)locations.size();

		for (int image = 0; image < num_of_images; ++image)
		{
			std::map<int, std::vector<bounding_box> >& label_to_bbox = locations[image];
			for (int prior = 0; prior < num_of_priors; ++prior)
			{
				int idx = prior * num_loc_classes * PRIOR_BOX_SIZE;
				for (int cls = 0; cls < num_loc_classes; ++cls)
				{
					int label = args.share_location ? -1 : cls;
					if (label_to_bbox.find(label) == label_to_bbox.end())
					{
						label_to_bbox[label].resize(num_of_priors);
					}
					label_to_bbox[label][prior].xmin = (float)(location_data[idx + cls * PRIOR_BOX_SIZE]);
					label_to_bbox[label][prior].ymin = (float)(location_data[idx + cls * PRIOR_BOX_SIZE + 1]);
					label_to_bbox[label][prior].xmax = (float)(location_data[idx + cls * PRIOR_BOX_SIZE + 2]);
					label_to_bbox[label][prior].ymax = (float)(location_data[idx + cls * PRIOR_BOX_SIZE + 3]);
				}
			}
			location_data += num_of_priors * num_loc_classes * PRIOR_BOX_SIZE;
		}
	}

	template<typename dtype>
	void extract_prior_boxes_and_variances(detection_output_inst& instance, std::vector<bounding_box>& prior_bboxes, std::vector<std::vector<float> >& prior_variances)
	{
		const auto& input_prior_box = instance.prior_box_memory();
		auto prior_box_ptr = input_prior_box.pointer<dtype>();
		const dtype* prior_box_data = prior_box_ptr.data();
		const int num_of_priors = (int)prior_bboxes.size();
		for (int prior = 0; prior < num_of_priors; ++prior)
		{
			int idx = prior * PRIOR_BOX_SIZE;
			bounding_box bbox;
			bbox.xmin = (float)(prior_box_data[idx]);
			bbox.ymin = (float)(prior_box_data[idx + 1]);
			bbox.xmax = (float)(prior_box_data[idx + 2]);
			bbox.ymax = (float)(prior_box_data[idx + 3]);
			prior_bboxes[prior] = bbox;
			idx += num_of_priors * PRIOR_BOX_SIZE;
			std::vector<float> var(PRIOR_BOX_SIZE);
			for (int j = 0; j < PRIOR_BOX_SIZE; ++j)
			{
				var[j] = (float)(prior_box_data[idx + j]);
			}
			prior_variances[prior] = var;
		}
	}

	template<typename dtype>
	void extract_confidences_per_image(detection_output_inst& instance, std::vector<std::map<int, std::vector<float> > >& confidences, const int num_of_priors)
	{
		auto args = *outer.get_primitive();
		const int num_of_images = (int)confidences.size();
		const auto& input_confidence = instance.confidence_memory();
		auto confidence_ptr = input_confidence.pointer<dtype>();
		const dtype* confidence_data = confidence_ptr.data();
		for (int image = 0; image < num_of_images; ++image)
		{
			std::map<int, std::vector<float> >& label_to_scores = confidences[image];
			for (int prior = 0; prior < num_of_priors; ++prior) {
				int idx = prior * args.num_classes;
				for (int cls = 0; cls < (int)args.num_classes; ++cls)
				{
					label_to_scores[cls].push_back((float)(confidence_data[idx + cls]));
				}
			}
			confidence_data += num_of_priors * args.num_classes;
		}
	}

	template<typename dtype>
	void prepare_data(detection_output_inst& instance, std::vector<std::map<int, std::vector<bounding_box>>> &bboxes, std::vector<std::map<int, std::vector<float> > >& confidences)
	{
		assert(bboxes.size() == confidences.size());

		auto& args = *outer.get_primitive();

		const int num_of_images = (int)bboxes.size();
		const int num_of_priors = instance.prior_box_memory().get_layout().size.spatial[1] / PRIOR_BOX_SIZE;
		const int num_loc_classes = args.share_location ? 1 : args.num_classes;

		// Extract locations per image.
		std::vector<std::map<int, std::vector<bounding_box> >> locations(num_of_images); // Per image : label -> bounding boxes.
		extract_locations_per_image<dtype>(instance, locations, num_of_priors, num_loc_classes);

		// Extract prior boxes - same within a batch.
		std::vector<bounding_box> prior_bboxes(num_of_priors); // Prior-Boxes (identical for all images since we assume all images in a batch are of same dimension).
		std::vector<std::vector<float> > prior_variances(num_of_priors); // Variances per prior-box (identical for all images since we assume all images in a batch are of same dimension).
		extract_prior_boxes_and_variances<dtype>(instance, prior_bboxes, prior_variances);

		// Create the decoded bounding boxes according to locations predictions and prior-boxes. 
		for (int image = 0; image < num_of_images; ++image)
		{
			std::map<int, std::vector<bounding_box> > & bboxes_per_image = bboxes[image];
			for (int cls = 0; cls < num_loc_classes; ++cls)
			{
				int label = args.share_location ? -1 : cls;
				if (label == args.background_label_id)
				{
					continue; // Skip background class.
				}
				if (locations[image].find(label) == locations[image].end())
				{
					assert(0); // No predictions for current label - shouldn't happen (locations[image] is constructed above and holds for each label a vector of prior-boxes).  
				}
				const std::vector<bounding_box>& label_loc_preds = locations[image].find(label)->second;
				assert(prior_bboxes.size() == label_loc_preds.size());
				bboxes_per_image[label].clear();
				for (int i = 0; i < (int)label_loc_preds.size(); ++i)
				{
					bounding_box decoded_bbox;
					decode_bounding_box(prior_bboxes[i], prior_variances[i], args.code_type, args.variance_encoded_in_target, label_loc_preds[i], &decoded_bbox);
					bboxes_per_image[label].push_back(decoded_bbox);
				}
			}
		}

		// Extract confidences per image.
		extract_confidences_per_image<dtype>(instance, confidences, num_of_priors);
	}

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, detection_output_inst& instance) override
    {
		for (auto& a : events) 
		{
			a->wait();
		}

		const int num_of_images = instance.location_memory().get_layout().size.batch[0]; //batch size
		
		std::vector<std::map<int, std::vector<bounding_box> >> bboxes(num_of_images); // Per image : label -> decoded bounding boxes.
		std::vector<std::map<int, std::vector<float> > > confidences(num_of_images); // Per image : class -> confidences per bounding box.

		if (instance.location_memory().get_layout().data_type == data_types::f32)
		{
			prepare_data<data_type_to_type<data_types::f32>::type>(instance, bboxes, confidences);

			generate_detections<data_type_to_type<data_types::f32>::type>(instance, num_of_images, bboxes, confidences);
		}
		else
		{
			prepare_data<data_type_to_type<data_types::f16>::type>(instance, bboxes, confidences);

			generate_detections<data_type_to_type<data_types::f16>::type>(instance, num_of_images, bboxes, confidences);
		}

		event_impl* ev = instance.get_network().get_engine().get()->create_user_event();
		ev->set();

		return ev;
    }

    static primitive_impl* create(const detection_output_node& arg)
	{
        return new detection_output_gpu(arg);
    }
};

namespace
{
    struct attach
    {
        attach()
        {
            implementation_map<detection_output>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::bfyx), detection_output_gpu::create);
			implementation_map<detection_output>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::bfyx), detection_output_gpu::create);
        }

        ~attach()
        {
        }
    };
    attach attach_impl;
}
}
