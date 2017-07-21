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

    bounding_box(const float xmin, const float ymin, const float xmax, const float ymax) : 
        xmin(xmin), ymin(ymin), xmax(xmax), ymax(ymax) {}

    // Computes the area of a bounding box.
    float area() const
    {
        return (xmax - xmin) * (ymax - ymin);
    }

    // Computes the intersection between 2 bounding boxes.
    bounding_box intersect(const bounding_box& other) const
    {
        return bounding_box(std::max(xmin, other.xmin), std::max(ymin, other.ymin), std::min(xmax, other.xmax), std::min(ymax, other.ymax));
    }

    // Computes the overlap between 2 bounding boxes.
    float overlap(const bounding_box& other) const
    {
        const bounding_box& intersect_bbox = intersect(other);
        const float intersect_width = intersect_bbox.xmax - intersect_bbox.xmin;
        const float intersect_height = intersect_bbox.ymax - intersect_bbox.ymin;
        if (intersect_width > 0 && intersect_height > 0)
        {
            const float intersect_size = intersect_width * intersect_height;
            return intersect_size / (area() + other.area() - intersect_size);
        }
        else
        {
            // There is no intersection.
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
        const bounding_box& prior_bbox, const std::array<float, PRIOR_BOX_SIZE>& prior_variance,
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
                const float prior_width = prior_bbox.xmax - prior_bbox.xmin;
                assert(prior_width > 0);
                const float prior_height = prior_bbox.ymax - prior_bbox.ymin;
                assert(prior_height > 0);
                const float prior_center_x = (prior_bbox.xmin + prior_bbox.xmax) / 2.f;
                const float prior_center_y = (prior_bbox.ymin + prior_bbox.ymax) / 2.f;
                float decode_bbox_center_x, decode_bbox_center_y;
                float decode_bbox_width, decode_bbox_height;
                if (variance_encoded_in_target)
                {
                    // variance is encoded in target, we simply need to restore the offset predictions.
                    decode_bbox_center_x = bbox.xmin * prior_width + prior_center_x;
                    decode_bbox_center_y = bbox.ymin * prior_height + prior_center_y;
                    decode_bbox_width = (exp(bbox.xmax) * prior_width) / 2.f;
                    decode_bbox_height = (exp(bbox.ymax) * prior_height) / 2.f;
                }
                else
                {
                    // variance is encoded in bbox, we need to scale the offset accordingly.
                    decode_bbox_center_x = prior_variance[0] * bbox.xmin * prior_width + prior_center_x;
                    decode_bbox_center_y = prior_variance[1] * bbox.ymin * prior_height + prior_center_y;
                    decode_bbox_width = (exp(prior_variance[2] * bbox.xmax) * prior_width) / 2.f;
                    decode_bbox_height = (exp(prior_variance[3] * bbox.ymax) * prior_height) / 2.f;
                }
                decoded_bbox->xmin = decode_bbox_center_x - decode_bbox_width;
                decoded_bbox->ymin = decode_bbox_center_y - decode_bbox_height;
                decoded_bbox->xmax = decode_bbox_center_x + decode_bbox_width;
                decoded_bbox->ymax = decode_bbox_center_y + decode_bbox_height;
                break;
            }
            case cldnn::prior_box_code_type::corner_size:
            {
                const float prior_width = prior_bbox.xmax - prior_bbox.xmin;
                assert(prior_width > 0);
                const float prior_height = prior_bbox.ymax - prior_bbox.ymin;
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
        const int scores_size = (int)scores.size();
        score_index_vec.reserve(scores_size);
        for (int i = 0; i < scores_size; ++i)
        {
            const float score = scores[i];
            if (score > score_threshold)
            {
                score_index_vec.emplace_back(std::make_pair(score, i));
            }
        }

        // Sort the scores in descending order and keep top_k scores if needed.
        if ((top_k != -1) && ((int)score_index_vec.size() > top_k))
        {
            std::partial_sort(score_index_vec.begin(), score_index_vec.begin() + top_k, score_index_vec.end(), [](const std::pair<float, int>& p1, const std::pair<float, int>& p2) { return (p1.first > p2.first) || (p1.first == p2.first && p1.second < p2.second); });
            score_index_vec.resize(top_k);
        }
        else
        {
            std::stable_sort(score_index_vec.begin(), score_index_vec.end(), [](const std::pair<float, int>& p1, const std::pair<float, int>& p2) { return p1.first > p2.first; });
        }

        // NMS
        float adaptive_threshold = nms_threshold;
        indices.reserve(score_index_vec.size());

        for(auto score_index : score_index_vec)
        {
            const int idx = score_index.second;
            bool keep = true;
            for (int k = 0; k < (int)indices.size(); ++k) 
            {
                if (keep) 
                {
                    const int kept_idx = indices[k];
                    const float overlap = bboxes[idx].overlap(bboxes[kept_idx]);
                    keep = (overlap <= adaptive_threshold);
                }
                else 
                {
                    break;
                }
            }
            if (keep) 
            {
                indices.emplace_back(idx);
            }
            if (keep && eta < 1 && adaptive_threshold > 0.5) 
            {
                adaptive_threshold *= eta;
            }
        }
    }

    template<typename dtype>
    void generate_detections(const detection_output_inst& instance, const int num_of_images, const std::vector<std::map<int, std::vector<bounding_box> >>& all_bboxes, const std::vector<std::map<int, std::vector<float> > >& confidences)
    {
        cldnn::pointer<dtype> out_ptr = instance.output_memory().pointer<dtype>();
        const auto& args = instance.argument;
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
                const auto& conf_cls = conf_per_image.find(cls);
                if (conf_cls == conf_per_image.end())
                {
                    assert(0); // No predictions for current label - shouldn't happen.
                    continue;
                }
                const std::vector<float>& scores = conf_cls->second;
                const int label = args.share_location ? -1 : cls;
                const auto& bbox_label = bboxes_per_image.find(label);
                if (bbox_label == bboxes_per_image.end())
                {
                    assert(0); // No predictions for current label - shouldn't happen.
                    continue;
                }
                apply_nms(bbox_label->second, scores, args.confidence_threshold, args.nms_threshold, args.eta, args.top_k, indices[cls]);
                num_det += (int)indices[cls].size();
            }
            if (num_det > args.keep_top_k)
            {
                std::vector<std::pair<float, std::pair<int, int> > > score_index_pairs;
                score_index_pairs.reserve(num_det);
                for (std::map<int, std::vector<int> >::iterator it = indices.begin(); it != indices.end(); ++it)
                {
                    int label = it->first;
                    const std::vector<int>& indices_per_label = it->second;
                    const auto& conf_label = conf_per_image.find(label);
                    if (conf_label == conf_per_image.end())
                    {
                        assert(0); // No predictions for current label - shouldn't happen.
                        continue;
                    }
                    const std::vector<float>& scores = conf_label->second;
                    for (int j = 0; j < (int)indices_per_label.size(); ++j)
                    {
                        int idx = indices_per_label[j];
                        assert(idx < (int)scores.size());
                        score_index_pairs.emplace_back(std::make_pair(scores[idx], std::make_pair(label, idx)));
                    }
                }

                // Keep top k results per image.
                auto sort_function = [](const std::pair<float, std::pair<int, int>>& p1, const std::pair<float, std::pair<int, int>>& p2) { return p1.first > p2.first; };
                if ((int)score_index_pairs.size() > args.keep_top_k)
                {
                    std::partial_sort(score_index_pairs.begin(), score_index_pairs.begin() + args.keep_top_k, score_index_pairs.end(), sort_function);
                    score_index_pairs.resize(args.keep_top_k);
                }
                else
                {
                    std::sort(score_index_pairs.begin(), score_index_pairs.end(), sort_function);
                }

                // Store the new indices.
                std::map<int, std::vector<int> > new_indices;
                for (int j = 0; j < (int)score_index_pairs.size(); ++j)
                {
                    int label = score_index_pairs[j].second.first;
                    int idx = score_index_pairs[j].second.second;
                    new_indices[label].emplace_back(idx);
                }
                all_indices.emplace_back(new_indices);
            }
            else
            {
                all_indices.emplace_back(indices);
            }
        }

        int count = 0;
        for (int image = 0; image < num_of_images; ++image)
        {
            int saved_detections_per_image = 0;
            const std::map<int, std::vector<float> >& conf_scores = confidences[image];
            const std::map<int, std::vector<bounding_box> >& bboxes_per_image = all_bboxes[image];
            auto& all_indices_per_image = all_indices[image];
            for (std::map<int, std::vector<int> >::iterator it = all_indices_per_image.begin(); it != all_indices_per_image.end(); ++it)
            {
                int label = it->first;
                const auto& conf_label = conf_scores.find(label);
                if (conf_label == conf_scores.end())
                {
                    assert(0); // No predictions for current label - shouldn't happen.
                    continue;
                }
                const std::vector<float>& scores = conf_label->second;
                int loc_label = args.share_location ? -1 : label;
                const auto& bbox_label = bboxes_per_image.find(loc_label);
                if (bbox_label == bboxes_per_image.end())
                {
                    assert(0); // No predictions for current label - shouldn't happen.
                    continue;
                }
                const std::vector<bounding_box>& bboxes = bbox_label->second;
                const std::vector<int>& indices = it->second;
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

    // Compute the linear index taking the padding into account.
    static inline int get_linear_feature_index(const int batch_id, const int feature_id, const int input_buffer_size_f, const int input_buffer_size_y, 
        const int input_buffer_size_x, const int input_padding_lower_y, const int input_padding_lower_x)
    {
        // This helper function assumes input layout with x_size = 1 and y_size = 1;
        // Location and confidence inputs should be tensors with size {b,f,1,1}.
        // This is validated in detection output primitive instance creation.

        int input_idx = (batch_id * input_buffer_size_f + feature_id) * input_buffer_size_y * input_buffer_size_x;
        input_idx += input_padding_lower_y * input_buffer_size_x + input_padding_lower_x;

        return input_idx;
    }

    template<typename dtype>
    void extract_locations_per_image(const detection_output_inst& instance, std::vector<std::map<int, std::vector<bounding_box> >>& locations, const int num_of_priors, const int num_loc_classes)
    {
        const bool share_location = instance.argument.share_location;
        const auto& input_location = instance.location_memory();
        const auto location_ptr = input_location.pointer<dtype>();
        const dtype* location_data = location_ptr.data();
        const int num_of_images = (int)locations.size();

        assert(num_of_priors * num_loc_classes * PRIOR_BOX_SIZE == input_location.get_layout().size.feature[0]);

        const auto& input_buffer_size = input_location.get_layout().get_buffer_size();
        const int input_buffer_size_x = input_buffer_size.spatial[0];
        const int input_buffer_size_y = input_buffer_size.spatial[1];
        const int input_buffer_size_f = input_buffer_size.feature[0];
        const auto& input_padding = input_location.get_layout().data_padding;
        const int input_padding_lower_x = input_padding.lower_size().spatial[0];
        const int input_padding_lower_y = input_padding.lower_size().spatial[1];

        for (int image = 0; image < num_of_images; ++image)
        {
            std::map<int, std::vector<bounding_box> >& label_to_bbox = locations[image];

            for (int cls = 0; cls < num_loc_classes; ++cls)
            {
                int label = share_location ? -1 : cls;
                auto & bboxes = label_to_bbox[label];
                bboxes.resize(num_of_priors);

                for (int prior = 0; prior < num_of_priors; ++prior)
                {
                    int idx = prior * num_loc_classes * PRIOR_BOX_SIZE;
                    bboxes[prior].xmin = (float)(location_data[get_linear_feature_index(image, idx + cls * PRIOR_BOX_SIZE, input_buffer_size_f, input_buffer_size_y,
                                                                                        input_buffer_size_x, input_padding_lower_y, input_padding_lower_x)]);
                    bboxes[prior].ymin = (float)(location_data[get_linear_feature_index(image, idx + cls * PRIOR_BOX_SIZE + 1, input_buffer_size_f, input_buffer_size_y,
                                                                                        input_buffer_size_x, input_padding_lower_y, input_padding_lower_x)]);
                    bboxes[prior].xmax = (float)(location_data[get_linear_feature_index(image, idx + cls * PRIOR_BOX_SIZE + 2, input_buffer_size_f, input_buffer_size_y,
                                                                                        input_buffer_size_x, input_padding_lower_y, input_padding_lower_x)]);
                    bboxes[prior].ymax = (float)(location_data[get_linear_feature_index(image, idx + cls * PRIOR_BOX_SIZE + 3, input_buffer_size_f, input_buffer_size_y,
                                                                                        input_buffer_size_x, input_padding_lower_y, input_padding_lower_x)]);
                }   
            }
        }
    }

    template<typename dtype>
    void extract_prior_boxes_and_variances(const detection_output_inst& instance, std::vector<bounding_box>& prior_bboxes, std::vector<std::array<float, PRIOR_BOX_SIZE>>& prior_variances)
    {
        const auto& input_prior_box = instance.prior_box_memory();
        const auto prior_box_ptr = input_prior_box.pointer<dtype>();
        const dtype* prior_box_data = prior_box_ptr.data();
        const int num_of_priors = (int)prior_bboxes.size();
        for (int prior = 0; prior < num_of_priors; ++prior)
        {
            int idx = prior * PRIOR_BOX_SIZE;
            prior_bboxes[prior] = bounding_box((float)(prior_box_data[idx]), (float)(prior_box_data[idx + 1]), (float)(prior_box_data[idx + 2]), (float)(prior_box_data[idx + 3]));
            idx += num_of_priors * PRIOR_BOX_SIZE;
            for (int j = 0; j < PRIOR_BOX_SIZE; ++j)
            {
                prior_variances[prior][j] = (float)(prior_box_data[idx + j]);
            }
        }
    }

    template<typename dtype>
    void extract_confidences_per_image(const detection_output_inst& instance, std::vector<std::map<int, std::vector<float> > >& confidences, const int num_of_priors)
    {
        const int num_classes = instance.argument.num_classes;

        const int num_of_images = (int)confidences.size();
        const auto& input_confidence = instance.confidence_memory();
        const auto confidence_ptr = input_confidence.pointer<dtype>();
        const dtype* confidence_data = confidence_ptr.data();

        assert(num_of_priors * num_classes == input_confidence.get_layout().size.feature[0]);

        const auto& input_buffer_size = input_confidence.get_layout().get_buffer_size();
        const int input_buffer_size_x = input_buffer_size.spatial[0];
        const int input_buffer_size_y = input_buffer_size.spatial[1];
        const int input_buffer_size_f = input_buffer_size.feature[0];
        const auto& input_padding = input_confidence.get_layout().data_padding;
        const int input_padding_lower_x = input_padding.lower_size().spatial[0];
        const int input_padding_lower_y = input_padding.lower_size().spatial[1];

        for (int image = 0; image < num_of_images; ++image)
        {
            std::map<int, std::vector<float> >& label_to_scores = confidences[image];

            for (int cls = 0; cls < num_classes; ++cls)
            {
                auto & scores = label_to_scores[cls];
                scores.resize(num_of_priors);
                for (int prior = 0; prior < num_of_priors; ++prior)
                {
                    int idx = prior * num_classes;
                    scores[prior] = ((float)confidence_data[get_linear_feature_index(image, idx + cls, input_buffer_size_f, input_buffer_size_y,
                                                                                        input_buffer_size_x, input_padding_lower_y, input_padding_lower_x)]);
                }
            }

        }    
    }

    template<typename dtype>
    void prepare_data(const detection_output_inst& instance, std::vector<std::map<int, std::vector<bounding_box>>> &bboxes, std::vector<std::map<int, std::vector<float> > >& confidences)
    {
        assert(bboxes.size() == confidences.size());

        const auto& args = instance.argument;

        const int num_of_images = (int)bboxes.size();
        const int num_of_priors = instance.prior_box_memory().get_layout().size.spatial[1] / PRIOR_BOX_SIZE;
        const int num_loc_classes = args.share_location ? 1 : args.num_classes;

        // Extract locations per image.
        std::vector<std::map<int, std::vector<bounding_box> >> locations(num_of_images); // Per image : label -> bounding boxes.
        extract_locations_per_image<dtype>(instance, locations, num_of_priors, num_loc_classes);

        // Extract prior boxes - same within a batch.
        std::vector<bounding_box> prior_bboxes(num_of_priors); // Prior-Boxes (identical for all images since we assume all images in a batch are of same dimension).
        std::vector<std::array<float, PRIOR_BOX_SIZE>> prior_variances(num_of_priors); // Variances per prior-box (identical for all images since we assume all images in a batch are of same dimension).
        extract_prior_boxes_and_variances<dtype>(instance, prior_bboxes, prior_variances);

        // Create the decoded bounding boxes according to locations predictions and prior-boxes. 
        for (int image = 0; image < num_of_images; ++image)
        {
            std::map<int, std::vector<bounding_box> > & bboxes_per_image = bboxes[image];
            for (int cls = 0; cls < num_loc_classes; ++cls)
            {
                const int label = args.share_location ? -1 : cls;
                if (label == args.background_label_id)
                {
                    continue; // Skip background class.
                }
                if (locations[image].find(label) == locations[image].end())
                {
                    assert(0); // No predictions for current label - shouldn't happen (locations[image] is constructed above and holds for each label a vector of prior-boxes).  
                }
                const std::vector<bounding_box>& label_loc_preds = locations[image].find(label)->second;
                int label_loc_preds_size = (int)label_loc_preds.size();
                assert((int)prior_bboxes.size() == label_loc_preds_size);
                
                bboxes_per_image[label].clear();

                for (int i = 0; i < label_loc_preds_size; ++i)
                {
                    bounding_box decoded_bbox;
                    decode_bounding_box(prior_bboxes[i], prior_variances[i], args.code_type, args.variance_encoded_in_target, label_loc_preds[i], &decoded_bbox);
                    bboxes_per_image[label].emplace_back(decoded_bbox);
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
