/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Modified from https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/blob/master/CUDA-BEVFusion/src/common/visualize.hpp
// Add functions to visualize BBOX predictions and planning trajectories in the ImageArtist and BEVArtist

#include <cuda_fp16.h>
#include <cuosd.h>
#include <math.h>
#include <string.h>
#include <algorithm>
#include <vector>
#include <tgmath.h>
#include <utility>
#include "tensor.hpp"
#include "launch.cuh"
#include "dtype.hpp"
#include "launch.cuh"
#include "tensor.hpp"
#include "visualize.hpp"

namespace nv {

#define UseFont "../../../common/dependencies/cuOSD/data/simhei.ttf"
#define MaxDistance 50
#define INTER_RESIZE_COEF_BITS 11
#define INTER_RESIZE_COEF_SCALE (1 << INTER_RESIZE_COEF_BITS)
#define CAST_BITS (INTER_RESIZE_COEF_BITS << 1)

typedef std::tuple<std::vector<nvtype::Float2>, int, float, float, std::vector<float>> Box3DInfo;
std::vector<Box3DInfo> transformation_predictions(const nvtype::Float4* viewport_4x4,
                                                  const std::vector<Prediction>& predictions) {
  if (predictions.empty()) return {};

  const int number_of_corner = 8;
  std::vector<Box3DInfo> output;
  output.reserve(predictions.size());

  // 8 x 3
  const nvtype::Float3 offset_of_corners[number_of_corner] = {{-1, -1, -1}, {+1, -1, -1}, {+1, +1, -1}, {-1, +1, -1},
                                                              {-1, -1, +1}, {+1, -1, +1}, {+1, +1, +1}, {-1, +1, +1}};

  for (size_t idx_predict = 0; idx_predict < predictions.size(); ++idx_predict) {
    auto& item = predictions[idx_predict];
    float cos_rotation = cos(item.z_rotation);
    float sin_rotation = sin(item.z_rotation);

    std::vector<nvtype::Float2> box3d;
    box3d.reserve(number_of_corner);

    nvtype::Float4 row0 = viewport_4x4[0];
    nvtype::Float4 row1 = viewport_4x4[1];
    nvtype::Float4 row2 = viewport_4x4[2];
    float zdepth = item.position.x * row2.x + item.position.y * row2.y + item.position.z * row2.z + row2.w;

    std::vector<float> velocity;
    float cx = item.position.x, cy = item.position.y, cz = item.position.z;
    float original_x = cx * row0.x + cy * row0.y + cz * row0.z + row0.w;
    float original_y = cx * row1.x + cy * row1.y + cz * row1.z + row1.w;
    float original_weight = cx * row2.x + cy * row2.y + cz * row2.z + row2.w;
    original_weight = std::max(1e-5f, std::min(1e5f, original_weight));
    velocity.push_back(original_x / original_weight);
    velocity.push_back(original_y / original_weight);
    float delta_t = 1.;
    float vx = item.velocity.vx, vy = item.velocity.vy;
    float next_x = (cx+vx*delta_t) * row0.x + (cy+vy*delta_t) * row0.y + cz * row0.z + row0.w;
    float next_y = (cx+vx*delta_t) * row1.x + (cy+vy*delta_t) * row1.y + cz * row1.z + row1.w;
    float next_weight = (cx+vx*delta_t) * row2.x + (cy+vy*delta_t) * row2.y + cz * row2.z + row2.w;
    next_weight = std::max(1e-5f, std::min(1e5f, next_weight));
    velocity.push_back(next_x / next_weight);
    velocity.push_back(next_y / next_weight);

    for (int idx_corner = 0; idx_corner < number_of_corner; ++idx_corner) {
      auto& offset = offset_of_corners[idx_corner];
      nvtype::Float3 corner;
      nvtype::Float3 std_corner;
      std_corner.x = item.size.w * offset.x * 0.5f;
      std_corner.y = item.size.l * offset.y * 0.5f;
      std_corner.z = item.size.h * offset.z * 0.5f;

      corner.x = item.position.x + std_corner.x * cos_rotation + std_corner.y * sin_rotation;
      corner.y = item.position.y + std_corner.x * -sin_rotation + std_corner.y * cos_rotation;
      corner.z = item.position.z + std_corner.z;

      float image_x = corner.x * row0.x + corner.y * row0.y + corner.z * row0.z + row0.w;
      float image_y = corner.x * row1.x + corner.y * row1.y + corner.z * row1.z + row1.w;
      float weight = corner.x * row2.x + corner.y * row2.y + corner.z * row2.z + row2.w;

      if (image_x <= 0 || image_y <= 0 || weight <= 0) {
        break;
      }

      weight = std::max(1e-5f, std::min(1e5f, weight));
      box3d.emplace_back(image_x / weight, image_y / weight);
    }

    if (box3d.size() != number_of_corner) continue;

    output.emplace_back(box3d, item.id, item.score, zdepth, velocity);
  }

  std::sort(output.begin(), output.end(), [](const Box3DInfo& a, const Box3DInfo& b) { return std::get<3>(a) > std::get<3>(b); });
  return output;
}

class ImageArtistImplement : public ImageArtist {
 public:
  virtual ~ImageArtistImplement() {
    if (cuosd_) cuosd_context_destroy(cuosd_);
  }

  bool init(const ImageArtistParameter& param) {
    param_ = param;
    if (param_.classes.empty()) {
      // printf("Use default nuscenes classes configuration.\n");
      param_.classes = {{"car", 255, 158, 0}, {"truck", 255, 99, 71}, {"construction_vehicle", 233, 150, 70},
                        {"bus", 255, 69, 0}, {"trailer", 255, 140, 0}, {"barrier", 112, 128, 144},
                        {"motorcycle", 255, 61, 99}, {"bicycle", 220, 20, 60}, {"pedestrian", 0, 0, 230},
                        {"traffic_cone", 47, 79, 79}};
    }
    cuosd_ = cuosd_context_create();
    return cuosd_ != nullptr;
  }

  virtual void draw_prediction(int camera_index, const std::vector<std::vector<float>>& predictions, bool flipx) override {
    if (predictions.size() == 0) return;
    std::vector<Prediction> bbox_container;
    for (size_t i=0; i < predictions.size(); ++i) {
      Prediction obj;
      obj.position.x = predictions[i][0];
      obj.position.y = predictions[i][1];
      obj.position.z = predictions[i][2];
      obj.size.w = predictions[i][3];
      obj.size.l = predictions[i][4];
      obj.size.h = predictions[i][5];
      obj.z_rotation = predictions[i][6];
      obj.velocity.vx = predictions[i][7];
      obj.velocity.vy = predictions[i][8];
      obj.id = predictions[i][9]; // label
      obj.score = predictions[i][10];
      bbox_container.emplace_back(obj);
    }
    draw_prediction(camera_index, bbox_container, flipx);
  }

  virtual void draw_prediction(int camera_index, const std::vector<Prediction>& predictions, bool flipx) override {
    auto points = transformation_predictions(this->param_.viewport_nx4x4.data() + camera_index * 4, predictions);
    size_t num = points.size();
    for (size_t i = 0; i < num; ++i) {
      auto& item = points[i];
      auto& corners = std::get<0>(item);
      auto label = std::get<1>(item);
      auto score = std::get<2>(item);
      const int idx_of_line[][2] = {
          {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7},
      };

      NameAndColor* name_color = &default_name_color_;
      if (label >= 0 && label < static_cast<int>(param_.classes.size())) {
        name_color = &param_.classes[label];
      }

      float size = std::sqrt(std::pow(corners[6].x - corners[0].x, 2) + std::pow(corners[6].y - corners[0].y, 2));
      float minx = param_.image_width;
      float miny = param_.image_height;
      for (size_t ioff = 0; ioff < sizeof(idx_of_line) / sizeof(idx_of_line[0]); ++ioff) {
        auto p0 = corners[idx_of_line[ioff][0]];
        auto p1 = corners[idx_of_line[ioff][1]];
        p0.x *= 4.; p0.y *= 4.;
        p1.x *= 4.; p1.y *= 4.;
        if (flipx) {
          p0.x = param_.image_width - p0.x - 1;
          p1.x = param_.image_width - p1.x - 1;
        }
        minx = std::min(minx, std::min(p0.x, p1.x));
        miny = std::min(miny, std::min(p0.y, p1.y));
        cuosd_draw_line(cuosd_, p0.x, p0.y, p1.x, p1.y, 5, {name_color->r, name_color->g, name_color->b, 255});
      }

      size = std::max(size * 0.15f, 8.0f);
      auto title = nv::format("%.2f", score);
      cuosd_draw_text(cuosd_, title.c_str(), size, UseFont, minx, miny, {255, 255, 255, 255}, {0, 0, 0, 0});
    }
  }

  virtual void draw_planning_traj(int camera_index, const std::vector<std::pair<float, float>>& planning_traj, bool flipx) {
    std::vector<std::pair<float, float>> image_points;
    nvtype::Float4 row0 = (this->param_.viewport_nx4x4.data() + camera_index * 4)[0];
    nvtype::Float4 row1 = (this->param_.viewport_nx4x4.data() + camera_index * 4)[1];
    nvtype::Float4 row2 = (this->param_.viewport_nx4x4.data() + camera_index * 4)[2];
    float ground_height = -1.562;
    float init_image_x = ground_height * row0.z + row0.w;
    float init_image_y = ground_height * row1.z + row1.w;
    float init_weight = ground_height * row2.z + row2.w;
    if (init_image_x > 0 && init_image_y > 0 && init_weight > 0) {
      init_weight = std::max(1e-5f, std::min(1e5f, init_weight));
      image_points.push_back({init_image_x/init_weight, init_image_y/init_weight});
    }
    for (size_t i=0; i<planning_traj.size(); ++i) {
      float image_x = planning_traj[i].first * row0.x + planning_traj[i].second * row0.y + ground_height * row0.z + row0.w;
      float image_y = planning_traj[i].first * row1.x + planning_traj[i].second * row1.y + ground_height * row1.z + row1.w;
      float weight = planning_traj[i].first * row2.x + planning_traj[i].second * row2.y + ground_height * row2.z + row2.w;
      if (image_x <= 0 || image_y <= 0 || weight <= 0) {
        continue;
      }
      weight = std::max(1e-5f, std::min(1e5f, weight));
      image_points.push_back({image_x/weight, image_y/weight});
    }
    if (image_points.size() <= 1) return;
    for (size_t i=0; i<image_points.size()-1; ++i) {
      float p0x = image_points[i].first, p0y = image_points[i].second, p1x = image_points[i+1].first, p1y = image_points[i+1].second;
      p0x *= 4.; p0y *= 4.;
      p1x *= 4.; p1y *= 4.;
      if (flipx) {
        p0x = param_.image_width - p0x - 1;
        p1x = param_.image_width - p1x - 1;
      }
      cuosd_draw_line(cuosd_, p0x, p0y, p1x, p1y, 5, {0, 255, 0, 255});
    }
    return;
  }

  virtual void apply(unsigned char* image_rgb_device, void* stream) override {
    cuosd_apply(cuosd_, image_rgb_device, nullptr, param_.image_width, param_.image_stride, param_.image_height,
                cuOSDImageFormat::RGB, stream);
  }

 private:
  cuOSDContext_t cuosd_ = nullptr;
  ImageArtistParameter param_;
  NameAndColor default_name_color_{"Unknow", 0, 0, 0};
};

std::shared_ptr<ImageArtist> create_image_artist(const ImageArtistParameter& param) {
  std::shared_ptr<ImageArtistImplement> instance(new ImageArtistImplement());
  if (!instance->init(param)) {
    printf("Failed to create ImageArtist\n");
    instance.reset();
  }
  return instance;
}

typedef struct {
  half val[5];
} half5;

template <typename _T>
static __host__ __device__ _T limit(_T value, _T amin, _T amax) {
  return value < amin ? amin : (value > amax ? amax : value);
}

static __global__ void draw_point_to(unsigned int num, const half5* points, float4* view_port, unsigned char* image,
                                     int image_width, int stride, int image_height) {
  unsigned int idx = cuda_linear_index;
  if (idx >= num) return;

  half5 point = points[idx];
  float px = point.val[0];
  float py = point.val[1];
  float pz = point.val[2];
  float reflection = point.val[3];
  float indensity = point.val[4];

  float4 r0 = view_port[0];
  float4 r1 = view_port[1];
  float4 r2 = view_port[2];
  float x = px * r0.x + py * r0.y + pz * r0.z + r0.w;
  float y = px * r1.x + py * r1.y + pz * r1.z + r1.w;
  float w = px * r2.x + py * r2.y + pz * r2.z + r2.w;

  if (w <= 0) return;

  x = x / w;
  y = y / w;

  if (x < 0 || x >= image_width || y < 0 || y >= image_height) {
    return;
  }

  int ix = static_cast<int>(x);
  int iy = static_cast<int>(y);
  float alpha = limit((pz + 5.0f) / 8.0f, 0.35f, 1.0f);
  unsigned char gray = limit(alpha * 255, 0.0f, 255.0f);
  *(uchar3*)&image[iy * stride + ix * 3] = make_uchar3(gray, gray, gray);
}

static std::vector<nvtype::Float4> rodrigues_rotation(float radian, const std::vector<float>& axis){
  /*
     Rodrigues Rotation
  */
  std::vector<nvtype::Float4> output(4);
  memset(&output[0], 0, output.size() * sizeof(nvtype::Float4));

  float nx = axis[0];
  float ny = axis[1];
  float nz = axis[2];
  float cos_val = cos(radian);
  float sin_val = sin(radian);
  output[3].w = 1;

  float a = 1 - cos_val;
  float identity[3][3] = {
    {1, 0, 0},
    {0, 1, 0},
    {0, 0, 1},
  };

  float M[3][3] = {
    {0, -nz, ny},
    {nz, 0, -nx},
    {-ny, nx, 0}
  };

  for(int i = 0; i < 3; ++i){
    for(int j = 0; j < 3; ++j){
      ((float*)&output[i])[j] = cos_val * identity[i][j] + a * axis[i] * axis[j] + sin_val * M[i][j];
    }
  }
  return output;
}

std::vector<nvtype::Float4> matmul(const std::vector<nvtype::Float4>& a, const std::vector<nvtype::Float4>& b){
  std::vector<nvtype::Float4> c(a.size());
  memset(&c[0], 0, c.size() * sizeof(nvtype::Float4));

  for(size_t m = 0; m < a.size(); ++m){
    auto& ra = a[m];
    auto& rc = c[m];
    for(size_t n = 0; n < b.size(); ++n){
      for(size_t k = 0; k < 4; ++k){
        auto& rb = b[k];
        ((float*)&rc)[n] += ((float*)&ra)[k] * ((float*)&rb)[n];
      }
    }
  }
  return c;
}

struct BEVArtistDrawPointCommand {
  const nvtype::half* points_device;
  unsigned int number_of_points;
};

class BEVArtistImplement : public BEVArtist {
 public:
  virtual ~BEVArtistImplement() {
    if (transform_matrix_device_) checkRuntime(cudaFree(transform_matrix_device_));
    if (cuosd_) cuosd_context_destroy(cuosd_);
  }

  bool init(const BEVArtistParameter& param) {
    param_ = param;
    if (param_.classes.empty()) {
      // printf("Use default nuscenes classes configuration.\n");
      param_.classes = {{"car", 255, 158, 0}, {"truck", 255, 99, 71}, {"construction_vehicle", 233, 150, 70},
                        {"bus", 255, 69, 0}, {"trailer", 255, 140, 0}, {"barrier", 112, 128, 144},
                        {"motorcycle", 255, 61, 99}, {"bicycle", 220, 20, 60}, {"pedestrian", 0, 0, 230},
                        {"traffic_cone", 47, 79, 79}};
    }

    std::vector<nvtype::Float4> lidar2image = {{param_.norm_size / MaxDistance, 0, 0, param_.cx},
                     {0, -param_.norm_size / MaxDistance, 0, param_.cy},
                     {0, 0, 0, 1},
                     {0, 0, 0, 1}};

    transform_matrix_.resize(4);
    memset(&transform_matrix_[0], 0, sizeof(nvtype::Float4) * transform_matrix_.size());

    // auto rotation_x = rodrigues_rotation(param.rotate_x / 180.0f * 3.141592653f, {1, 0, 0});
    // auto rotation_z = rodrigues_rotation(10.0f / 180.0f * 3.141592653f, {0, 0, 1});
    auto rotation_x = rodrigues_rotation(0.f, {1, 0, 0});
    auto rotation_z = rodrigues_rotation(0.f, {0, 0, 1});
    transform_matrix_ = matmul(lidar2image, matmul(rotation_x, rotation_z));

    checkRuntime(cudaMalloc(&transform_matrix_device_, sizeof(nvtype::Float4) * transform_matrix_.size()));
    checkRuntime(cudaMemcpy(transform_matrix_device_, transform_matrix_.data(), sizeof(nvtype::Float4) * transform_matrix_.size(),
                            cudaMemcpyHostToDevice));
    cuosd_ = cuosd_context_create();
    return cuosd_ != nullptr;
  }

  virtual void draw_lidar_points(const nvtype::half* points_device, unsigned int number_of_points) override {
    draw_point_cmds_.emplace_back(BEVArtistDrawPointCommand{points_device, number_of_points});
  }

  virtual void draw_ego() override {
    Prediction ego;
    ego.position.x = 0;
    ego.position.y = 0;
    ego.position.z = 0;
    ego.size.w = 1.5f;
    ego.size.l = 3.0f;
    ego.size.h = 2.0f;
    ego.z_rotation = 0;

    auto points = transformation_predictions(transform_matrix_.data(), {ego});
    size_t num = points.size();
    for (size_t i = 0; i < num; ++i) {
      auto& item = points[i];
      auto& corners = std::get<0>(item);
      auto label = std::get<1>(item);
      auto score = std::get<2>(item);
      const int idx_of_line[][2] = {
          {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7},
      };

      float size = std::sqrt(std::pow(corners[6].x - corners[0].x, 2) + std::pow(corners[6].y - corners[0].y, 2));
      float minx = param_.image_width;
      float miny = param_.image_height;
      for (size_t ioff = 0; ioff < sizeof(idx_of_line) / sizeof(idx_of_line[0]); ++ioff) {
        auto& p0 = corners[idx_of_line[ioff][0]];
        auto& p1 = corners[idx_of_line[ioff][1]];
        minx = std::min(minx, std::min(p0.x, p1.x));
        miny = std::min(miny, std::min(p0.y, p1.y));
        cuosd_draw_line(cuosd_, p0.x, p0.y, p1.x, p1.y, 5, {0, 255, 0, 255});
      }
    }
  }

  virtual void draw_circle(int x, int y, int r) override {
    nvtype::Float4 row0 = transform_matrix_.data()[0];
    nvtype::Float4 row1 = transform_matrix_.data()[1];
    nvtype::Float4 row2 = transform_matrix_.data()[2];
    float image_x = x * row0.x + y * row0.y + 0. * row0.z + row0.w;
    float image_y = x * row1.x + y * row1.y + 0. * row1.z + row1.w;
    float weight = x * row2.x + y * row2.y + 0. * row2.z + row2.w;
    weight = std::max(1e-5f, std::min(1e5f, weight));
    image_x /= weight; image_y /= weight;
    float image_r_x = r * row0.x + row0.w;
    float image_r_y = r * row1.x + row1.w;
    float weight_r = r * row2.x + row2.w;
    weight_r = std::max(1e-5f, std::min(1e5f, weight_r));
    image_r_x /= weight_r; image_r_y /= weight_r;
    float image_r = std::sqrt(std::pow(image_r_x-image_x, 2) + std::pow(image_r_y-image_y, 2));
    cuosd_draw_circle(cuosd_, image_x, image_y, image_r, 0.5, {255, 255, 255, 255}, {0, 0, 0, 0});
    cuosd_draw_text(cuosd_, std::to_string(r).c_str(), 10, UseFont, image_r_x, image_r_y, {255, 255, 255, 255}, {0, 0, 0, 0});
  }

  virtual void draw_planning_traj(const std::vector<std::pair<float, float>>& planning_traj, std::string& command) override {
    std::vector<std::pair<float, float>> image_points;
    nvtype::Float4 row0 = transform_matrix_.data()[0];
    nvtype::Float4 row1 = transform_matrix_.data()[1];
    nvtype::Float4 row2 = transform_matrix_.data()[2];
    float init_image_x = row0.w, init_image_y = row1.w, init_weight = row2.w;
    init_weight = std::max(1e-5f, std::min(1e5f, init_weight));
    image_points.push_back({init_image_x / init_weight, init_image_y / init_weight});
    for (size_t i=0; i<planning_traj.size(); ++i) {
      float image_x = planning_traj[i].first * row0.x + planning_traj[i].second * row0.y + 0. * row0.z + row0.w;
      float image_y = planning_traj[i].first * row1.x + planning_traj[i].second * row1.y + 0. * row1.z + row1.w;
      float weight = planning_traj[i].first * row2.x + planning_traj[i].second * row2.y + 0. * row2.z + row2.w;
      weight = std::max(1e-5f, std::min(1e5f, weight));
      image_points.push_back({image_x / weight, image_y / weight});
    }
    for (size_t i=0; i<image_points.size()-1; ++i) {
      cuosd_draw_line(cuosd_, image_points[i].first, image_points[i].second, image_points[i+1].first, image_points[i+1].second, 5, {0, 255, 0, 255});
    }
    cuosd_draw_text(cuosd_, command.c_str(), 25, UseFont, 8*param_.image_width/11, 10*param_.image_height/11, {127, 255, 127, 255}, {0, 0, 0, 0});
  }

  virtual void draw_prediction(const std::vector<std::vector<float>>& predictions, bool take_title) {
    for (size_t ilabel=0; ilabel<param_.classes.size(); ++ilabel) {
      cuosd_draw_text(cuosd_, param_.classes[ilabel].name.c_str(), 
                      10, UseFont, 
                      0, (19-0.25*ilabel)*param_.image_height/20, 
                      {param_.classes[ilabel].r, param_.classes[ilabel].g, param_.classes[ilabel].b, 255}, 
                      {0, 0, 0, 0});
    }
    if (predictions.size() == 0) return;
    std::vector<Prediction> bbox_container;
    for (size_t i=0; i < predictions.size(); ++i) {
      Prediction obj;
      obj.position.x = predictions[i][0];
      obj.position.y = predictions[i][1];
      obj.position.z = predictions[i][2];
      obj.size.w = predictions[i][3];
      obj.size.l = predictions[i][4];
      obj.size.h = predictions[i][5];
      obj.z_rotation = predictions[i][6];
      obj.velocity.vx = predictions[i][7];
      obj.velocity.vy = predictions[i][8];
      obj.id = predictions[i][9]; // label
      obj.score = predictions[i][10];
      bbox_container.emplace_back(obj);
    }
    draw_prediction(bbox_container, take_title);
  }

  virtual void draw_prediction(const std::vector<Prediction>& predictions, bool take_title) override {
    auto points = transformation_predictions(transform_matrix_.data(), predictions);
    size_t num = points.size();
    for (size_t i = 0; i < num; ++i) {
      auto& item = points[i];
      auto& corners = std::get<0>(item);
      auto label = std::get<1>(item);
      auto score = std::get<2>(item);
      auto velocity = std::get<4>(item);
      const int idx_of_line[][2] = {
          {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7},
      };

      NameAndColor* name_color = &default_name_color_;
      if (label >= 0 && label < static_cast<int>(param_.classes.size())) {
        name_color = &param_.classes[label];
      }

      float size = std::sqrt(std::pow(corners[6].x - corners[0].x, 2) + std::pow(corners[6].y - corners[0].y, 2));
      size = std::max(size * 0.02f, 5.0f);
      float minx = param_.image_width;
      float miny = param_.image_height;
      for (size_t ioff = 0; ioff < sizeof(idx_of_line) / sizeof(idx_of_line[0]); ++ioff) {
        bool heading_lines = (ioff==0 || ioff==4);
        auto& p0 = corners[idx_of_line[ioff][0]];
        auto& p1 = corners[idx_of_line[ioff][1]];
        minx = std::min(minx, std::min(p0.x, p1.x));
        miny = std::min(miny, std::min(p0.y, p1.y));
        cuosd_draw_line(cuosd_, p0.x, p0.y, p1.x, p1.y, size, 
                        {(!heading_lines)?name_color->r:static_cast<unsigned char>(255), (!heading_lines)?name_color->g:static_cast<unsigned char>(255), (!heading_lines)?name_color->b:static_cast<unsigned char>(255), 255});
      }
      cuosd_draw_line(cuosd_, velocity[0], velocity[1], velocity[2], velocity[3], size, {name_color->r, name_color->g, name_color->b, 255});

      if (take_title) {
        auto title = nv::format("%.2f", score);
        cuosd_draw_text(cuosd_, title.c_str(), size, UseFont, minx, miny, {255, 255, 255, 255}, {0, 0, 0, 0});
      }
    }
  }

  virtual void apply(unsigned char* image_rgb_device, void* stream) override {
    for (size_t i = 0; i < draw_point_cmds_.size(); ++i) {
      auto& item = draw_point_cmds_[i];
      cuda_linear_launch(draw_point_to, static_cast<cudaStream_t>(stream), item.number_of_points,
                         reinterpret_cast<const half5*>(item.points_device), transform_matrix_device_, image_rgb_device,
                         param_.image_width, param_.image_stride, param_.image_height);
    }
    draw_point_cmds_.clear();

    cuosd_apply(cuosd_, image_rgb_device, nullptr, param_.image_width, param_.image_stride, param_.image_height,
                cuOSDImageFormat::RGB, stream);
  }

 private:
  std::vector<BEVArtistDrawPointCommand> draw_point_cmds_;
  std::vector<nvtype::Float4> transform_matrix_;
  float4* transform_matrix_device_ = nullptr;
  cuOSDContext_t cuosd_ = nullptr;
  BEVArtistParameter param_;
  NameAndColor default_name_color_{"Unknow", 0, 0, 0};
};

std::shared_ptr<BEVArtist> create_bev_artist(const BEVArtistParameter& param) {
  std::shared_ptr<BEVArtistImplement> instance(new BEVArtistImplement());
  if (!instance->init(param)) {
    printf("Failed to create BEVArtist\n");
    instance.reset();
  }
  return instance;
}

static __device__ uchar3 load_pixel(const unsigned char* image, int x, int y, float sx, float sy, int width, int stride,
                                    int height) {
  uchar3 rgb[4];
  float src_x = (x + 0.5f) * sx - 0.5f;
  float src_y = (y + 0.5f) * sy - 0.5f;
  int y_low = floorf(src_y);
  int x_low = floorf(src_x);
  int y_high = limit(y_low + 1, 0, height - 1);
  int x_high = limit(x_low + 1, 0, width - 1);
  y_low = limit(y_low, 0, height - 1);
  x_low = limit(x_low, 0, width - 1);

  int ly = rint((src_y - y_low) * INTER_RESIZE_COEF_SCALE);
  int lx = rint((src_x - x_low) * INTER_RESIZE_COEF_SCALE);
  int hy = INTER_RESIZE_COEF_SCALE - ly;
  int hx = INTER_RESIZE_COEF_SCALE - lx;

  rgb[0] = *(uchar3*)&image[y_low * stride + x_low * 3];
  rgb[1] = *(uchar3*)&image[y_low * stride + x_high * 3];
  rgb[2] = *(uchar3*)&image[y_high * stride + x_low * 3];
  rgb[3] = *(uchar3*)&image[y_high * stride + x_high * 3];

  uchar3 output;
  output.x =
      (((hy * ((hx * rgb[0].x + lx * rgb[1].x) >> 4)) >> 16) + ((ly * ((hx * rgb[2].x + lx * rgb[3].x) >> 4)) >> 16) + 2) >> 2;
  output.y =
      (((hy * ((hx * rgb[0].y + lx * rgb[1].y) >> 4)) >> 16) + ((ly * ((hx * rgb[2].y + lx * rgb[3].y) >> 4)) >> 16) + 2) >> 2;
  output.z =
      (((hy * ((hx * rgb[0].z + lx * rgb[1].z) >> 4)) >> 16) + ((ly * ((hx * rgb[2].z + lx * rgb[3].z) >> 4)) >> 16) + 2) >> 2;
  return output;
}

static __global__ void resize_to_kernel(int nx, int ny, int nz, int x0, int y0, float sx, float sy, const unsigned char* img,
                                        int image_width, int image_stride, int image_height, float alpha, unsigned char* output,
                                        int output_stride) {
  int ox = cuda_2d_x;
  int oy = cuda_2d_y;
  if (ox >= nx || oy >= ny) return;

  uchar3 pixel = load_pixel(img, ox, oy, sx, sy, image_width, image_stride, image_height);
  auto& old = *(uchar3*)(output + output_stride * (oy + y0) + (ox + x0) * 3);
  old = make_uchar3(limit(pixel.x * alpha + old.x * (1.0f - alpha), 0.0f, 255.0f),
                    limit(pixel.y * alpha + old.y * (1.0f - alpha), 0.0f, 255.0f),
                    limit(pixel.z * alpha + old.z * (1.0f - alpha), 0.0f, 255.0f));
}

static __global__ void flipx_kernel(int nx, int ny, int nz, const unsigned char* img, int image_stride, unsigned char* output,
                                    int output_stride) {
  int ox = cuda_2d_x;
  int oy = cuda_2d_y;
  if (ox >= nx || oy >= ny) return;

  *(uchar3*)&output[oy * output_stride + ox * 3] = *(uchar3*)&img[oy * image_stride + (nx - ox - 1) * 3];
}

class SceneArtistImplement : public SceneArtist {
 public:
  virtual ~SceneArtistImplement() {
    if (cuosd_) cuosd_context_destroy(cuosd_);
  }

  bool init(const SceneArtistParameter& param) {
    this->param_ = param;
    cuosd_ = cuosd_context_create();
    return cuosd_ != nullptr;
  }

  virtual void flipx(const unsigned char* image_device, int image_width, int image_stride, int image_height,
                     unsigned char* output_device, int output_stride, void* stream) override {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    cuda_2d_launch(flipx_kernel, _stream, image_width, image_height, 1, image_device, image_stride, output_device, output_stride);
  }

  virtual void resize_to(const unsigned char* image, int x0, int y0, int x1, int y1, int image_width, int image_stride,
                         int image_height, float alpha, void* stream) override {
    x0 = limit(x0, 0, param_.width - 1);
    y0 = limit(y0, 0, param_.height - 1);
    x1 = limit(x1, 1, param_.width);
    y1 = limit(y1, 1, param_.height);
    int ow = x1 - x0;
    int oh = y1 - y0;
    if (ow <= 0 || oh <= 0) return;

    float sx = image_width / (float)ow;
    float sy = image_height / (float)oh;
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    cuda_2d_launch(resize_to_kernel, _stream, ow, oh, 1, x0, y0, sx, sy, image, image_width, image_stride, image_height, alpha,
                   param_.image_device, param_.stride);
  }

 private:
  SceneArtistParameter param_;
  cuOSDContext_t cuosd_ = nullptr;
};

std::shared_ptr<SceneArtist> create_scene_artist(const SceneArtistParameter& param) {
  std::shared_ptr<SceneArtistImplement> instance(new SceneArtistImplement());
  if (!instance->init(param)) {
    printf("Failed to create SceneArtist\n");
    instance.reset();
  }
  return instance;
}

};  // namespace nv