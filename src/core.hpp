#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <nav2_msgs/msg/costmap.hpp>
#include <opencv2/core.hpp>

namespace grid_plan_tester
{
struct MapGeometry
{
  std::string frame;
  int width{}, height{};
  double resolution{}, origin_x{}, origin_y{}, origin_yaw{};
  cv::Point2d center(int mx, int my) const;
  cv::Point cell(double x, double y) const;
};

struct Target
{
  std::string id, kind;
  int group{}, mx{}, my{}, cost{};
  double x{}, y{}, yaw{};
};

struct SamplingOptions
{
  double spacing{3.0}, merge_radius{0.01};
  int max_free_per_component{0}, min_cost{1}, max_cost{200};
  double goal_yaw{0.0};
};

struct Dataset
{
  MapGeometry geometry;
  geometry_msgs::msg::Pose start;
  std::vector<Target> targets;
  std::string id, map_id;
};

std::vector<Target> sampleTargets(
  const cv::Mat & costmap, const MapGeometry & geometry, const SamplingOptions & options);
MapGeometry geometryOf(const nav2_msgs::msg::Costmap & map);
std::string costmapId(const nav2_msgs::msg::Costmap & map);
void saveDataset(
  const std::filesystem::path & destination, const nav2_msgs::msg::Costmap & map,
  const geometry_msgs::msg::Pose & start, const builtin_interfaces::msg::Time & start_stamp,
  const SamplingOptions & options, const std::vector<Target> & targets);
Dataset loadDataset(const std::filesystem::path & directory);
bool validRunName(const std::string & name);
std::string csvField(std::string value);
std::vector<std::string> parseCsv(const std::string & line);
}  // namespace grid_plan_tester
