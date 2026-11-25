#include "core.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <thread>

#include <nav2_msgs/srv/get_costmap.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2/time.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>

namespace gt = grid_plan_tester;
using namespace std::chrono_literals;

namespace
{
constexpr char kUsage[] =
  "Usage: sample DATASET [--mode quick|regular|dense] [--spacing METERS]\n"
  "       [--merge-radius METERS] [--min-cost 1] [--max-cost 200] [--goal-yaw RADIANS]\n"
  "       [--inflation-cost-step N] (1..252, default 252; one goal per obstacle group per occupied cost band)\n"
  "       [--base-frame base_footprint] [--costmap-service /global_costmap/get_costmap]\n"
  "       [--timeout SECONDS] [--ros-args ...]\n"
  "Regular uses 0.5 m base tiles by default, merging fully free 2x2 and 4x4 tile blocks.\n"
  "Quick uses 3 m tiles capped at 5 per free component; dense uses fixed 1 m tiles.\n"
  "Writes DATASET (all goals) and DATASET_inflation (inflation goals only). Both paths must be new.\n"
  "Goal yaw defaults to zero in the costmap frame.\n";

struct Arguments
{
  std::filesystem::path directory;
  gt::SamplingOptions options;
  std::string base_frame{"base_footprint"}, service{"/global_costmap/get_costmap"};
  double timeout{10.0};
};

template<typename T>
T number(const std::string & value)
{
  T out{};
  const auto end = value.data() + value.size();
  const auto parsed = std::from_chars(value.data(), end, out);
  if (parsed.ec != std::errc{} || parsed.ptr != end || !std::isfinite(out)) {
    throw std::invalid_argument("Invalid number: " + value);
  }
  return out;
}

Arguments parse(const std::vector<std::string> & args)
{
  if (args.size() < 2 || args[1].empty() || args[1][0] == '-') {
    throw std::invalid_argument(kUsage);
  }
  Arguments a;
  a.directory = std::filesystem::path(args[1]).lexically_normal();
  if (!a.directory.has_filename() && a.directory.has_relative_path()) {
    a.directory = a.directory.parent_path();
  }
  std::string mode = "regular";
  std::optional<double> spacing;
  for (size_t i = 2; i < args.size(); i += 2) {
    if (i + 1 == args.size()) {
      throw std::invalid_argument("Missing value for " + args[i]);
    }
    const auto & key = args[i];
    const auto & val = args[i + 1];
    if (key == "--mode") {
      mode = val;
    } else if (key == "--spacing") {
      spacing = number<double>(val);
    } else if (key == "--merge-radius") {
      a.options.merge_radius = number<double>(val);
    } else if (key == "--min-cost") {
      a.options.min_cost = number<int>(val);
    } else if (key == "--max-cost") {
      a.options.max_cost = number<int>(val);
    } else if (key == "--inflation-cost-step") {
      a.options.inflation_cost_step = number<int>(val);
      if (a.options.inflation_cost_step < 1 || a.options.inflation_cost_step > 252) {
        throw std::invalid_argument("--inflation-cost-step must be in [1, 252]");
      }
    } else if (key == "--goal-yaw") {
      a.options.goal_yaw = number<double>(val);
    } else if (key == "--base-frame") {
      a.base_frame = val;
    } else if (key == "--costmap-service") {
      a.service = val;
    } else if (key == "--timeout") {
      a.timeout = number<double>(val);
    } else {
      throw std::invalid_argument("Unknown option: " + key);
    }
  }
  if (mode != "quick" && mode != "regular" && mode != "dense") {
    throw std::invalid_argument("--mode must be quick, regular or dense");
  }
  if (mode == "quick") {a.options.spacing = 3.0;}
  if (mode == "dense") {a.options.spacing = 1.0;}
  a.options.spacing = spacing.value_or(a.options.spacing);
  a.options.adaptive_free_tiles = mode == "regular";
  a.options.max_free_per_component = mode == "quick" ? 5 : 0;
  if (a.options.spacing <= 0 || a.options.merge_radius < 0 || a.options.min_cost < 1 ||
    a.options.max_cost > 252 || a.options.max_cost < a.options.min_cost ||
    a.base_frame.empty() || a.service.empty() || a.timeout <= 0 || a.timeout > 86400)
  {
    throw std::invalid_argument("Invalid spacing, merge radius, cost range, frame, service or timeout");
  }
  return a;
}

void sample(const Arguments & a)
{
  const std::filesystem::path inflation_dir = a.directory.string() + "_inflation";
  for (const auto & dir : {a.directory, inflation_dir}) {
    if (std::filesystem::exists(dir)) {
      throw std::runtime_error("Dataset already exists: " + dir.string());
    }
  }
  auto node = std::make_shared<rclcpp::Node>("grid_plan_sample");
  tf2_ros::Buffer buffer(node->get_clock());
  tf2_ros::TransformListener listener(buffer, node, true);
  auto client = node->create_client<nav2_msgs::srv::GetCostmap>(a.service);
  const auto wait = std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::duration<double>(a.timeout));
  if (!client->wait_for_service(wait)) {
    throw std::runtime_error("Costmap service unavailable: " + a.service);
  }
  auto request = std::make_shared<nav2_msgs::srv::GetCostmap::Request>();
  auto future = client->async_send_request(request);
  if (rclcpp::spin_until_future_complete(node, future, wait) != rclcpp::FutureReturnCode::SUCCESS) {
    throw std::runtime_error("No response from " + a.service);
  }
  auto response = future.get();
  auto & map = response->map;
  const auto g = gt::geometryOf(map);
  // A paused simulator must not prevent a wall-clock timeout.
  std::string tf_error;
  const auto deadline = std::chrono::steady_clock::now() + wait;
  while (!buffer.canTransform(g.frame, a.base_frame, tf2::TimePointZero, &tf_error)) {
    if (!rclcpp::ok() || std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error("Cannot freeze TF " + g.frame + " <- " + a.base_frame + ": " + tf_error);
    }
    std::this_thread::sleep_for(25ms);
  }
  const auto tf = buffer.lookupTransform(g.frame, a.base_frame, tf2::TimePointZero);
  geometry_msgs::msg::Pose start;
  start.position.x = tf.transform.translation.x;
  start.position.y = tf.transform.translation.y;
  start.position.z = tf.transform.translation.z;
  start.orientation = tf.transform.rotation;
  const cv::Mat pixels(g.height, g.width, CV_8UC1, map.data.data());
  auto targets = gt::sampleTargets(pixels, g, a.options);
  gt::saveDataset(a.directory, map, start, tf.header.stamp, a.options, targets);
  const auto total = targets.size();
  targets.erase(std::remove_if(targets.begin(), targets.end(),
    [](const auto & t) {return t.kind != "inflation";}), targets.end());
  gt::saveDataset(inflation_dir, map, start, tf.header.stamp, a.options, targets);
  std::cout << a.directory << ": " << g.width << "x" << g.height << ", frame=" << g.frame
            << ", free=" << total - targets.size() << ", inflation=" << targets.size()
            << ", total=" << total << '\n'
            << inflation_dir << ": inflation only, total=" << targets.size() << '\n'
            << "ros2 run grid_plan_tester run " << inflation_dir << " navfn\n";
}
}  // namespace

int main(int argc, char ** argv)
{
  int status = 0;
  try {
    const auto args = rclcpp::remove_ros_arguments(argc, argv);
    if (args.size() == 2 && (args[1] == "--help" || args[1] == "-h")) {
      std::cout << kUsage;
    } else {
      const auto options = parse(args);
      rclcpp::init(argc, argv);
      sample(options);
    }
  } catch (const std::exception & e) {
    std::cerr << "sample: " << e.what() << '\n';
    status = 1;
  }
  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
  return status;
}
