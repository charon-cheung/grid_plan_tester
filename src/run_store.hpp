#pragma once

#include "core.hpp"

#include <cstdio>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>

#include <nav2_msgs/action/compute_path_to_pose.hpp>
#include <rclcpp_action/client_goal_handle.hpp>
#include <yaml-cpp/yaml.h>

namespace grid_plan_tester
{
using PlanAction = nav2_msgs::action::ComputePathToPose;
using PlanHandle = rclcpp_action::ClientGoalHandle<PlanAction>;

struct PlanRecord
{
  std::string status{"incomplete"}, action_status{"UNKNOWN"}, error_msg;
  std::optional<uint16_t> error_code;
  std::optional<double> wall_ms;
  std::optional<double> planning_ms, length_m, end_x, end_y, end_error_m;
  size_t poses{};
};

PlanRecord classifyResult(
  const MapGeometry & geometry, const Target & target, const PlanHandle::WrappedResult & result);

struct OutstandingGoal
{
  std::string id;
  rclcpp_action::GoalUUID uuid;
  std::string cutoff_row;
};

class RunStore
{
public:
  RunStore(
    const std::filesystem::path & dataset_dir, const std::string & name, const Dataset & dataset,
    const std::string & planner, const std::string & action, const std::string & costmap_service);
  bool completed(const std::string & id) const;
  bool recorded(const std::string & id) const;
  std::vector<const Target *> pendingTargets() const;
  const std::optional<OutstandingGoal> & outstanding() const {return outstanding_;}
  void markOutstanding(const std::string & id, const rclcpp_action::GoalUUID & uuid);
  void noteCutoff(const std::string & reason);
  void recoverTerminal(const PlanHandle::WrappedResult & result);
  void clearOutstanding();
  void record(std::string id, const PlanRecord & row);
  void beginObservation(double timeout_s, const std::optional<std::string> & map_id);
  void finishObservation(const std::string & state, const std::optional<std::string> & map_id);

private:
  struct SavedRow
  {
    bool completed{};
    std::string line;
  };
  void readResults();
  void saveMetadata();
  void saveOutstanding(OutstandingGoal goal);
  void saveRow(const std::string & id, SavedRow replacement);

  const Dataset & dataset_;
  std::filesystem::path dir_;
  std::unique_ptr<std::FILE, int (*)(std::FILE *)> lock_{nullptr, &std::fclose};
  YAML::Node metadata_;
  std::unordered_map<std::string, SavedRow> rows_;
  std::string last_attempt_;
  std::optional<OutstandingGoal> outstanding_;
};
}  // namespace grid_plan_tester
