#include "run_store.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <nav2_msgs/srv/get_costmap.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

namespace grid_plan_tester
{
namespace
{
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;
using PlanClient = rclcpp_action::Client<PlanAction>;
using GetMap = nav2_msgs::srv::GetCostmap;
using Executor = rclcpp::executors::SingleThreadedExecutor;
constexpr Clock::duration poll_interval = 100ms;
constexpr auto cancel_wait = 2s;
volatile std::sig_atomic_t stop_signal = 0;

void requestStop(int sig) {stop_signal = sig;}

bool stopping() {return stop_signal != 0 || !rclcpp::ok();}

class TrackedPlanClient final : public PlanClient
{
public:
  using PlanClient::PlanClient;

  void prepare(RunStore & store, const std::string & target_id)
  {
    uuid_ = store.outstanding() ? store.outstanding()->uuid : PlanClient::generate_goal_id();
    store.markOutstanding(target_id, uuid_);
  }

private:
  // Jazzy's async_send_goal obtains the on-wire UUID from this hook, before sending the request.
  rclcpp_action::GoalUUID generate_goal_id() override {return uuid_;}
  rclcpp_action::GoalUUID uuid_{};
};

struct Options
{
  std::filesystem::path dataset_dir;
  std::string run_name, planner{"GridBased"}, action{"/compute_path_to_pose"};
  std::string costmap_service{"/global_costmap/get_costmap"};
  double timeout_s{10.0};
  size_t limit{std::numeric_limits<size_t>::max()};
};

constexpr char usage[] =
  "Usage: ros2 run grid_plan_tester run DATASET RUN_NAME [--planner ID] [--timeout SECONDS]\n"
  "       [--limit N] [--action NAME] [--costmap-service NAME]\n"
  "Defaults: GridBased, 10 seconds, /compute_path_to_pose, /global_costmap/get_costmap.\n"
  "The limit counts attempts in this invocation. Reuse RUN_NAME to retry incomplete goals.\n";

Options parseOptions(const std::vector<std::string> & args)
{
  if (args.size() < 3 || !validRunName(args[2])) {
    throw std::invalid_argument(usage);
  }
  Options o;
  o.dataset_dir = args[1];
  o.run_name = args[2];
  for (size_t i = 3; i < args.size(); i += 2) {
    const auto & key = args[i];
    if (i + 1 == args.size() || args[i + 1].empty()) {
      throw std::invalid_argument("Missing value for " + key);
    }
    const auto & value = args[i + 1];
    if (key == "--planner") {
      o.planner = value;
    } else if (key == "--action") {
      o.action = value;
    } else if (key == "--costmap-service") {
      o.costmap_service = value;
    } else if (key == "--timeout") {
      size_t consumed = 0;
      o.timeout_s = std::stod(value, &consumed);
      const auto maximum = std::chrono::duration<double>(Clock::duration::max() / 2).count();
      if (consumed != value.size() || !std::isfinite(o.timeout_s) ||
        o.timeout_s <= 0.0 || o.timeout_s >= maximum)
      {
        throw std::invalid_argument("Expected a positive, finite --timeout in seconds");
      }
    } else if (key == "--limit") {
      const auto end = value.data() + value.size();
      const auto result = std::from_chars(value.data(), end, o.limit);
      if (result.ec != std::errc{} || result.ptr != end) {
        throw std::invalid_argument("Expected a nonnegative integer for --limit");
      }
    } else {
      throw std::invalid_argument("Unknown option " + key);
    }
  }
  return o;
}

Clock::duration waitSlice(Clock::time_point deadline)
{
  return std::clamp(deadline - Clock::now(), Clock::duration::zero(), poll_interval);
}

template<typename Wait>
bool waitForServer(Wait wait, Clock::time_point deadline, bool finish_cancel = false)
{
  while (rclcpp::ok() && (finish_cancel || !stop_signal)) {
    const auto slice = waitSlice(deadline);
    if (wait(slice)) {return true;}
    if (slice == Clock::duration::zero()) {break;}
  }
  return false;
}

template<typename Future>
bool waitForReply(Executor & executor, const Future & future, Clock::time_point deadline,
  bool finish_cancel = false)
{
  while (rclcpp::ok() && (finish_cancel || !stop_signal)) {
    const auto slice = waitSlice(deadline);
    const auto result = executor.spin_until_future_complete(future, slice);
    if (result == rclcpp::FutureReturnCode::SUCCESS) {
      return (finish_cancel || !stopping()) && Clock::now() <= deadline;
    }
    if (slice == Clock::duration::zero() || result == rclcpp::FutureReturnCode::INTERRUPTED) {
      break;
    }
  }
  return false;
}

std::optional<std::string> observeMap(
  Executor & executor, rclcpp::Client<GetMap> & client, const std::string & frozen_id)
{
  if (stopping()) {return std::nullopt;}
  try {
    const auto deadline = Clock::now() + 2s;
    if (waitForServer([&](auto slice) {return client.wait_for_service(slice);}, deadline)) {
      auto request = client.async_send_request(std::make_shared<GetMap::Request>());
      if (waitForReply(executor, request.future, deadline)) {
        const auto id = costmapId(request.get()->map);
        if (id != frozen_id) {
          std::cerr << "Live costmap " << id << " differs from frozen costmap " << frozen_id << '\n';
        }
        return id;
      }
      client.remove_pending_request(request);
    }
    std::cerr << "Live costmap signature unavailable from " << client.get_service_name() << '\n';
  } catch (const std::exception & e) {
    std::cerr << "Live costmap signature unavailable: " << e.what() << '\n';
  }
  return std::nullopt;
}

struct Attempt
{
  PlanRecord row;
  bool can_advance;
  Clock::time_point finished{Clock::now()};
};

Attempt incomplete(const std::string & action_status, const std::string & reason, bool can_advance)
{
  PlanRecord row;
  row.action_status = action_status;
  row.error_msg = reason;
  return {row, can_advance};
}

Attempt outcome(const Dataset & dataset, const Target & target, const PlanHandle::WrappedResult & result)
{
  const auto received = Clock::now();
  auto row = classifyResult(dataset.geometry, target, result);
  const bool terminal = row.action_status != "UNKNOWN";
  return {std::move(row), terminal, received};
}

bool reconcileOutstanding(
  Executor & executor, rclcpp::Node & node, const std::string & action, RunStore & store)
{
  const auto previous = store.outstanding();
  if (!previous) {return true;}
  if (store.completed(previous->id)) {  // Crashed between persisting a terminal row and unlinking.
    store.clearOutstanding();
    return true;
  }
  using Lookup = PlanAction::Impl::GetResultService;
  using Cancel = PlanAction::Impl::CancelGoalService;
  using Status = rclcpp_action::GoalStatus;
  auto lookup = node.create_client<Lookup>(action + "/_action/get_result");
  std::optional<rclcpp::Client<Lookup>::FutureAndRequestId> query;
  auto get_result = [&](Clock::time_point deadline, bool canceling) -> Lookup::Response::SharedPtr {
    if (!query) {
      if (!waitForServer([&](auto slice) {return lookup->wait_for_service(slice);}, deadline, canceling)) {
        return {};
      }
      auto request = std::make_shared<Lookup::Request>();
      request->goal_id.uuid = previous->uuid;
      query.emplace(lookup->async_send_request(request));
    }
    if (!waitForReply(executor, query->future, deadline, canceling)) {return {};}
    auto response = query->get();
    query.reset();
    return response;
  };
  auto settle = [&](const Lookup::Response::SharedPtr & response) {
    if (!response) {return false;}
    if (response->status == Status::STATUS_UNKNOWN) {
      // A delayed SendGoal from the earlier process may still arrive. Replay only this UUID.
      std::cerr << "Server has no result for " << previous->id << ", retrying its original UUID\n";
      return true;
    }
    if (response->status != Status::STATUS_SUCCEEDED && response->status != Status::STATUS_ABORTED &&
      response->status != Status::STATUS_CANCELED)
    {
      return false;
    }
    store.recoverTerminal({previous->uuid, static_cast<rclcpp_action::ResultCode>(response->status),
      std::make_shared<PlanAction::Result>(std::move(response->result))});
    return true;
  };
  if (settle(get_result(Clock::now() + cancel_wait, false))) {return true;}

  auto cancel = node.create_client<Cancel>(action + "/_action/cancel_goal");
  const auto deadline = Clock::now() + cancel_wait;
  std::optional<rclcpp::Client<Cancel>::FutureAndRequestId> cancel_reply;
  if (waitForServer([&](auto slice) {return cancel->wait_for_service(slice);}, deadline, true)) {
    auto request = std::make_shared<Cancel::Request>();
    request->goal_info.goal_id.uuid = previous->uuid;  // Nonzero UUID, zero timestamp: only this goal.
    store.noteCutoff("Previous runner exited with this goal active. Recovery requested cancellation");
    cancel_reply.emplace(cancel->async_send_request(request));
  }
  auto result = get_result(deadline, true);
  if (query) {lookup->remove_pending_request(*query);}
  // An outstanding cancel for an expired UUID could cancel its replay. Drain the cancel in that case.
  bool may_replay = true;
  if (cancel_reply) {
    if (result && result->status == Status::STATUS_UNKNOWN) {
      may_replay = cancel_reply->future.wait_for(0s) == std::future_status::ready ||
        waitForReply(executor, cancel_reply->future, deadline, true);
    }
    cancel->remove_pending_request(*cancel_reply);
  }
  if (may_replay && settle(result)) {return true;}
  std::cerr << "Previous goal " << previous->id << " (" << rclcpp_action::to_string(previous->uuid)
            << ") has no confirmed terminal result. No new goal sent.\n";
  return false;
}

Attempt cancelAndDrain(
  Executor & executor, PlanClient & client, const PlanHandle::SharedPtr & handle,
  const std::shared_future<PlanHandle::WrappedResult> & result, const Dataset & dataset,
  const Target & target, std::string reason)
{
  const auto deadline = Clock::now() + cancel_wait;
  std::shared_future<PlanClient::CancelResponse::SharedPtr> canceled;
  try {
    canceled = client.async_cancel_goal(handle);  // Cancel only this request.
  } catch (const std::exception & e) {
    reason += ". Cancel request failed: " + std::string(e.what());
  }
  try {
    if (result.valid() && waitForReply(executor, result, deadline, true)) {
      auto attempt = outcome(dataset, target, result.get());
      attempt.row.status = "incomplete";
      if (!attempt.row.error_msg.empty()) {reason += ". " + attempt.row.error_msg;}
      attempt.row.error_msg = reason;
      return attempt;
    }
    if (canceled.valid() && waitForReply(executor, canceled, deadline, true)) {
      reason += ". Cancel reply code " + std::to_string(canceled.get()->return_code);
    }
  } catch (const std::exception & e) {
    reason += ". " + std::string(e.what());
  }
  return incomplete("UNKNOWN", reason + ". Terminal action result unconfirmed, stopping batch", false);
}

Attempt requestPlan(
  Executor & executor, TrackedPlanClient & client, const Options & options,
  const Dataset & dataset, const Target & target, RunStore & store, Clock::time_point & wall_start)
{
  const auto timeout = std::chrono::ceil<Clock::duration>(std::chrono::duration<double>(options.timeout_s));
  PlanHandle::SharedPtr handle;
  std::shared_future<PlanHandle::WrappedResult> result;
  try {
    if (!waitForServer([&](auto slice) {return client.wait_for_action_server(slice);}, wall_start + timeout)) {
      return incomplete("UNAVAILABLE", "Action server unavailable or discovery interrupted", false);
    }
    PlanAction::Goal goal;
    goal.use_start = true;
    goal.planner_id = options.planner;
    goal.start.header.frame_id = dataset.geometry.frame;
    // Zero requests the latest transform if the planner needs one in its own clock domain.
    goal.start.pose = dataset.start;
    goal.goal.header = goal.start.header;
    goal.goal.pose.position.x = target.x;
    goal.goal.pose.position.y = target.y;
    goal.goal.pose.orientation.z = std::sin(target.yaw / 2.0);
    goal.goal.pose.orientation.w = std::cos(target.yaw / 2.0);
    const bool replay = store.outstanding().has_value();
    client.prepare(store, target.id);
    if (stopping()) {return incomplete("UNSENT", "Interrupted before goal submission", !replay);}
    wall_start = Clock::now();
    const auto deadline = wall_start + timeout;
    auto accepted = client.async_send_goal(goal);
    std::string cutoff;
    if (!waitForReply(executor, accepted, deadline)) {
      cutoff = stopping() ? "Interrupted while awaiting acceptance" : "Client acceptance deadline exceeded";
      store.noteCutoff(cutoff);
      if (!waitForReply(executor, accepted, Clock::now() + cancel_wait, true)) {
        return incomplete("UNKNOWN", cutoff + ". Goal acceptance unconfirmed, stopping batch", false);
      }
    }
    handle = accepted.get();
    if (!handle) {
      if (replay) {
        return incomplete("REJECTED", "Original UUID rejected on replay. Its first request may still be active", false);
      }
      return incomplete("REJECTED", cutoff.empty() ? "Action server rejected goal" : cutoff, true);
    }
    result = client.async_get_result(handle);
    if (cutoff.empty()) {
      if (waitForReply(executor, result, deadline)) {
        return outcome(dataset, target, result.get());
      }
      cutoff = stopping() ? "Client interrupted" : "Client planning deadline exceeded";
    }
    store.noteCutoff(cutoff);
    return cancelAndDrain(executor, client, handle, result, dataset, target, cutoff);
  } catch (const std::exception & e) {
    if (handle && rclcpp::ok()) {
      store.noteCutoff(e.what());
      return cancelAndDrain(executor, client, handle, result, dataset, target, e.what());
    }
    return incomplete("UNKNOWN", std::string("Action transport error: ") + e.what(), false);
  }
}

int runPlans(const Options & options, const Dataset & dataset)
{
  auto node = std::make_shared<rclcpp::Node>("grid_plan_runner");
  const auto action = node->get_node_topics_interface()->resolve_topic_name(options.action);
  auto client = std::make_shared<TrackedPlanClient>(node->get_node_base_interface(),
    node->get_node_graph_interface(), node->get_node_logging_interface(), options.action);
  node->get_node_waitables_interface()->add_waitable(client, nullptr);
  auto map_client = node->create_client<GetMap>(options.costmap_service);
  Executor executor;
  executor.add_node(node);
  RunStore store(options.dataset_dir, options.run_name, dataset, options.planner, action,
    map_client->get_service_name());
  if (options.limit > 0 && !reconcileOutstanding(executor, *node, action, store)) {
    return stop_signal ? 128 + stop_signal : 2;
  }
  const auto targets = store.pendingTargets();
  const size_t pending = targets.size();
  const size_t selected = std::min(pending, options.limit);
  if (selected == 0) {
    std::cout << "No targets selected (" << pending << " pending).\n";
    return 0;
  }

  store.beginObservation(options.timeout_s, observeMap(executor, *map_client, dataset.map_id));
  size_t attempted = 0, incomplete_count = 0;
  bool stopped = false;
  for (const auto * target : targets) {
    if (stopping() || attempted == selected) {break;}
    auto start = Clock::now();
    auto attempt = requestPlan(executor, *client, options, dataset, *target, store, start);
    attempt.row.wall_ms = std::chrono::duration<double, std::milli>(attempt.finished - start).count();
    store.record(target->id, attempt.row);
    if (attempt.can_advance) {store.clearOutstanding();}
    ++attempted;
    if (attempt.row.status == "incomplete") {
      ++incomplete_count;
      std::cerr << target->id << " incomplete: " << attempt.row.error_msg << '\n';
    }
    if (attempted % 50 == 0 || attempted == selected) {
      std::cout << attempted << '/' << selected << " attempted\n" << std::flush;
    }
    if (!attempt.can_advance) {stopped = true; break;}
  }
  const size_t remaining = pending - (attempted - incomplete_count);
  const std::string state = stopping() ? "interrupted" : stopped ? "stopped" :
    remaining == 0 ? "complete" : "partial";
  store.finishObservation(state, observeMap(executor, *map_client, dataset.map_id));
  std::cout << "Run " << options.run_name << ": " << attempted << " attempted, " << remaining
            << " pending. Results in " << (options.dataset_dir / "runs" / options.run_name) << '\n';
  if (stop_signal) {return 128 + stop_signal;}
  return stopped || incomplete_count != 0 ? 2 : 0;
}
}  // namespace
}  // namespace grid_plan_tester

int main(int argc, char ** argv)
{
  using namespace grid_plan_tester;
  try {
    const auto args = rclcpp::remove_ros_arguments(argc, argv);
    if (std::find(args.begin() + 1, args.end(), "--help") != args.end()) {
      std::cout << usage;
      return 0;
    }
    const auto options = parseOptions(args);
    const auto dataset = loadDataset(options.dataset_dir);
    // Keep the ROS context alive on Ctrl-C until this client's active goal has been canceled.
    std::signal(SIGINT, requestStop);
    std::signal(SIGTERM, requestStop);
    rclcpp::init(argc, argv, rclcpp::InitOptions{}, rclcpp::SignalHandlerOptions::None);
    const int code = runPlans(options, dataset);
    rclcpp::shutdown();
    return code;
  } catch (const std::exception & e) {
    if (rclcpp::ok()) {rclcpp::shutdown();}
    std::cerr << "grid_plan_tester run: " << e.what() << '\n';
    return 1;
  }
}
