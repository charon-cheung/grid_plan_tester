#include "run_store.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>

#include <gtest/gtest.h>

namespace gt = grid_plan_tester;
namespace fs = std::filesystem;
using Code = rclcpp_action::ResultCode;

namespace
{
class RunTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    std::string dir = (fs::temp_directory_path() / "grid-plan-run-test-XXXXXX").string();
    ASSERT_NE(::mkdtemp(dir.data()), nullptr);
    root = dir;
    data.id = "frozen-dataset";
    data.map_id = "frozen-map";
    data.geometry = {"map", 5, 4, 0.2, -1.5, 2.0, 0.71};
    const auto xy = data.geometry.center(2, 1);
    data.targets.push_back({"target-a", "free", 1, 2, 1, 0, xy.x, xy.y, -0.3});
    const auto next = data.geometry.center(3, 1);
    data.targets.push_back({"target-b", "free", 1, 3, 1, 0, next.x, next.y, -0.3});
  }
  void TearDown() override {fs::remove_all(root);}

  gt::PlanHandle::WrappedResult success() const
  {
    gt::PlanHandle::WrappedResult reply{};
    reply.code = Code::SUCCEEDED;
    reply.result = std::make_shared<gt::PlanAction::Result>();
    reply.result->planning_time.nanosec = 5200000;
    reply.result->path.header.frame_id = data.geometry.frame;
    for (int mx : {0, 1, 2}) {
      const auto p = data.geometry.center(mx, 1);
      geometry_msgs::msg::PoseStamped pose;
      pose.pose.position.x = p.x;
      pose.pose.position.y = p.y;
      reply.result->path.poses.push_back(pose);
    }
    return reply;
  }

  fs::path root;
  gt::Dataset data;
};

TEST_F(RunTest, ClassifiesEndpointByRequestedRotatedCellAndMeasuresTheReturnedPath)
{
  auto reply = success();
  const auto & target = data.targets.front();
  auto & end = reply.result->path.poses.back().pose.position;
  end.x += 0.001;
  auto row = gt::classifyResult(data.geometry, target, reply);
  EXPECT_EQ(row.status, "hit");
  EXPECT_EQ(row.action_status, "SUCCEEDED");
  ASSERT_TRUE(row.end_error_m && row.planning_ms);
  EXPECT_NEAR(*row.end_error_m, 0.001, 1e-12);
  EXPECT_NEAR(*row.planning_ms, 5.2, 1e-12);
  end.x = data.targets[1].x;
  end.y = data.targets[1].y;
  row = gt::classifyResult(data.geometry, target, reply);
  EXPECT_EQ(row.status, "adjusted");
  EXPECT_EQ(row.poses, 3u);
  ASSERT_TRUE(row.length_m && row.end_error_m && row.end_x && row.end_y);
  EXPECT_NEAR(*row.length_m, 0.6, 1e-12);
  EXPECT_NEAR(*row.end_error_m, 0.2, 1e-12);
  EXPECT_DOUBLE_EQ(*row.end_x, end.x);
  EXPECT_DOUBLE_EQ(*row.end_y, end.y);
}

TEST_F(RunTest, SeparatesInvalidSuccessAndPlannerTimeoutFromCanceledOrUnknown)
{
  auto reply = success();
  const auto classify = [&]() {return gt::classifyResult(data.geometry, data.targets[0], reply);};
  reply.result->path.poses.back().header.frame_id = "odom";
  EXPECT_EQ(classify().status, "failed");
  reply.result->path.poses.back().header.frame_id.clear();
  reply.result->path.header.frame_id = "odom";
  EXPECT_EQ(classify().status, "failed");
  reply.result->path.header.frame_id = "map";
  reply.result->path.poses[0].pose.position.x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(classify().status, "failed");
  reply.result->path.poses.clear();
  EXPECT_EQ(classify().status, "failed");
  reply.code = Code::ABORTED;
  reply.result->error_code = gt::PlanAction::Result::TIMEOUT;
  reply.result->error_msg = "Search timed out";
  const auto timeout = classify();
  EXPECT_EQ(timeout.status, "failed");
  EXPECT_EQ(timeout.error_code, 207);
  EXPECT_EQ(timeout.error_msg, "Search timed out");
  reply.code = Code::CANCELED;
  EXPECT_EQ(classify().status, "incomplete");
  reply.code = Code::UNKNOWN;
  EXPECT_EQ(classify().status, "incomplete");
}

TEST_F(RunTest, ResumesIncompleteRowsRepairsTornAppendAndKeepsOneOutcomePerTarget)
{
  const auto dir = root / "runs" / "baseline";
  const auto csv = dir / "results.csv";
  auto hit = gt::classifyResult(data.geometry, data.targets[0], success());
  gt::PlanRecord cutoff;
  cutoff.error_msg = "Cutoff, awaiting \"cancel\"\nresult";
  {
    gt::RunStore store(root, "baseline", data, "GridBased", "/plan", "/map");
    EXPECT_THROW(gt::RunStore(root, "baseline", data, "GridBased", "/plan", "/map"),
      std::system_error);
    store.beginObservation(1.0, data.map_id);
    store.record("target-a", hit);
    store.record("target-b", cutoff);
    store.finishObservation("partial", std::nullopt);
  }
  std::ofstream(csv, std::ios::app) << "target-b,failed,\"unfinished";
  {
    gt::RunStore store(root, "baseline", data, "GridBased", "/plan", "/map");
    EXPECT_TRUE(store.completed("target-a"));
    EXPECT_FALSE(store.completed("target-b"));
    EXPECT_THROW(store.record("target-a", hit), std::logic_error);
    store.beginObservation(20.0, "changed-map");
    auto timeout = cutoff;
    timeout.status = "failed";
    timeout.action_status = "ABORTED";
    timeout.error_code = gt::PlanAction::Result::TIMEOUT;
    store.record("target-b", timeout);
    store.finishObservation("complete", "changed-map");
  }
  std::ifstream in(csv);
  std::string line;
  ASSERT_TRUE(std::getline(in, line));
  ASSERT_TRUE(std::getline(in, line));
  EXPECT_EQ(gt::parseCsv(line)[1], "hit");
  ASSERT_TRUE(std::getline(in, line));
  const auto fields = gt::parseCsv(line);
  ASSERT_EQ(fields.size(), 12u);
  EXPECT_EQ(fields[0], "target-b");
  EXPECT_EQ(fields[3], "207");
  EXPECT_EQ(fields[4], "Cutoff, awaiting \"cancel\" result");
  EXPECT_FALSE(std::getline(in, line));

  gt::RunStore reopened(root, "baseline", data, "GridBased", "/plan", "/map");
  EXPECT_TRUE(reopened.completed("target-b"));
  const auto history = YAML::LoadFile((dir / "run.yaml").string())["live_costmap_observations"];
  ASSERT_EQ(history.size(), 2u);
  EXPECT_TRUE(history[0]["end_map_id"].IsNull());
  EXPECT_EQ(history[1]["start_map_id"].as<std::string>(), "changed-map");
  EXPECT_EQ(history[1]["timeout_s"].as<double>(), 20.0);
}

TEST_F(RunTest, RefusesToMixDifferentPlannerEndpointOrDatasetInOneRun)
{
  {gt::RunStore store(root, "v1", data, "GridBased", "/plan", "/map");}
  EXPECT_THROW(gt::RunStore(root, "v1", data, "Other", "/plan", "/map"), std::runtime_error);
  EXPECT_THROW(gt::RunStore(root, "v1", data, "GridBased", "/other", "/map"), std::runtime_error);
  EXPECT_THROW(gt::RunStore(root, "v1", data, "GridBased", "/plan", "/other"), std::runtime_error);
  data.id = "different";
  EXPECT_THROW(gt::RunStore(root, "v1", data, "GridBased", "/plan", "/map"), std::runtime_error);
}

TEST_F(RunTest, PreservesOutstandingIdentityAcrossRestartAndPrioritizesUntriedGoalsAfterSettlement)
{
  rclcpp_action::GoalUUID uuid{};
  uuid.front() = 0x80;
  uuid.back() = 0xff;
  const gt::PlanRecord incomplete;
  {
    gt::RunStore store(root, "run", data, "P", "/plan", "/map");
    EXPECT_THROW(store.markOutstanding("target-a", {}), std::logic_error);
    store.markOutstanding("target-a", uuid);
    EXPECT_THROW(store.clearOutstanding(), std::logic_error);
    store.record("target-a", incomplete);  // A timeout without a confirmed terminal result.
  }
  {
    gt::RunStore store(root, "run", data, "P", "/plan", "/map");
    ASSERT_TRUE(store.outstanding());
    EXPECT_EQ(store.outstanding()->uuid, uuid);
    EXPECT_EQ(store.pendingTargets().front()->id, "target-a");  // Reconcile before another goal.
    EXPECT_THROW(store.markOutstanding("target-b", uuid), std::logic_error);
    auto different = uuid;
    different.back() = 1;
    EXPECT_THROW(store.markOutstanding("target-a", different), std::logic_error);
    EXPECT_NO_THROW(store.markOutstanding("target-a", uuid));
    store.record("target-a", incomplete);
    store.clearOutstanding();  // The old goal has now reached a confirmed terminal state.
    EXPECT_EQ(store.pendingTargets().front()->id, "target-b");
    store.record("target-b", incomplete);
    EXPECT_EQ(store.pendingTargets().front()->id, "target-a");
    store.record("target-a", incomplete);
  }
  gt::RunStore reopened(root, "run", data, "P", "/plan", "/map");
  EXPECT_FALSE(reopened.outstanding());
  EXPECT_EQ(reopened.pendingTargets().front()->id, "target-b");  // Next --limit 1 gets a turn.
}

TEST_F(RunTest, RecoveredSuccessBelongsToNewUuidDespiteOlderIncompleteRowForSameTarget)
{
  rclcpp_action::GoalUUID old_id{}, new_id{};
  old_id.back() = 1;
  new_id.back() = 2;
  {
    gt::RunStore store(root, "run", data, "P", "/plan", "/map");
    store.markOutstanding("target-a", old_id);
    store.noteCutoff("First attempt timed out");
    store.clearOutstanding();
    store.markOutstanding("target-a", new_id);  // Killed after dispatch, before recording its result.
  }
  gt::RunStore reopened(root, "run", data, "P", "/plan", "/map");
  ASSERT_TRUE(reopened.recorded("target-a"));
  auto reply = success();
  reply.goal_id = old_id;
  EXPECT_THROW(reopened.recoverTerminal(reply), std::logic_error);
  reply.goal_id = new_id;
  reopened.recoverTerminal(reply);
  EXPECT_TRUE(reopened.completed("target-a"));
  EXPECT_FALSE(reopened.outstanding());
}

TEST_F(RunTest, RestoresCutoffFromCurrentUuidJournalBeforeConsideringItsRecoveredSuccess)
{
  rclcpp_action::GoalUUID uuid{};
  uuid.back() = 3;
  const auto csv = root / "runs" / "run" / "results.csv";
  std::string header;
  {
    gt::RunStore store(root, "run", data, "P", "/plan", "/map");
    std::ifstream initial(csv);
    ASSERT_TRUE(std::getline(initial, header));
    store.markOutstanding("target-a", uuid);
    store.noteCutoff("Current attempt deadline");
  }
  std::ofstream(csv) << header << '\n';  // Crash after the journal rename, before publishing its CSV row.
  gt::RunStore reopened(root, "run", data, "P", "/plan", "/map");
  ASSERT_TRUE(reopened.recorded("target-a"));
  auto late = success();
  late.goal_id = uuid;
  reopened.recoverTerminal(late);
  EXPECT_FALSE(reopened.completed("target-a"));
  EXPECT_FALSE(reopened.outstanding());
  std::ifstream rows(csv);
  std::string line;
  ASSERT_TRUE(std::getline(rows, line));
  ASSERT_TRUE(std::getline(rows, line));
  EXPECT_EQ(gt::parseCsv(line)[4], "Current attempt deadline");
}
}  // namespace
