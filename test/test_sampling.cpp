#include "core.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace grid_plan_tester
{
namespace
{
MapGeometry geometry(const cv::Mat & map, double resolution = 1.0)
{
  return {"map", map.cols, map.rows, resolution, 0.0, 0.0, 0.0};
}

TEST(Sampling, ChoosesNearestFreeCellToEachClippedTileCenter)
{
  cv::Mat map(3, 6, CV_8U, cv::Scalar(255));
  map.row(1).setTo(0);
  auto grid = geometry(map, 0.5);
  grid.origin_x = 10.0;
  grid.origin_y = -2.0;
  grid.origin_yaw = std::acos(-1.0) / 2.0;
  SamplingOptions options;
  options.spacing = 2.0;
  options.goal_yaw = -0.45;

  const auto targets = sampleTargets(map, grid, options);
  ASSERT_EQ(targets.size(), 2u);
  EXPECT_EQ(targets[0].kind, "free");
  EXPECT_EQ(targets[0].cost, 0);
  EXPECT_EQ(targets[0].mx, 1);
  EXPECT_EQ(targets[0].my, 1);
  EXPECT_NEAR(targets[0].x, 9.25, 1e-10);
  EXPECT_NEAR(targets[0].y, -1.25, 1e-10);
  EXPECT_DOUBLE_EQ(targets[0].yaw, -0.45);
  EXPECT_EQ(targets[1].mx, 4);
  EXPECT_EQ(targets[1].my, 1);
  EXPECT_EQ(targets[0].group, targets[1].group);
  EXPECT_NE(targets[0].id, targets[1].id);
}

TEST(Sampling, RepresentsEvenTinyFourConnectedFreeRegionsSharingOneTile)
{
  cv::Mat map(5, 5, CV_8U, cv::Scalar(255));
  map.at<uchar>(0, 0) = 0;
  map.row(2).colRange(1, 4).setTo(0);
  map.at<uchar>(3, 4) = 0;
  SamplingOptions options;
  options.spacing = 20.0;
  const auto targets = sampleTargets(map, geometry(map), options);

  std::set<std::pair<int, int>> cells;
  std::set<int> groups;
  for (const auto & target : targets) {
    EXPECT_EQ(target.kind, "free");
    cells.emplace(target.mx, target.my);
    groups.insert(target.group);
  }
  EXPECT_EQ(cells, (std::set<std::pair<int, int>>{{0, 0}, {2, 2}, {4, 3}}));
  EXPECT_EQ(groups.size(), 3u);
}

TEST(Sampling, QuickCapSpansEachFreeRegionAndPreservesTargetIds)
{
  cv::Mat map(1, 31, CV_8U, cv::Scalar(0));
  map.at<uchar>(0, 17) = 255;
  SamplingOptions options;
  options.spacing = 1.0;
  const auto all = sampleTargets(map, geometry(map), options);
  ASSERT_EQ(all.size(), 30u);
  std::map<int, std::string> ids;
  for (const auto & target : all) {
    ids.emplace(target.mx, target.id);
  }
  options.max_free_per_component = 5;
  const auto quick = sampleTargets(map, geometry(map), options);
  ASSERT_EQ(quick.size(), 10u);
  std::map<int, std::vector<int>> positions;
  for (const auto & target : quick) {
    EXPECT_EQ(ids.at(target.mx), target.id);
    positions[target.group].push_back(target.mx);
  }
  ASSERT_EQ(positions.size(), 2u);
  for (auto & entry : positions) {
    auto & xs = entry.second;
    std::sort(xs.begin(), xs.end());
    ASSERT_EQ(xs.size(), 5u);
    EXPECT_TRUE((xs.front() == 0 && xs.back() == 16) || (xs.front() == 18 && xs.back() == 30));
    for (size_t i = 1; i < xs.size(); ++i) {
      EXPECT_GE(xs[i] - xs[i - 1], 3);
    }
  }
}

TEST(Sampling, MergesOnlyCompleteZeroCostSquaresAtTwoScales)
{
  // One complete open square, a second with a single non-free cell, and a clipped column.
  for (const int cost : {1, 253, 254, 255}) {
    cv::Mat map(4, 9, CV_8U, cv::Scalar(0));
    map.at<uchar>(0, 4) = static_cast<uchar>(cost);
    SamplingOptions options;
    options.spacing = 0.5;
    const auto targets = sampleTargets(map, geometry(map, 0.5), options);
    int open = 0, mixed = 0, clipped = 0;
    for (const auto & target : targets) {
      ASSERT_EQ(target.kind, "free");
      EXPECT_EQ(map.at<uchar>(target.my, target.mx), 0);
      if (target.mx < 4) {
        ++open;
      } else if (target.mx < 8) {
        ++mixed;
      } else {
        ++clipped;
      }
    }
    EXPECT_EQ(open, 1);
    EXPECT_EQ(mixed, 6);  // Three merged 2x2 squares plus three individual free cells.
    EXPECT_EQ(clipped, 4);
    options.adaptive_free_tiles = false;
    EXPECT_EQ(sampleTargets(map, geometry(map, 0.5), options).size(), 35u);
  }
}

TEST(Sampling, InflationResolvesMedianCostThenCentroidThenRowColumn)
{
  const cv::Mat map = (cv::Mat_<uchar>(1, 11) <<
    10, 254, 30, 60, 253, 60, 90, 120, 252, 254, 200);
  SamplingOptions options;
  options.merge_radius = 0.0;
  const auto targets = sampleTargets(map, geometry(map), options);
  ASSERT_EQ(targets.size(), 2u);
  // x=5 is equidistant and belongs to the left obstacle. Its group's median is 45,
  // with costs 30 and 60 equally close and cells 2 and 3 equally near the centroid.
  EXPECT_EQ(targets[0].kind, "inflation");
  EXPECT_EQ(targets[0].mx, 2);
  EXPECT_EQ(targets[0].cost, 30);
  EXPECT_EQ(targets[1].mx, 7);
  EXPECT_EQ(targets[1].cost, 120);
  EXPECT_NE(targets[0].group, targets[1].group);
}

TEST(Sampling, AssociatesInflationWithTheExactEuclideanNearestLethalCell)
{
  cv::Mat map(10, 21, CV_8U, cv::Scalar(255));
  map.at<uchar>(0, 20) = 254;
  map.at<uchar>(9, 18) = 254;
  map.at<uchar>(8, 18) = 80;
  map.at<uchar>(0, 0) = 25;
  // The origin is 20 cells from the first obstacle and sqrt(405) from the second.
  // A five-pixel chamfer approximation can reverse these distances.
  SamplingOptions options;
  options.merge_radius = 0.0;
  const auto targets = sampleTargets(map, geometry(map), options);
  ASSERT_EQ(targets.size(), 2u);
  EXPECT_EQ(targets[0].mx, 0);
  EXPECT_EQ(targets[0].my, 0);
  EXPECT_EQ(targets[1].mx, 18);
  EXPECT_EQ(targets[1].my, 8);
}

TEST(Sampling, RepresentsEveryOccupiedCostBandOfEachObstacle)
{
  const cv::Mat map = (cv::Mat_<uchar>(1, 13) <<
    4, 254, 5, 44, 45, 84, 255, 205, 241, 254, 240, 5, 0);
  SamplingOptions options;
  options.min_cost = 5;
  options.max_cost = 240;
  options.merge_radius = 0.0;
  options.inflation_cost_step = 40;
  const auto targets = sampleTargets(map, geometry(map), options);
  std::map<int, std::multiset<int>> bands;
  int free = 0;
  for (const auto & t : targets) {
    if (t.kind == "free") {
      ++free;
      continue;
    }
    ASSERT_GE(t.cost, 5);
    ASSERT_LE(t.cost, 240);
    bands[t.group].insert((t.cost - 5) / 40);
  }
  EXPECT_EQ(free, 1);
  ASSERT_EQ(bands.size(), 2u);
  EXPECT_EQ(bands.at(1), (std::multiset<int>{0, 1}));
  EXPECT_EQ(bands.at(2), (std::multiset<int>{0, 5}));
}

TEST(Sampling, GroupsLethalFragmentsByEuclideanDilationInMeters)
{
  const cv::Mat map = (cv::Mat_<uchar>(1, 9) << 50, 254, 50, 100, 254, 100, 253, 120, 254);
  SamplingOptions options;
  options.merge_radius = 0.0;
  EXPECT_EQ(sampleTargets(map, geometry(map, 0.01), options).size(), 3u);
  options.merge_radius = 0.01;
  EXPECT_EQ(sampleTargets(map, geometry(map, 0.01), options).size(), 2u);
  EXPECT_EQ(sampleTargets(map, geometry(map, 0.05), options).size(), 3u);

  cv::Mat diagonal(4, 4, CV_8U, cv::Scalar(255));
  diagonal.at<uchar>(0, 0) = diagonal.at<uchar>(3, 3) = 254;
  diagonal.at<uchar>(0, 1) = diagonal.at<uchar>(3, 2) = 80;
  EXPECT_EQ(sampleTargets(diagonal, geometry(diagonal, 0.01), options).size(), 2u);
}

TEST(Sampling, StrictLethalGroupsUseEightConnectivityAndRequireAnEligibleCell)
{
  cv::Mat map(3, 7, CV_8U, cv::Scalar(255));
  map.at<uchar>(0, 6) = map.at<uchar>(1, 1) = map.at<uchar>(2, 2) = 254;
  map.at<uchar>(0, 5) = 252;
  map.at<uchar>(0, 0) = 1;
  map.at<uchar>(2, 3) = 200;
  SamplingOptions options;
  options.merge_radius = 0.0;
  EXPECT_EQ(sampleTargets(map, geometry(map), options).size(), 1u);
  options.max_cost = 252;
  const auto targets = sampleTargets(map, geometry(map), options);
  ASSERT_EQ(targets.size(), 2u);
  EXPECT_TRUE(std::any_of(targets.begin(), targets.end(), [](const auto & p) {return p.cost == 252;}));
  map.setTo(25);
  EXPECT_TRUE(sampleTargets(map, geometry(map), options).empty());
}

TEST(Sampling, RejectsInvalidGeometryAndSamplingParameters)
{
  const cv::Mat map(1, 2, CV_8U, cv::Scalar(0));
  auto grid = geometry(map);
  SamplingOptions options;
  options.spacing = 0.0;
  EXPECT_THROW(sampleTargets(map, grid, options), std::invalid_argument);
  options = {};
  options.merge_radius = -0.01;
  EXPECT_THROW(sampleTargets(map, grid, options), std::invalid_argument);
  options = {};
  options.max_free_per_component = -1;
  EXPECT_THROW(sampleTargets(map, grid, options), std::invalid_argument);
  options = {};
  options.max_cost = 253;
  EXPECT_THROW(sampleTargets(map, grid, options), std::invalid_argument);
  options.max_cost = 100;
  options.min_cost = 101;
  EXPECT_THROW(sampleTargets(map, grid, options), std::invalid_argument);
  options = {};
  options.goal_yaw = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(sampleTargets(map, grid, options), std::invalid_argument);
  options = {};
  grid.height = 2;
  EXPECT_THROW(sampleTargets(map, grid, options), std::invalid_argument);
  grid = geometry(map, 0.0);
  EXPECT_THROW(sampleTargets(map, grid, options), std::invalid_argument);
  EXPECT_THROW(sampleTargets(cv::Mat(1, 2, CV_16U), geometry(map), options), std::invalid_argument);
}
}  // namespace
}  // namespace grid_plan_tester
