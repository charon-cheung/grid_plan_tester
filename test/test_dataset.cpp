#include "core.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

namespace gt = grid_plan_tester;
namespace fs = std::filesystem;

namespace
{
class DatasetTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    std::string path = (fs::temp_directory_path() / "grid-plan-dataset-test-XXXXXX").string();
    ASSERT_NE(::mkdtemp(path.data()), nullptr);
    root = path;
    map.header.frame_id = "map";
    map.metadata.size_x = 3;
    map.metadata.size_y = 2;
    map.metadata.resolution = 0.01f;
    map.metadata.origin.position.x = -2.776;
    map.metadata.origin.position.y = -27.535;
    map.metadata.origin.orientation.z = std::sin(0.37);
    map.metadata.origin.orientation.w = std::cos(0.37);
    map.data = {255, 254, 253, 200, 0, 1};
    start.position.x = 7.065429741571183;
    start.position.y = -7.776865248538132;
    start.orientation.z = std::sin(0.81);
    start.orientation.w = std::cos(0.81);
  }
  void TearDown() override {fs::remove_all(root);}

  fs::path root;
  nav2_msgs::msg::Costmap map;
  geometry_msgs::msg::Pose start;
};

TEST_F(DatasetTest, PreservesRawBytesFrozenPoseAndRotatedCellCenters)
{
  const auto g = gt::geometryOf(map);
  auto xy = g.center(1, 1);
  const gt::Target t{"free-1", "free", 1, 1, 1, 0, xy.x, xy.y, -0.3};
  const auto dir = root / "snapshot";
  gt::saveDataset(dir, map, start, map.header.stamp, {}, {t});
  const auto d = gt::loadDataset(dir);
  ASSERT_EQ(d.targets.size(), 1u);
  EXPECT_EQ(d.start, start);
  EXPECT_EQ(d.targets[0].yaw, t.yaw);
  EXPECT_EQ(d.geometry.cell(xy.x, xy.y), cv::Point(1, 1));
  EXPECT_EQ(d.geometry.cell(g.center(-1, 0).x, g.center(-1, 0).y), cv::Point(-1, -1));
  const auto pixels = cv::imread((dir / "costmap.png").string(), cv::IMREAD_UNCHANGED);
  EXPECT_EQ(std::vector<uint8_t>(pixels.datastart, pixels.dataend), map.data);
  EXPECT_THROW(gt::saveDataset(dir, map, start, map.header.stamp, {}, {}), std::runtime_error);
  EXPECT_EQ(gt::loadDataset(dir).id, d.id);
}

TEST_F(DatasetTest, ComparesCostmapContentIndependentlyOfTimestampsAndDetectsChangedTargets)
{
  const auto dir = root / "input";
  gt::saveDataset(dir, map, start, map.header.stamp, {}, {});
  const auto frozen = gt::loadDataset(dir);
  ++map.header.stamp.sec;
  EXPECT_EQ(gt::costmapId(map), frozen.map_id);
  map.data[0] = 0;
  EXPECT_NE(gt::costmapId(map), frozen.map_id);
  std::ofstream(dir / "targets.csv", std::ios::app) << "tampered\n";
  EXPECT_THROW(gt::loadDataset(dir), std::runtime_error);
}

TEST(DatasetCsv, DiagnosticFieldsRoundTripAndEachResultUsesOneLine)
{
  const auto fields = gt::parseCsv("foo," + gt::csvField("a,\"b\"\nc") + ",");
  EXPECT_EQ(fields, (std::vector<std::string>{"foo", "a,\"b\" c", ""}));
  EXPECT_THROW(gt::parseCsv("\"part"), std::runtime_error);
}
}  // namespace
