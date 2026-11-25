#include "core.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>

#include <opencv2/imgcodecs.hpp>
#include <yaml-cpp/yaml.h>

namespace fs = std::filesystem;

namespace grid_plan_tester
{
namespace
{
constexpr char kHeader[] = "id,kind,group,mx,my,cost,x,y,yaw";

std::ostringstream numbers()
{
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::setprecision(std::numeric_limits<double>::max_digits10);
  return out;
}

// Stable fingerprint for comparing test inputs across processes. Exclude live timestamps.
class Fingerprint
{
public:
  void add(const unsigned char * bytes, size_t count)
  {
    for (size_t i = 0; i < count; ++i) {
      value_ ^= bytes[i];
      value_ *= UINT64_C(1099511628211);
    }
  }

  void add(const std::string & text)
  {
    add(reinterpret_cast<const unsigned char *>(text.data()), text.size());
  }

  std::string str() const
  {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << value_;
    return out.str();
  }

private:
  uint64_t value_{UINT64_C(14695981039346656037)};
};

std::string mapId(const MapGeometry & g, const unsigned char * pixels)
{
  auto out = numbers();
  out << g.frame << '\n' << g.width << ' ' << g.height << ' ' << g.resolution
      << ' ' << g.origin_x << ' ' << g.origin_y << ' ' << g.origin_yaw << '\n';
  Fingerprint hash;
  hash.add(out.str());
  hash.add(pixels, static_cast<size_t>(g.width) * g.height);
  return hash.str();
}

std::string datasetId(
  const std::string & map_id, const geometry_msgs::msg::Pose & start,
  const std::string & csv)
{
  auto out = numbers();
  const auto & p = start.position;
  const auto & q = start.orientation;
  out << map_id << '\n' << p.x << ' ' << p.y << ' ' << p.z << ' '
      << q.x << ' ' << q.y << ' ' << q.z << ' ' << q.w << '\n';
  Fingerprint hash;
  hash.add(out.str());
  hash.add(csv);
  return hash.str();
}

std::string targetCsv(const std::vector<Target> & targets)
{
  auto out = numbers();
  out << kHeader << '\n';
  for (const auto & t : targets) {
    out << t.id << ',' << t.kind << ',' << t.group << ',' << t.mx << ',' << t.my << ','
        << t.cost << ',' << t.x << ',' << t.y << ',' << t.yaw << '\n';
  }
  return out.str();
}

void checkStart(const geometry_msgs::msg::Pose & start)
{
  const auto & p = start.position;
  const auto & q = start.orientation;
  const double norm = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
  if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
    !std::isfinite(norm) || std::abs(norm - 1.0) > 1e-3)
  {
    throw std::runtime_error("Invalid frozen start pose");
  }
}

void writeFile(const fs::path & path, const std::string & text)
{
  std::ofstream out;
  out.exceptions(std::ios::badbit | std::ios::failbit);
  out.open(path, std::ios::binary);
  out << text;
  out.close();
}

YAML::Node stampNode(const builtin_interfaces::msg::Time & stamp)
{
  YAML::Node result;
  result["sec"] = stamp.sec;
  result["nanosec"] = stamp.nanosec;
  return result;
}
}  // namespace

cv::Point2d MapGeometry::center(int mx, int my) const
{
  const double dx = (mx + 0.5) * resolution, dy = (my + 0.5) * resolution;
  const double c = std::cos(origin_yaw), s = std::sin(origin_yaw);
  return {origin_x + c * dx - s * dy, origin_y + s * dx + c * dy};
}

cv::Point MapGeometry::cell(double x, double y) const
{
  const double dx = x - origin_x, dy = y - origin_y;
  const double c = std::cos(origin_yaw), s = std::sin(origin_yaw);
  const double mx = (c * dx + s * dy) / resolution;
  const double my = (-s * dx + c * dy) / resolution;
  if (!(mx >= 0 && my >= 0 && mx < width && my < height)) {
    return {-1, -1};
  }
  return {static_cast<int>(mx), static_cast<int>(my)};
}

MapGeometry geometryOf(const nav2_msgs::msg::Costmap & map)
{
  const auto & m = map.metadata;
  const auto & q = m.origin.orientation;
  const auto & p = m.origin.position;
  const double norm = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
  const uint64_t cells = static_cast<uint64_t>(m.size_x) * m.size_y;
  if (map.header.frame_id.empty() || !m.size_x || !m.size_y ||
    cells > static_cast<uint64_t>(std::numeric_limits<int>::max()) || cells != map.data.size() ||
    !std::isfinite(m.resolution) || m.resolution <= 0 ||
    !std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(norm) ||
    std::abs(norm - 1.0) > 1e-3 || std::abs(q.x) > 1e-5 || std::abs(q.y) > 1e-5)
  {
    throw std::runtime_error("Invalid raw costmap geometry, origin orientation, or data size");
  }
  return {map.header.frame_id, static_cast<int>(m.size_x), static_cast<int>(m.size_y),
    m.resolution, p.x, p.y, std::atan2(2 * q.w * q.z, q.w * q.w - q.z * q.z)};
}

std::string costmapId(const nav2_msgs::msg::Costmap & map)
{
  return mapId(geometryOf(map), map.data.data());
}

bool validRunName(const std::string & name)
{
  return !name.empty() && name != "." && name != ".." &&
    name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-") ==
    std::string::npos;
}

std::string csvField(std::string value)
{
  std::replace(value.begin(), value.end(), '\r', ' ');
  std::replace(value.begin(), value.end(), '\n', ' ');
  if (value.find_first_of(",\"") == std::string::npos) {
    return value;
  }
  std::string result = "\"";
  for (char c : value) {
    if (c == '"') {
      result += '"';
    }
    result += c;
  }
  return result + '"';
}

std::vector<std::string> parseCsv(const std::string & line)
{
  std::vector<std::string> result;
  size_t at = 0;
  for (;;) {
    std::string field;
    if (at < line.size() && line[at] == '"') {
      ++at;
      for (;;) {
        if (at == line.size()) {
          throw std::runtime_error("Unterminated quoted CSV field");
        }
        const char next = line[at++];
        if (next != '"') {
          field += next;
        } else if (at < line.size() && line[at] == '"') {
          field += line[at++];
        } else {
          break;
        }
      }
    } else {
      const size_t end = line.find_first_of(",\"", at);
      field = line.substr(at, end - at);
      at += field.size();
    }
    result.push_back(std::move(field));
    if (at == line.size()) {
      return result;
    }
    if (line[at++] != ',') {
      throw std::runtime_error("Invalid CSV field separator");
    }
  }
}

void saveDataset(
  const fs::path & destination, const nav2_msgs::msg::Costmap & map,
  const geometry_msgs::msg::Pose & start, const builtin_interfaces::msg::Time & start_stamp,
  const SamplingOptions & options, const std::vector<Target> & targets)
{
  const auto g = geometryOf(map);
  checkStart(start);
  const auto csv = targetCsv(targets);
  const auto map_id = mapId(g, map.data.data());
  YAML::Node meta;
  meta["schema_version"] = 1;
  meta["dataset_id"] = datasetId(map_id, start, csv);
  meta["map_id"] = map_id;
  meta["frame"] = g.frame;
  meta["width"] = g.width;
  meta["height"] = g.height;
  meta["resolution"] = g.resolution;
  meta["origin_x"] = g.origin_x;
  meta["origin_y"] = g.origin_y;
  meta["origin_yaw"] = g.origin_yaw;
  meta["costmap_stamp"] = stampNode(map.header.stamp);
  meta["start_tf_stamp"] = stampNode(start_stamp);
  auto pose = meta["start"];
  pose["x"] = start.position.x;
  pose["y"] = start.position.y;
  pose["z"] = start.position.z;
  pose["qx"] = start.orientation.x;
  pose["qy"] = start.orientation.y;
  pose["qz"] = start.orientation.z;
  pose["qw"] = start.orientation.w;
  auto sampling = meta["sampling"];
  sampling["spacing"] = options.spacing;
  sampling["adaptive_free_tiles"] = options.adaptive_free_tiles;
  sampling["max_free_per_component"] = options.max_free_per_component;
  sampling["merge_radius"] = options.merge_radius;
  sampling["min_cost"] = options.min_cost;
  sampling["max_cost"] = options.max_cost;
  sampling["inflation_cost_step"] = options.inflation_cost_step;
  sampling["goal_yaw"] = options.goal_yaw;

  const fs::path dest = fs::absolute(destination).lexically_normal();
  if (fs::exists(dest)) {
    throw std::runtime_error("Dataset already exists: " + dest.string());
  }
  fs::create_directories(dest.parent_path());
  std::string staging = dest.string() + ".sampling-XXXXXX";
  if (!::mkdtemp(staging.data())) {
    throw std::runtime_error("Cannot create dataset staging directory beside " + dest.string());
  }
  try {
    const fs::path dir = staging;
    // OpenCV's external-buffer constructor is mutable; imwrite only reads the view.
    const cv::Mat pixels(g.height, g.width, CV_8UC1, const_cast<uint8_t *>(map.data.data()));
    if (!cv::imwrite((dir / "costmap.png").string(), pixels)) {
      throw std::runtime_error("Failed to write costmap.png");
    }
    writeFile(dir / "targets.csv", csv);
    writeFile(dir / "manifest.yaml", YAML::Dump(meta) + "\n");
    if (::renameat2(AT_FDCWD, dir.c_str(), AT_FDCWD, dest.c_str(), RENAME_NOREPLACE) != 0) {
      throw std::system_error(errno, std::generic_category(), "Cannot publish dataset " + dest.string());
    }
  } catch (...) {
    fs::remove_all(staging);
    throw;
  }
}

Dataset loadDataset(const fs::path & directory)
{
  const auto meta = YAML::LoadFile((directory / "manifest.yaml").string());
  if (meta["schema_version"].as<int>() != 1) {
    throw std::runtime_error("Unsupported dataset schema");
  }
  Dataset d;
  auto & g = d.geometry;
  g = {meta["frame"].as<std::string>(), meta["width"].as<int>(), meta["height"].as<int>(),
    meta["resolution"].as<double>(), meta["origin_x"].as<double>(), meta["origin_y"].as<double>(),
    meta["origin_yaw"].as<double>()};
  if (!(std::isfinite(g.resolution) && g.resolution > 0) || !std::isfinite(g.origin_x) ||
    !std::isfinite(g.origin_y) || !std::isfinite(g.origin_yaw) || g.frame.empty())
  {
    throw std::runtime_error("Invalid dataset geometry");
  }
  const auto pose = meta["start"];
  d.start.position.x = pose["x"].as<double>();
  d.start.position.y = pose["y"].as<double>();
  d.start.position.z = pose["z"].as<double>();
  d.start.orientation.x = pose["qx"].as<double>();
  d.start.orientation.y = pose["qy"].as<double>();
  d.start.orientation.z = pose["qz"].as<double>();
  d.start.orientation.w = pose["qw"].as<double>();
  checkStart(d.start);
  const auto pixels = cv::imread((directory / "costmap.png").string(), cv::IMREAD_UNCHANGED);
  if (pixels.empty() || pixels.type() != CV_8UC1 || pixels.cols != g.width || pixels.rows != g.height) {
    throw std::runtime_error("Snapshot image does not match manifest geometry");
  }
  d.map_id = mapId(g, pixels.ptr());
  if (d.map_id != meta["map_id"].as<std::string>()) {
    throw std::runtime_error("Snapshot pixels or geometry changed since sampling");
  }
  std::ifstream file(directory / "targets.csv", std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open targets.csv");
  }
  const std::string contents{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  d.id = datasetId(d.map_id, d.start, contents);
  if (d.id != meta["dataset_id"].as<std::string>()) {
    throw std::runtime_error("Frozen start or targets changed since sampling");
  }
  std::istringstream in(contents);
  std::string line;
  if (!std::getline(in, line) || line != kHeader) {
    throw std::runtime_error("Invalid targets.csv header");
  }
  std::set<std::string> ids;
  while (std::getline(in, line)) {
    const auto fields = parseCsv(line);
    if (fields.size() != 9) {
      throw std::runtime_error("Invalid target in targets.csv");
    }
    Target t{fields[0], fields[1], std::stoi(fields[2]), std::stoi(fields[3]),
      std::stoi(fields[4]), std::stoi(fields[5]), std::stod(fields[6]), std::stod(fields[7]),
      std::stod(fields[8])};
    if (!ids.insert(t.id).second || t.mx < 0 || t.my < 0 || t.mx >= g.width || t.my >= g.height ||
      g.cell(t.x, t.y) != cv::Point(t.mx, t.my) || pixels.at<uint8_t>(t.my, t.mx) != t.cost ||
      !std::isfinite(t.yaw))
    {
      throw std::runtime_error("Invalid or duplicate target: " + t.id);
    }
    d.targets.push_back(std::move(t));
  }
  return d;
}
}  // namespace grid_plan_tester
