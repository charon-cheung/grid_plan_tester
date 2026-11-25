#include "core.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace grid_plan_tester
{
namespace
{
constexpr int lethal_cost = 254;

struct Centroid
{
  int count{};
  int64_t sum_x{}, sum_y{};

  void add(int x, int y)
  {
    ++count;
    sum_x += x;
    sum_y += y;
  }

  // Orders cells by squared distance to the centroid without rounding its coordinates.
  long double score(const cv::Point & p) const
  {
    const long double x = p.x, y = p.y;
    return count * (x * x + y * y) - 2 * (x * sum_x + y * sum_y);
  }
};

int64_t square(int value)
{
  return static_cast<int64_t>(value) * value;
}

Target makeTarget(
  const char * kind, int group, const cv::Point & p, int cost,
  const MapGeometry & geometry, double yaw)
{
  const auto world = geometry.center(p.x, p.y);
  const auto id = std::string(kind) + "_" + std::to_string(p.x) + "_" + std::to_string(p.y);
  return {id, kind, group, p.x, p.y, cost, world.x, world.y, yaw};
}

void limitFreeSamples(std::vector<cv::Point> & points, const Centroid & center, int cap)
{
  if (cap == 0 || points.size() <= static_cast<size_t>(cap)) {
    return;
  }
  auto next = std::min_element(points.begin(), points.end(), [&](const auto & a, const auto & b) {
      return center.score(a) < center.score(b);
    });
  std::vector<int64_t> nearest(points.size(), std::numeric_limits<int64_t>::max());
  std::vector<cv::Point> chosen;
  chosen.reserve(cap);
  while (true) {
    const auto p = *next;
    chosen.push_back(p);
    if (chosen.size() == static_cast<size_t>(cap)) {
      break;
    }
    for (size_t i = 0; i < points.size(); ++i) {
      const auto distance = square(points[i].x - p.x) + square(points[i].y - p.y);
      nearest[i] = std::min(nearest[i], distance);
    }
    next = points.begin() + (std::max_element(nearest.begin(), nearest.end()) - nearest.begin());
  }
  points = std::move(chosen);
}

// Mark the base tiles that retain a representative. Only complete, all-zero squares merge.
cv::Mat1b adaptiveFreeTiles(const cv::Mat & map, int tile)
{
  cv::Mat1b keep(1 + (map.rows - 1) / tile, 1 + (map.cols - 1) / tile, uchar{1});
  for (const int factor : {2, 4}) {
    if (tile > std::min(map.rows, map.cols) / factor) {continue;}
    const int side = tile * factor;
    // Even-sized base tiles pick the lower of their two center cells. Keep the
    // base representative nearest the merged center, breaking ties toward the origin.
    const int middle = factor / 2 - tile % 2;
    for (int y = 0; y <= map.rows - side; y += side) {
      for (int x = 0; x <= map.cols - side; x += side) {
        if (cv::countNonZero(map(cv::Rect(x, y, side, side))) != 0) {continue;}
        const int row = y / tile, col = x / tile;
        keep(cv::Rect(col, row, factor, factor)).setTo(0);
        keep(row + middle, col + middle) = 1;
      }
    }
  }
  return keep;
}

void appendFreeTargets(
  const cv::Mat & map, const MapGeometry & geometry, const SamplingOptions & options,
  std::vector<Target> & out)
{
  cv::Mat1i labels;
  const int count = cv::connectedComponents(map == 0, labels, 4, CV_32S, cv::CCL_SAUF);
  struct Component
  {
    Centroid center;
    std::vector<cv::Point> points;
    cv::Point supplement{-1, -1};
  };
  std::vector<Component> components(count);
  const double max_side = std::max(map.cols, map.rows);
  const int tile = std::max(
    1, static_cast<int>(std::round(std::min(options.spacing / geometry.resolution, max_side))));
  const auto keep = options.adaptive_free_tiles ? adaptiveFreeTiles(map, tile) : cv::Mat1b{};
  for (int top = 0; top < map.rows; ) {
    const int bottom = top + std::min(tile, map.rows - top);
    for (int left = 0; left < map.cols; ) {
      const int right = left + std::min(tile, map.cols - left);
      const double cx = left + (right - left - 1) * 0.5;
      const double cy = top + (bottom - top - 1) * 0.5;
      double closest = std::numeric_limits<double>::infinity();
      cv::Point best;
      int best_group = 0;
      for (int y = top; y < bottom; ++y) {
        const auto * row = labels.ptr<int>(y);
        for (int x = left; x < right; ++x) {
          const int group = row[x];
          if (group != 0) {
            components[group].center.add(x, y);
            const double distance = (x - cx) * (x - cx) + (y - cy) * (y - cy);
            if (distance < closest) {
              closest = distance;
              best = {x, y};
              best_group = group;
            }
          }
        }
      }
      if (best_group != 0 && (keep.empty() || keep(top / tile, left / tile))) {
        components[best_group].points.push_back(best);
      }
      left = right;
    }
    top = bottom;
  }

  // Tiles may straddle several components. Give each missed one its nearest cell to its centroid.
  if (std::any_of(components.begin() + 1, components.end(),
    [](const auto & c) {return c.points.empty();}))
  {
    for (int y = 0; y < map.rows; ++y) {
      const auto * row = labels.ptr<int>(y);
      for (int x = 0; x < map.cols; ++x) {
        const int group = row[x];
        if (group == 0 || !components[group].points.empty()) {
          continue;
        }
        auto & c = components[group];
        const cv::Point p(x, y);
        if (c.supplement.x < 0 || c.center.score(p) < c.center.score(c.supplement)) {
          c.supplement = p;
        }
      }
    }
  }
  for (int group = 1; group < count; ++group) {
    auto & c = components[group];
    if (c.points.empty()) {
      c.points.push_back(c.supplement);
    }
    limitFreeSamples(c.points, c.center, options.max_free_per_component);
    for (const auto & p : c.points) {
      out.push_back(makeTarget("free", group, p, 0, geometry, options.goal_yaw));
    }
  }
}

// Exact separable Euclidean feature transform. The output stores the row-major index of
// the nearest original lethal cell. The input mask becomes the union of the dilation disks.
cv::Mat1i nearestLethal(cv::Mat1b & mask, double radius_cells)
{
  cv::Mat1i owner(mask.size());
  for (int y = 0; y < mask.rows; ++y) {
    const auto * lethal = mask.ptr<uchar>(y);
    auto * nearest_x = owner.ptr<int>(y);
    int left = -1;
    for (int x = 0; x < mask.cols; ++x) {
      if (lethal[x]) {
        left = x;
      }
      nearest_x[x] = left;
    }
    int right = -1;
    for (int x = mask.cols - 1; x >= 0; --x) {
      if (lethal[x]) {
        right = x;
      }
      if (right >= 0 && (nearest_x[x] < 0 || right - x < x - nearest_x[x])) {
        nearest_x[x] = right;
      }
    }
  }

  struct Site
  {
    cv::Point cell;
    int64_t distance_at_zero, first_row;
  };
  std::vector<Site> envelope;
  envelope.reserve(mask.rows);
  const double radius_squared = radius_cells * radius_cells;
  for (int x = 0; x < mask.cols; ++x) {
    envelope.clear();
    for (int y = 0; y < mask.rows; ++y) {
      const int sx = owner(y, x);
      if (sx < 0) {
        continue;
      }
      // Each row contributes a parabola (x-sx)^2 + (query_y-y)^2. Keep its
      // first winning integer row; a tie belongs to the earlier original row.
      Site site{{sx, y}, square(x - sx) + square(y), 0};
      while (!envelope.empty()) {
        const auto & prev = envelope.back();
        const int64_t delta = site.distance_at_zero - prev.distance_at_zero;
        const int64_t divisor = 2LL * (y - prev.cell.y);
        site.first_row = delta / divisor - (delta % divisor < 0) + 1;
        if (site.first_row > prev.first_row) {
          break;
        }
        envelope.pop_back();
      }
      if (envelope.empty()) {
        site.first_row = 0;
      }
      if (site.first_row < mask.rows) {
        envelope.push_back(site);
      }
    }
    size_t active = 0;
    for (int y = 0; y < mask.rows; ++y) {
      if (active + 1 < envelope.size() && envelope[active + 1].first_row == y) {
        ++active;
      }
      const auto p = envelope[active].cell;
      owner(y, x) = p.y * mask.cols + p.x;
      mask(y, x) = square(x - p.x) + square(y - p.y) <= radius_squared ? 255 : 0;
    }
  }
  return owner;
}

void appendInflationTargets(
  const cv::Mat & map, const MapGeometry & geometry, const SamplingOptions & options,
  std::vector<Target> & out)
{
  cv::Mat1b dilated = map == lethal_cost;
  if (cv::countNonZero(dilated) == 0) {
    return;
  }
  auto owners = nearestLethal(dilated, options.merge_radius / geometry.resolution);
  cv::Mat1i labels;
  const int count = cv::connectedComponents(dilated, labels, 8, CV_32S, cv::CCL_SAUF);
  dilated.release();
  struct Band
  {
    Centroid center;
    std::vector<uchar> costs;
    int twice_median{}, best_error{std::numeric_limits<int>::max()};
    cv::Point best{-1, -1};
  };
  std::vector<std::map<int, Band>> groups(count);
  const auto bandIndex = [&](int cost) {
      return (cost - options.min_cost) / options.inflation_cost_step;
    };
  for (int y = 0; y < map.rows; ++y) {
    const auto * costs = map.ptr<uchar>(y);
    auto * row = owners.ptr<int>(y);
    for (int x = 0; x < map.cols; ++x) {
      const int cost = costs[x];
      if (cost >= options.min_cost && cost <= options.max_cost) {
        const int source = row[x];
        row[x] = labels(source / map.cols, source % map.cols);
        auto & g = groups[row[x]][bandIndex(cost)];
        g.center.add(x, y);
        g.costs.push_back(costs[x]);
      } else {
        row[x] = 0;
      }
    }
  }
  labels.release();
  for (auto & group : groups) {
    for (auto & entry : group) {
      auto & g = entry.second;
      auto mid = g.costs.begin() + g.costs.size() / 2;
      std::nth_element(g.costs.begin(), mid, g.costs.end());
      const int lower = g.costs.size() % 2 ? *mid : *std::max_element(g.costs.begin(), mid);
      g.twice_median = lower + *mid;
    }
  }
  for (int y = 0; y < map.rows; ++y) {
    const auto * row = owners.ptr<int>(y);
    const auto * costs = map.ptr<uchar>(y);
    for (int x = 0; x < map.cols; ++x) {
      if (row[x] == 0) {
        continue;
      }
      auto & g = groups[row[x]].at(bandIndex(costs[x]));
      const int error = std::abs(2 * costs[x] - g.twice_median);
      const cv::Point p(x, y);
      if (error < g.best_error ||
        (error == g.best_error && g.center.score(p) < g.center.score(g.best)))
      {
        g.best_error = error;
        g.best = p;
      }
    }
  }
  for (int group = 1; group < count; ++group) {
    for (const auto & entry : groups[group]) {
      const auto p = entry.second.best;
      out.push_back(makeTarget("inflation", group, p, map.at<uchar>(p), geometry, options.goal_yaw));
    }
  }
}
}  // namespace

std::vector<Target> sampleTargets(
  const cv::Mat & costmap, const MapGeometry & geometry, const SamplingOptions & options)
{
  if (costmap.empty() || costmap.dims != 2 || costmap.type() != CV_8UC1 ||
    costmap.cols != geometry.width || costmap.rows != geometry.height ||
    costmap.total() > static_cast<size_t>(std::numeric_limits<int>::max()) ||
    !std::isfinite(geometry.resolution) || geometry.resolution <= 0.0 ||
    !std::isfinite(geometry.origin_x) || !std::isfinite(geometry.origin_y) ||
    !std::isfinite(geometry.origin_yaw))
  {
    throw std::invalid_argument("Invalid costmap or map geometry");
  }
  if (!std::isfinite(options.spacing) || options.spacing <= 0.0 ||
    !std::isfinite(options.merge_radius) || options.merge_radius < 0.0 ||
    options.max_free_per_component < 0 || options.min_cost < 1 ||
    options.max_cost > 252 || options.min_cost > options.max_cost ||
    options.inflation_cost_step < 1 || options.inflation_cost_step > 252 ||
    !std::isfinite(options.goal_yaw))
  {
    throw std::invalid_argument("Invalid sampling spacing, merge radius, cap, cost range or yaw");
  }
  std::vector<Target> targets;
  appendFreeTargets(costmap, geometry, options, targets);
  appendInflationTargets(costmap, geometry, options, targets);
  return targets;
}
}  // namespace grid_plan_tester
