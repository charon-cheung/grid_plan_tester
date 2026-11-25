#include "run_store.hpp"

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace fs = std::filesystem;

namespace grid_plan_tester
{
namespace
{
constexpr char header[] =
  "id,status,action_status,error_code,error_msg,wall_ms,planning_ms,poses,length_m,end_x,end_y,end_error_m";

void syncPath(const fs::path & path)
{
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    throw std::system_error(errno, std::generic_category(), "Cannot open " + path.string());
  }
  const int result = ::fsync(fd);
  const int error = errno;
  ::close(fd);
  if (result != 0) {
    throw std::system_error(error, std::generic_category(), "Cannot sync " + path.string());
  }
}

template<typename Writer>
void replaceFile(const fs::path & path, Writer write)
{
  const auto staging = path.string() + ".tmp";
  std::ofstream out;
  out.exceptions(std::ios::badbit | std::ios::failbit);
  out.open(staging, std::ios::binary | std::ios::trunc);
  write(out);
  out.close();
  syncPath(staging);
  fs::rename(staging, path);
  syncPath(path.parent_path());
}

bool terminalStatus(const std::string & status)
{
  if (status == "incomplete") {
    return false;
  }
  if (status == "hit" || status == "adjusted" || status == "failed") {
    return true;
  }
  throw std::runtime_error("Unknown result status " + status);
}

template<typename T>
void optionalField(std::ostream & out, const std::optional<T> & value)
{
  out << ',';
  if (value) {
    out << *value;
  }
}

std::string resultLine(const std::string & id, const PlanRecord & r)
{
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::setprecision(std::numeric_limits<double>::max_digits10);
  out << csvField(id) << ',' << r.status << ',' << csvField(r.action_status);
  optionalField(out, r.error_code);
  out << ',' << csvField(r.error_msg);
  optionalField(out, r.wall_ms);
  optionalField(out, r.planning_ms);
  out << ',' << r.poses;
  optionalField(out, r.length_m);
  optionalField(out, r.end_x);
  optionalField(out, r.end_y);
  optionalField(out, r.end_error_m);
  return out.str();
}

int64_t unixMillis()
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count();
}
}  // namespace

PlanRecord classifyResult(
  const MapGeometry & geometry, const Target & target, const PlanHandle::WrappedResult & wrapped)
{
  using Code = rclcpp_action::ResultCode;
  PlanRecord row;
  switch (wrapped.code) {
    case Code::SUCCEEDED:
      row.action_status = "SUCCEEDED";
      row.status = "failed";  // The returned path still needs validation.
      break;
    case Code::ABORTED:
      row.action_status = "ABORTED";
      row.status = "failed";
      break;
    case Code::CANCELED:
      row.action_status = "CANCELED";
      break;
    default:
      break;
  }
  if (!wrapped.result) {
    row.error_msg = "Missing action result payload";
    return row;
  }
  const auto & result = *wrapped.result;
  row.error_code = result.error_code;
  row.error_msg = result.error_msg;
  const auto & t = result.planning_time;
  if (t.sec >= 0 && t.nanosec < 1000000000u) {
    row.planning_ms = 1000.0 * t.sec + 0.000001 * t.nanosec;
  }
  const auto & path = result.path;
  row.poses = path.poses.size();
  if (wrapped.code != Code::SUCCEEDED || result.error_code != PlanAction::Result::NONE) {
    if (row.error_msg.empty()) {
      row.error_msg = "Action " + row.action_status + " with Nav2 error " +
        std::to_string(result.error_code);
    }
    return row;
  }
  if (path.poses.empty() || path.header.frame_id != geometry.frame) {
    row.error_msg = path.poses.empty() ? "Empty successful path" :
      "Path frame '" + path.header.frame_id + "' differs from frozen frame '" + geometry.frame + "'";
    return row;
  }
  double length = 0.0;
  const auto * previous = &path.poses.front().pose.position;
  for (const auto & pose : path.poses) {
    const auto & p = pose.pose.position;
    if (!pose.header.frame_id.empty() && pose.header.frame_id != geometry.frame) {
      row.error_msg = "Path pose frame '" + pose.header.frame_id + "' differs from frozen frame '" +
        geometry.frame + "'";
      return row;
    }
    if (!std::isfinite(p.x) || !std::isfinite(p.y)) {
      row.error_msg = "Nonfinite path coordinates";
      return row;
    }
    length += std::hypot(p.x - previous->x, p.y - previous->y);
    previous = &p;
  }
  const double error = std::hypot(previous->x - target.x, previous->y - target.y);
  if (!std::isfinite(length) || !std::isfinite(error)) {
    row.error_msg = "Nonfinite path length or endpoint error";
    return row;
  }
  row.length_m = length;
  row.end_x = previous->x;
  row.end_y = previous->y;
  row.end_error_m = error;
  const auto cell = geometry.cell(previous->x, previous->y);
  row.status = cell == cv::Point(target.mx, target.my) ? "hit" : "adjusted";
  if (cell.x < 0 || cell.y < 0) {
    row.error_msg = "Endpoint outside the frozen grid";
  }
  return row;
}

RunStore::RunStore(
  const fs::path & dataset_dir, const std::string & name, const Dataset & dataset,
  const std::string & planner, const std::string & action, const std::string & costmap_service)
: dataset_(dataset), dir_(dataset_dir / "runs" / name)
{
  if (!validRunName(name)) {
    throw std::invalid_argument("Invalid run name " + name);
  }
  fs::create_directories(dir_);
  lock_.reset(std::fopen((dir_ / ".lock").c_str(), "a"));
  if (!lock_ || ::flock(::fileno(lock_.get()), LOCK_EX | LOCK_NB) != 0) {
    throw std::system_error(errno, std::generic_category(), "Cannot lock run " + dir_.string());
  }
  const auto path = dir_ / "run.yaml";
  if (fs::exists(path)) {
    metadata_ = YAML::LoadFile(path.string());
    if (metadata_["schema_version"].as<int>() != 1 ||
      metadata_["dataset_id"].as<std::string>() != dataset.id ||
      metadata_["map_id"].as<std::string>() != dataset.map_id ||
      metadata_["planner_id"].as<std::string>() != planner ||
      metadata_["action"].as<std::string>() != action ||
      metadata_["costmap_service"].as<std::string>() != costmap_service)
    {
      throw std::runtime_error("Run settings or dataset differ from " + path.string());
    }
  } else {
    if (fs::exists(dir_ / "results.csv")) {
      throw std::runtime_error("Cannot identify existing results lacking " + path.string());
    }
    metadata_["schema_version"] = 1;
    metadata_["dataset_id"] = dataset.id;
    metadata_["map_id"] = dataset.map_id;
    metadata_["planner_id"] = planner;
    metadata_["action"] = action;
    metadata_["costmap_service"] = costmap_service;
    metadata_["live_costmap_observations"] = YAML::Node(YAML::NodeType::Sequence);
    saveMetadata();
  }
  for (const auto & target : dataset.targets) {
    rows_.emplace(target.id, SavedRow{});
  }
  readResults();
  const auto journal = dir_ / "outstanding.yaml";
  if (fs::exists(journal)) {
    const auto saved = YAML::LoadFile(journal.string());
    OutstandingGoal goal{saved["target_id"].as<std::string>(), {},
      saved["incomplete_row"].as<std::string>("")};
    const auto bytes = saved["uuid"].as<YAML::Binary>();
    if (rows_.count(goal.id) == 0 || bytes.size() != goal.uuid.size()) {
      throw std::runtime_error("Invalid outstanding goal in " + journal.string());
    }
    std::copy_n(bytes.data(), goal.uuid.size(), goal.uuid.begin());
    if (goal.uuid == rclcpp_action::GoalUUID{}) {
      throw std::runtime_error("Zero UUID in " + journal.string());
    }
    if (!goal.cutoff_row.empty()) {
      const auto fields = parseCsv(goal.cutoff_row);
      if (fields.size() != 12 || fields[0] != goal.id || fields[1] != "incomplete") {
        throw std::runtime_error("Invalid cutoff record in " + journal.string());
      }
      // The journal is committed first. Complete a CSV update interrupted by a process exit.
      if (!completed(goal.id) && rows_.at(goal.id).line != goal.cutoff_row) {
        saveRow(goal.id, {false, goal.cutoff_row});
      }
    }
    outstanding_ = std::move(goal);
  }
}

void RunStore::readResults()
{
  const auto path = dir_ / "results.csv";
  if (!fs::exists(path)) {
    replaceFile(path, [](auto & out) {out << header << '\n';});
    return;
  }
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Cannot read " + path.string());
  }
  std::string line;
  if (!std::getline(in, line) || line != header || in.eof()) {
    throw std::runtime_error("Invalid results.csv header");
  }
  uintmax_t committed = line.size() + 1;
  while (std::getline(in, line)) {
    if (in.eof()) {  // A crash interrupted the last append before its commit newline.
      in.close();
      fs::resize_file(path, committed);
      syncPath(path);
      return;
    }
    const auto fields = parseCsv(line);
    const auto found = rows_.find(fields.at(0));
    if (fields.size() != 12 || found == rows_.end() || !found->second.line.empty()) {
      throw std::runtime_error("Invalid or duplicate row in " + path.string());
    }
    found->second = {terminalStatus(fields[1]), line};
    last_attempt_ = fields[0];
    committed += line.size() + 1;
  }
  if (!in.eof() || in.bad()) {
    throw std::runtime_error("Failed reading " + path.string());
  }
}

bool RunStore::completed(const std::string & id) const
{
  return rows_.at(id).completed;
}

bool RunStore::recorded(const std::string & id) const
{
  return !rows_.at(id).line.empty();
}

std::vector<const Target *> RunStore::pendingTargets() const
{
  std::vector<const Target *> pending;
  size_t next = 0;
  for (size_t i = 0; i < dataset_.targets.size(); ++i) {
    const auto & target = dataset_.targets[i];
    if (!recorded(target.id)) {pending.push_back(&target);}
    if (target.id == last_attempt_) {next = i + 1;}
  }
  // Give all fresh targets a turn, then retry incomplete targets round robin across --limit runs.
  for (size_t i = 0; i < dataset_.targets.size(); ++i) {
    const auto & target = dataset_.targets[(next + i) % dataset_.targets.size()];
    if (recorded(target.id) && !completed(target.id)) {pending.push_back(&target);}
  }
  if (outstanding_) {
    auto old = std::find_if(pending.begin(), pending.end(),
      [&](const auto * target) {return target->id == outstanding_->id;});
    if (old != pending.end()) {std::rotate(pending.begin(), old, old + 1);}
  }
  return pending;
}

void RunStore::markOutstanding(const std::string & id, const rclcpp_action::GoalUUID & uuid)
{
  if (outstanding_) {
    if (outstanding_->id != id || outstanding_->uuid != uuid) {
      throw std::logic_error("Previous goal still outstanding: " + outstanding_->id);
    }
    if (outstanding_->cutoff_row.empty()) {return;}
  }
  if (completed(id) || uuid == rclcpp_action::GoalUUID{}) {
    throw std::logic_error("Invalid outstanding goal for " + id);
  }
  saveOutstanding({id, uuid, {}});
}

void RunStore::saveOutstanding(OutstandingGoal goal)
{
  YAML::Node journal;
  journal["target_id"] = goal.id;
  journal["uuid"] = YAML::Binary(goal.uuid.data(), goal.uuid.size());
  if (!goal.cutoff_row.empty()) {journal["incomplete_row"] = goal.cutoff_row;}
  replaceFile(dir_ / "outstanding.yaml", [&](auto & out) {out << YAML::Dump(journal) << '\n';});
  outstanding_ = std::move(goal);
}

void RunStore::noteCutoff(const std::string & reason)
{
  if (outstanding_ && outstanding_->cutoff_row.empty()) {
    PlanRecord cutoff;
    cutoff.error_msg = reason;
    record(outstanding_->id, cutoff);
  }
}

void RunStore::recoverTerminal(const PlanHandle::WrappedResult & result)
{
  if (!outstanding_ || result.goal_id != outstanding_->uuid) {
    throw std::logic_error("Recovered result UUID does not match the outstanding request");
  }
  const auto & id = outstanding_->id;
  if (outstanding_->cutoff_row.empty() && !completed(id)) {
    const auto target = std::find_if(dataset_.targets.begin(), dataset_.targets.end(),
      [&](const Target & t) {return t.id == id;});
    auto row = classifyResult(dataset_.geometry, *target, result);
    row.error_msg = "Recovered after runner restart" + (row.error_msg.empty() ? "" : ". " + row.error_msg);
    record(id, row);
  }
  clearOutstanding();
}

void RunStore::clearOutstanding()
{
  if (outstanding_) {
    if (!completed(outstanding_->id) &&
      (outstanding_->cutoff_row.empty() || rows_.at(outstanding_->id).line != outstanding_->cutoff_row))
    {
      throw std::logic_error("Cannot clear outstanding goal before recording " + outstanding_->id);
    }
    fs::remove(dir_ / "outstanding.yaml");
    syncPath(dir_);
    outstanding_.reset();
  }
}

void RunStore::record(std::string id, const PlanRecord & row)
{
  if (completed(id)) {
    throw std::logic_error("Already completed target " + id);
  }
  SavedRow replacement{terminalStatus(row.status), resultLine(id, row)};
  if (!replacement.completed && outstanding_ && outstanding_->id == id &&
    outstanding_->cutoff_row != replacement.line)
  {
    saveOutstanding({id, outstanding_->uuid, replacement.line});
  }
  saveRow(id, std::move(replacement));
}

void RunStore::saveRow(const std::string & id, SavedRow replacement)
{
  auto & saved = rows_.at(id);
  const auto path = dir_ / "results.csv";
  if (saved.line.empty()) {
    std::ofstream out;
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.open(path, std::ios::app | std::ios::binary);
    out << replacement.line << '\n';
    out.close();
    syncPath(path);
  } else {
    // On retry, readers see either the old incomplete record or its replacement.
    replaceFile(path, [&](auto & out) {
      out << header << '\n';
      for (const auto & target : dataset_.targets) {
        const auto & line = rows_.at(target.id).line;
        if (target.id != id && !line.empty()) {
          out << line << '\n';
        }
      }
      out << replacement.line << '\n';
    });
  }
  saved = std::move(replacement);
  last_attempt_ = id;
}

void RunStore::saveMetadata()
{
  replaceFile(dir_ / "run.yaml", [&](auto & out) {out << YAML::Dump(metadata_) << '\n';});
}

void RunStore::beginObservation(double timeout_s, const std::optional<std::string> & map_id)
{
  YAML::Node entry;
  entry["started_unix_ms"] = unixMillis();
  entry["timeout_s"] = timeout_s;
  entry["start_map_id"] = map_id ? YAML::Node(*map_id) : YAML::Node(YAML::NodeType::Null);
  entry["end_map_id"] = YAML::Node(YAML::NodeType::Null);
  entry["state"] = "running";
  metadata_["live_costmap_observations"].push_back(entry);
  saveMetadata();
}

void RunStore::finishObservation(const std::string & state, const std::optional<std::string> & map_id)
{
  auto entries = metadata_["live_costmap_observations"];
  auto entry = entries[entries.size() - 1];
  entry["finished_unix_ms"] = unixMillis();
  entry["end_map_id"] = map_id ? YAML::Node(*map_id) : YAML::Node(YAML::NodeType::Null);
  entry["state"] = state;
  saveMetadata();
}
}  // namespace grid_plan_tester
