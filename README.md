# grid_plan_tester

ROS 2 Jazzy 批量路径规划测试工具。从全局代价地图选取代表性目标，冻结采样时的机器人起点，逐个调用 Nav2 的 `ComputePathToPose`。整个测试只请求路径，机器人无需实际行驶。

**目前只能用它检查路径规划是否成功、返回终点是否命中目标栅格，以及规划耗时。它不能评价路径质量。** 即使所有目标都命中，也不能据此判定路径平滑、绕行合理、障碍净空充足或控制器能稳定跟踪。

## 快速开始：采样、只测膨胀层、生成报告

先启动仿真和导航，定位就绪后在测试终端执行。以下命令均在工作区根目录运行，并在同一个 Bash 终端中复用变量。

```bash
cd ~/s100_docker
source /opt/ros/jazzy/setup.bash
source install/setup.bash
source src/webots_simulation/setup_simulation.bash

batch="log/hall_regular_step5_$(date +%Y%m%d_%H%M%S)"
ros2 run grid_plan_tester sample "$batch" --mode regular --spacing 0.5 \
  --min-cost 5 --max-cost 240 --inflation-cost-step 5
ros2 run grid_plan_tester run "${batch}_inflation" navfn
ros2 run grid_plan_tester report "${batch}_inflation" navfn
```

- **空闲目标（`free`）**来自代价为 0 的栅格。上述配置用 0.5 m 分块，在整块全为空闲的区域合并为 1 m 或 2 m 分块。
- **膨胀层目标（`inflation`）**来自指定的代价区间，包含上下界，允许范围为 1–252。上述配置在每个障碍组内按 5–9、10–14 等代价段取点，每个有候选栅格的段取一个代表点。省略 `--inflation-cost-step` 时每个有候选点的障碍组只取一个点。

## 分开测试两类目标

`run` 按数据集中的目标清单运行。混合数据集同时包含两类目标，`--limit` 只限制尝试次数。采样完成后会自动生成混合目录 `$batch` 和只含膨胀层目标的同级目录 `${batch}_inflation`，并打印运行命令。两份数据使用同一张采样地图和同一个起点。两个输出目录都必须是新目录，如遇重名请换一个批次名。规划请求使用执行当时的实时 costmap。

### 在新终端查看已有膨胀层报告

新终端加载上述环境后，可直接使用采样时打印的膨胀层目录。以下是已有 hall 批次的示例，查看其他批次时替换路径。报告命令的目录和运行名应与当次 `run` 一致。

```bash
ros2 run grid_plan_tester report ~/s100_docker/log/hall_regular_step5_20261125_154904_inflation navfn
```

报告在终端显示汇总，并将覆盖图写入该目录的 `runs/navfn/coverage.png`。逐目标耗时见同目录下的 `runs/navfn/results.csv`。

### 从混合采样补建子集（只测空闲目标或使用旧数据时）

对已有混合采样，先将 `batch` 设为其目录。以下命令复用包内报告程序的数据校验逻辑，重新计算子集 ID，保留原始数据和结果。

```bash
python3 - "$batch" <<'PY'
import csv
import runpy
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

src = Path(sys.argv[1])
report = runpy.run_path("src/grid_plan_tester/scripts/report")
manifest, targets, _ = report["read_dataset"](src)
lines = (src / "targets.csv").read_bytes().splitlines(keepends=True)
pose = " ".join(format(manifest["start"][k], ".17g") for k in ("x", "y", "z", "qx", "qy", "qz", "qw"))
for kind in ("free", "inflation"):
    dst = src.with_name(src.name + "_" + kind)
    if dst.exists():
        print(f"目录已存在，跳过：{dst}")
        continue
    selected = [line for line in lines[1:] if next(csv.reader([line.decode()]))[1] == kind]
    payload = lines[0] + b"".join(selected)
    meta = yaml.safe_load((src / "manifest.yaml").read_text())
    meta["map_id"] = manifest["map_id"]
    meta["dataset_id"] = report["fingerprint"](f'{meta["map_id"]}\n{pose}\n', payload)
    with tempfile.TemporaryDirectory(dir=src.parent) as tmp:
        staged = Path(tmp) / kind
        staged.mkdir()
        shutil.copyfile(src / "costmap.png", staged / "costmap.png")
        (staged / "targets.csv").write_bytes(payload)
        (staged / "manifest.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
        _, subset, _ = report["read_dataset"](staged)
        assert subset == {k: v for k, v in targets.items() if v.kind == kind}
        staged.rename(dst)
    print(f"{dst}: {len(selected)} {kind}")
PY
```

### 只测空闲栅格目标

```bash
ros2 run grid_plan_tester run "${batch}_free" navfn
ros2 run grid_plan_tester report "${batch}_free" navfn
```

`navfn` 是本次运行的记录名。默认请求 planner ID `GridBased`，其他配置可通过 `--planner ID` 指定。同一数据集和记录名支持续跑；重新测整组时换一个记录名。

## 看结果和耗时

每组结果在对应子集的 `runs/navfn/` 下。`results.csv` 保留逐目标记录，`report` 输出汇总并生成 `coverage.png`。加 `--rviz` 可发布目标标记，在 RViz 中添加话题 `/grid_plan_tester/results`（MarkerArray）。

| 字段或状态 | 如何理解 |
|---|---|
| `hit` | 返回有效路径，终点落在请求目标栅格内。 |
| `adjusted` | 返回有效路径，终点落在其他栅格。可结合 `end_error_m` 查看位置偏差。 |
| `failed` | 规划失败，结合 `error_code` 和 `error_msg` 定位原因。 |
| `planning_ms` | 服务端返回的规划耗时。启用集成平滑时包含平滑处理。 |
| `wall_ms` | 客户端观察到的请求耗时，含通信与等待。两个耗时字段不能相加。 |

`poses` 和 `length_m` 是路径点数和长度的描述性记录。覆盖图展示的是请求目标的位置和规划结果，颜色表示状态，不代表路径质量。
