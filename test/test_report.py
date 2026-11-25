import csv
import math
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest
import yaml


REPORT = Path(__file__).resolve().parents[1] / 'scripts' / 'report'
MAP_ID = '4c9ced947e279734'
DATASET_ID = 'b9b68b9bcf7ab837'
FREE_ONLY_ID = 'c9ac0e8506f4548b'
RESULT_HEADER = (
    'id,status,action_status,error_code,error_msg,wall_ms,planning_ms,poses,'
    'length_m,end_x,end_y,end_error_m'
).split(',')


@pytest.fixture
def dataset(tmp_path):
    manifest = {
        'schema_version': 1, 'dataset_id': DATASET_ID, 'map_id': MAP_ID, 'frame': 'map',
        'width': 64, 'height': 48, 'resolution': 0.5,
        'origin_x': 10., 'origin_y': 20., 'origin_yaw': math.pi / 2,
        # Grid cell (16, 8) in the rotated map.
        'start': dict(x=5.75, y=28.25, z=0., qx=0., qy=0., qz=0., qw=1.),
    }
    (tmp_path / 'manifest.yaml').write_text(yaml.safe_dump(manifest))
    costs = Image.new('L', (64, 48))
    costs.paste(254, (0, 0, 4, 4))
    for x in (40, 50, 60):
        costs.putpixel((x, 8), 128)
    costs.save(tmp_path / 'costmap.png')
    with (tmp_path / 'targets.csv').open('w') as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow('id kind group mx my cost x y yaw'.split())
        for name, kind, mx, my in (
            ('f1', 'free', 10, 35), ('f2', 'free', 20, 35),
            ('f3', 'free', 30, 35), ('i1', 'inflation', 40, 8),
            ('i2', 'inflation', 50, 8), ('i3', 'inflation', 60, 8),
        ):
            writer.writerow([name, kind, 1, mx, my, 0 if kind == 'free' else 128,
                             10. - (my + 0.5) * 0.5, 20. + (mx + 0.5) * 0.5, 0.])
    return tmp_path


def make_run(dataset, name, attempts, dataset_id=DATASET_ID, observations=None):
    directory = dataset / 'runs' / name
    directory.mkdir(parents=True)
    info = {'dataset_id': dataset_id, 'map_id': MAP_ID, 'planner_id': 'lattice'}
    if observations is not None:
        info['live_costmap_observations'] = observations
    (directory / 'run.yaml').write_text(yaml.safe_dump(info))
    with (directory / 'results.csv').open('w') as output:
        writer = csv.writer(output)
        writer.writerow(RESULT_HEADER)
        for goal, status, code in attempts:
            writer.writerow([goal, status, 4, code, 'note, with comma', 10, 9, 2, 1, 2, 3, 0])
    return directory


def report(*args):
    return subprocess.run([sys.executable, str(REPORT), *map(str, args)],
                          capture_output=True, text=True)


def summary_for(stdout, group):
    lines = stdout.splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith(group + ' '))
    return ' '.join(lines[index:index + 3])


def test_latest_committed_attempts_and_rates(dataset):
    run = make_run(dataset, 'candidate', [
        ('f1', 'failed', 207), ('f1', 'hit', 0), ('f2', 'adjusted', 0),
        ('f3', 'failed', 204), ('i1', 'failed', 207), ('i2', 'incomplete', 207),
        ('unknown', 'hit', 0),
    ], observations=[dict(start_map_id=MAP_ID, end_map_id='live-b'),
                     dict(start_map_id=None, end_map_id=None)])
    with (run / 'results.csv').open('ab') as output:
        output.write(b'f1,"unterminated\n')
        output.write(b'i3,bad_status,0,0,,1,1,2,3,4,5,6\n')
        output.write(b'i1,hit,4,0,,1,1,2,3,4,5,')  # All columns, uncommitted.
    result = report(dataset, 'candidate')
    assert result.returncode == 0, result.stderr
    all_line = summary_for(result.stdout, 'all')
    free = summary_for(result.stdout, 'free')
    inflation = summary_for(result.stdout, 'inflation')
    assert 'hit=1 adjusted=1 failed=2 incomplete=1 pending=1' in all_line
    assert 'planner_timeouts=1' in all_line
    assert 'hit/completed=1/3 (33.3%)' in free
    assert 'path/completed=2/3 (66.7%)' in free
    assert 'completed/total=1/3 (33.3%)' in inflation
    assert 'hit/total=1/6 (16.7%)' in all_line
    assert 'matching=1 different=1 missing=2' in result.stdout
    assert 'unknown' in result.stderr and 'uncommitted' in result.stderr
    with Image.open(run / 'coverage.png') as coverage:
        # The known obstacle starts at grid y=0 and must appear below the free space.
        assert min(coverage.getpixel((0, 0))) > 180
        first_dark = next(y for y in range(coverage.height)
                          if max(coverage.getpixel((0, y))) < 80)
        scale = first_dark / 44

        def cell(mx, my, dx=0, dy=0):
            return coverage.getpixel((int((mx + 0.5) * scale) + dx,
                                      int((47.5 - my) * scale) + dy))

        assert cell(10, 35) == cell(10, 12)  # Hollow hit marker.
        assert cell(10, 35, 4, 0) == (0, 142, 102)
        assert cell(30, 35, 3, 3) == cell(30, 35, -3, 3) == (208, 58, 45)
        assert cell(30, 35, 3, 0) == cell(10, 12)  # Space between the cross arms.
        assert max(cell(10, 12)) - min(cell(10, 12)) < 10  # Mirror is empty.
        assert max(cell(16, 8)) - min(cell(16, 8)) > 65  # Rotated start pose.
        assert len({cell(30, 35), cell(40, 8), cell(50, 8)}) == 3  # Failure / 207 / incomplete.


def test_comparison_uses_paired_finished_cases_and_rejects_other_dataset(dataset):
    make_run(dataset, 'base', [
        ('f1', 'hit', 0), ('f2', 'adjusted', 0), ('f3', 'failed', 204),
        ('i1', 'hit', 0), ('i2', 'incomplete', 0),
    ])
    make_run(dataset, 'next', [
        ('f1', 'adjusted', 0), ('f2', 'failed', 204), ('f3', 'hit', 0),
        ('i1', 'incomplete', 0), ('i2', 'failed', 204),
    ])
    result = report(dataset, 'next', '--compare', 'base')
    assert result.returncode == 0, result.stderr
    assert 'base -> next' in result.stdout
    assert 'comparable=3/6' in result.stdout
    assert 'regressions=2' in result.stdout
    assert 'hit -> adjusted (1)  f1' in result.stdout
    assert 'adjusted -> failed (1)  f2' in result.stdout
    assert 'failed -> hit (1)  f3' in result.stdout
    make_run(dataset, 'other', [], 'map-b')
    mismatch = report(dataset, 'next', '--compare', 'other')
    assert mismatch.returncode != 0
    assert 'dataset_id' in mismatch.stderr and 'other' in mismatch.stderr


def test_new_run_and_empty_group_have_no_success_rate(dataset):
    targets = dataset / 'targets.csv'
    targets.write_text(''.join(line for line in targets.read_text().splitlines(True)
                               if ',inflation,' not in line))
    manifest = dataset / 'manifest.yaml'
    info = yaml.safe_load(manifest.read_text())
    info['dataset_id'] = FREE_ONLY_ID
    manifest.write_text(yaml.safe_dump(info))
    run = make_run(dataset, 'new', [], FREE_ONLY_ID)
    (run / 'results.csv').unlink()
    result = report(dataset, 'new')
    assert result.returncode == 0, result.stderr
    free = summary_for(result.stdout, 'free')
    inflation = summary_for(result.stdout, 'inflation')
    assert 'pending=3' in free and 'hit/completed=0/0 (n/a)' in free
    assert 'completed/total=0/0 (n/a)' in inflation


@pytest.mark.parametrize(('changed', 'signature'), [
    ('costmap.png', 'map_id'), ('targets.csv', 'dataset_id'), ('manifest.yaml', 'dataset_id'),
])
def test_changed_frozen_inputs_cannot_produce_coverage(dataset, changed, signature):
    run = make_run(dataset, 'before', [('f1', 'hit', 0)])
    path = dataset / changed
    if changed == 'costmap.png':
        with Image.open(path) as image:
            swapped = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        swapped.save(path)
    elif changed == 'targets.csv':
        path.write_bytes(path.read_bytes().replace(b'f1,', b'g1,'))
    else:
        info = yaml.safe_load(path.read_text())
        info['start']['x'] += 0.25
        path.write_text(yaml.safe_dump(info))
    result = report(dataset, 'before')
    assert result.returncode != 0 and signature in result.stderr
    assert not (run / 'coverage.png').exists()
