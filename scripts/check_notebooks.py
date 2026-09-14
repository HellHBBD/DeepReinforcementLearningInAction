"""Execute notebook cells unchanged, in isolated processes and scratch directories."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / 'scripts/notebooks.json').read_text())


def worker(path, result, limit, seed):
    import random
    import nbformat
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from IPython.core.interactiveshell import InteractiveShell

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    os.chdir(path.parent)
    sys.path[:0] = [str(path.parent), str(path.parent.parent)]
    shell = InteractiveShell.instance()
    # Only display pacing is omitted; training code and budgets are unchanged.
    time.sleep = lambda *_args, **_kwargs: None
    plt.show = lambda *_args, **_kwargs: plt.close('all')
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    records = []
    try:
        for index, cell in enumerate(notebook.cells):
            if index > limit:
                break
            if cell.cell_type != 'code' or not cell.source.strip():
                continue
            print(f'CELL {index}', flush=True)
            start = time.monotonic()
            outcome = shell.run_cell(cell.source, store_history=False)
            error = outcome.error_before_exec or outcome.error_in_exec
            # IPython shell commands otherwise hide a failed worker subprocess.
            exit_code = shell.user_ns.pop('_exit_code', 0)
            if error:
                raise RuntimeError(f'Cell {index} failed: {error}') from error
            if exit_code:
                raise RuntimeError(f'Cell {index} shell command exited {exit_code}')
            metrics = {}
            for name in ('losses', 'losses1', 'losses2', 'score', 'rewards',
                         'episode_successes', 'episode_distances', 'battle_parameter_change'):
                value = shell.user_ns.get(name)
                if value is None:
                    continue
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                try:
                    values = np.asarray(value, dtype=float)
                except (ValueError, TypeError):
                    continue
                if values.size:
                    if not np.isfinite(values).all():
                        raise AssertionError(f'Cell {index}: non-finite {name}')
                    metrics[name] = {'count': int(values.size), 'mean': float(values.mean()),
                                     'last100_mean': float(values.ravel()[-100:].mean())}
            records.append({'cell': index, 'seconds': time.monotonic() - start, 'metrics': metrics})
            plt.close('all')
        if limit > len(notebook.cells):
            check_outcomes(path, shell.user_ns)
        result.write_text(json.dumps({'success': True, 'cells': records}, indent=2))
    except BaseException as error:
        result.write_text(json.dumps({'success': False, 'error': str(error), 'cells': records}, indent=2))
        raise


def check_outcomes(path, ns):
    """Broad learning alarms, not assertions that every seed matches a book figure."""
    import numpy as np
    import torch
    chapter = path.parent.name
    if chapter == 'Chapter 7':
        # Fresh held-out episodes use the notebook's own distributional policy.
        import gymnasium as gym
        crossings = []
        for seed in range(5):
            env = gym.make('ALE/Freeway-v5', obs_type='ram', frameskip=(2, 5),
                           repeat_action_probability=.25)
            try:
                state, _ = env.reset(seed=seed)
                total = 0
                for _ in range(1300):
                    with torch.no_grad():
                        pred = ns['dist_dqn'](ns['preproc_state'](state), ns['theta'], aspace=3)
                        action = int(ns['get_action'](pred.unsqueeze(0), ns['support']).item())
                    state, reward, terminated, truncated, _ = env.step(action)
                    total += reward
                    if terminated or truncated:
                        state, _ = env.reset()
                crossings.append(total)
            finally:
                env.close()
        print('Held-out Freeway crossings:', crossings, flush=True)
        assert min(crossings) >= 4, f'Freeway below book crossing criterion: {crossings}'
    if chapter == 'Chapter 9':
        assert min(ns['battle_parameter_change']) > 0, 'Battle parameters did not change'
    if chapter == 'Chapter 10':
        successes = ns['episode_successes']
        assert len(successes) >= 100, 'Too few completed DoorKey episodes'
        assert np.mean(successes[-100:]) >= .8, 'DoorKey learning regressed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--notebook', help='One exact path from scripts/notebooks.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/notebooks')
    parser.add_argument('--seed', type=int, default=46)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--limit', type=int, default=10**6, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker.resolve(), args.output.resolve(), args.limit, args.seed)
        return
    import nbformat
    selected = [row for row in MANIFEST if not args.notebook or row['path'] == args.notebook]
    if not selected:
        parser.error('Notebook is not in the manifest')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for row in selected:
        source = ROOT / row['path']
        notebook = nbformat.read(source, as_version=4)
        nbformat.validate(notebook)
        boundary = notebook.cells[row['smoke_through']].source
        assert hashlib.sha256(boundary.encode()).hexdigest() == row['boundary_sha256'], (
            f"Smoke boundary changed in {row['path']}; review scripts/notebooks.json")
        tag = source.parent.name.replace(' ', '_') + '__' + source.stem.replace(' ', '_')
        result = args.output / f'{tag}.json'
        result.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(prefix='drl-ci-') as directory:
            scratch = Path(directory)
            # Copy code only, never downloaded data or artifacts. Each notebook owns its helper writes.
            for file in ROOT.rglob('*'):
                relative = file.relative_to(ROOT)
                if any(part.startswith('.') or part in ('data', 'artifacts', '__pycache__') for part in relative.parts):
                    continue
                if file.is_file() and file.suffix in ('.py', '.ipynb'):
                    target = scratch / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file, target)
            (ROOT / 'data').mkdir(exist_ok=True)
            (scratch / 'data').symlink_to(ROOT / 'data', target_is_directory=True)
            limit = row['smoke_through'] if args.mode == 'smoke' else 10**6
            command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                       str(scratch / row['path']), '--output', str(result),
                       '--limit', str(limit), '--seed', str(args.seed)]
            env = dict(os.environ, MPLBACKEND='Agg', SDL_VIDEODRIVER='dummy',
                       SDL_AUDIODRIVER='dummy', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            with (args.output / f'{tag}.log').open('w') as log:
                try:
                    run = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                    passed = run.wait(timeout=600 if args.mode == 'smoke' else 10800) == 0
                except subprocess.TimeoutExpired:
                    os.killpg(run.pid, signal.SIGKILL)
                    run.wait()
                    passed = False
                    log.write('\nNotebook exceeded execution timeout\n')
            expected = [i for i, c in enumerate(notebook.cells)
                        if i <= limit and c.cell_type == 'code' and c.source.strip()]
            report = json.loads(result.read_text()) if result.exists() else {}
            passed = passed and report.get('success', False) and [c['cell'] for c in report.get('cells', [])] == expected
            summary.append({'notebook': row['path'], 'mode': args.mode, 'seed': args.seed,
                            'success': passed, 'code_cells': len(expected),
                            'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
            print(f"{'PASS' if passed else 'FAIL'} {row['path']} ({len(expected)} cells)", flush=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    if not all(row['success'] for row in summary):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
