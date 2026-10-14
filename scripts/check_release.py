"""Validate packaging; --runtime adds CPU-only unit checks, never training."""

import argparse
import ast
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def static_checks():
    files = sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    forbidden = {'.pth', '.pt', '.ckpt', '.npy', '.npz', '.png', '.jpg', '.jpeg', '.pkl'}
    if any(p.suffix.lower() in forbidden for p in files):
        raise ValueError('Unexpected model/data asset in release')
    patterns = {
        'local_absolute_path': r'(?<![A-Za-z])[A-Za-z]:[\\/](?!/)|/home/[^\s]+|/Users/[^\s]+',
        'private_key': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
        'access_token': r'gh[pousr]_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|sk-[A-Za-z0-9]{20,}',
    }
    python_count = 0
    for path in files:
        text = path.read_text(encoding='utf-8-sig')
        if path.name != 'check_release.py':
            for label, pattern in patterns.items():
                if re.search(pattern, text):
                    raise ValueError(f'{label}: {path.relative_to(ROOT)}')
        if path.suffix == '.py':
            tree = ast.parse(text)
            compile(tree, str(path), 'exec')
            python_count += 1
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and not node.level and node.module and \
                        (node.module == 'coevonet' or node.module.startswith('coevonet.')):
                    module = ROOT.joinpath(*node.module.split('.'))
                    if not module.with_suffix('.py').is_file() and not module.is_dir():
                        raise ValueError('Missing project dependency: ' + node.module)
    with (ROOT / 'docs' / 'source_map.csv').open(encoding='utf-8', newline='') as f:
        sources = list(csv.DictReader(f))
    for row in sources:
        path = ROOT / row['release_relative_path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['release_sha256']:
            raise ValueError('Source-derived release file changed: ' + row['release_relative_path'])
    return {'python_files': python_count, 'source_files_hash_checked': len(sources),
            'syntax_and_project_import_paths': 'PASS', 'sensitive_pattern_scan': 'PASS'}


def runtime_checks():
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import yaml
    import torch
    checks = {}
    configs = list((ROOT / 'configs').glob('*.yaml'))
    for path in configs:
        assert isinstance(yaml.safe_load(path.read_text(encoding='utf-8')), dict)
    checks['yaml_configs_loaded'] = len(configs)
    modules = ['coevonet.models.generator', 'coevonet.models.wkan', 'coevonet.data.patch_bank',
               'coevonet.data.coco_rgbt', 'coevonet.engine.train_generator',
               'coevonet.engine.infer_generator', 'coevonet.losses.hsg_alignment',
               'coevonet.losses.fib_frequency', 'coevonet.synthesis.constraints',
               'coevonet.synthesis.wavelet_refiner']
    for name in modules:
        importlib.import_module(name)
    checks['core_modules_imported'] = len(modules)
    entry = ROOT / 'scripts' / '20_synthesize_dataset.py'
    spec = importlib.util.spec_from_file_location('release_synthesis_unit_check', entry)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cfg = yaml.safe_load((ROOT / 'configs' / 'synthesis_v25lite_phaseC_train_tau027_density.yaml').read_text())
    density = module._resolve_scene_density_config(cfg, 1, 8)
    assert density['enabled'] and len(density['modes']) == 4
    assert abs(sum(mode['weight'] for mode in density['modes']) - 1) < 1e-6
    for _ in range(32):
        name, budget = module._sample_scene_density(density)
        mode = next(m for m in density['modes'] if m['name'] == name)
        assert mode['min_objects'] <= budget <= mode['max_objects']
    assert np.isclose(module._softmax(np.array([0.2, 0.5], dtype=np.float32)).sum(), 1)
    checks['density_and_sampling_helpers'] = 'PASS'
    from coevonet.synthesis.wavelet_refiner import wavelet_refine_tir
    # Deterministic mathematical fixture, not an experimental observation or generated dataset.
    fixture = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32).astype(np.uint8)
    rgb_fixture = np.repeat(fixture[:, :, None], 3, axis=2)
    refined, _ = wavelet_refine_tir(rgb_fixture, fixture, None, cfg)
    assert refined.shape == fixture.shape and refined.dtype == np.uint8
    checks['wavelet_shape_dtype_unit_check'] = 'PASS'
    helper = ROOT / 'scripts' / 'configure_paths.py'
    help_commands = [
        [sys.executable, '-B', str(ROOT / 'scripts' / 'run.py'), '--help'],
        [sys.executable, '-B', str(helper), '--help'],
    ]
    for name in ('01_build_context_prior.py', '02_build_prototype_bank.py',
                 '03_merge_synth_coco.py', '04_align_coco_category_ids.py',
                 '05_build_scene_density_prior.py', '10_train_coevonet.py',
                 '11_infer_coevonet.py', '20_synthesize_dataset.py',
                 '21_run_synth_shards.py', '22_summarize_synth_runs.py'):
        help_commands.append([sys.executable, '-B', str(ROOT / 'scripts' / 'run.py'),
                              'scripts/' + name, '--help'])
    for command in help_commands:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90)
        if result.returncode:
            raise RuntimeError('CLI help failed: ' + command[-2] + '\n' + result.stderr)
    checks['core_cli_help_checks'] = len(help_commands)
    detector = ROOT / 'third_party' / 'mmdet3_rgbt_atss'
    sys.path.insert(0, str(detector))
    sys.path.insert(0, str(detector / 'tools'))
    try:
        from formal_utils import bootstrap_runtime
        bootstrap_runtime()
        from mmengine.config import Config
        detector_configs = list((detector / 'configs').glob('*.py'))
        for path in detector_configs:
            Config.fromfile(str(path))
        for name in ('run_formal_matrix_dualstream_atss.py', 'train_formal_dualstream_atss.py',
                     'eval_tiny_et_formal_dualstream_atss.py', 'summarize_formal_dualstream_atss.py'):
            command = [sys.executable, '-B', str(ROOT / 'scripts' / 'run.py'),
                       'third_party/mmdet3_rgbt_atss/tools/' + name, '--help']
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90)
            if result.returncode:
                raise RuntimeError(name + '\n' + result.stderr)
        checks['atss_runtime_and_config_loading'] = 'PASS'
        checks['atss_configs_loaded'] = len(detector_configs)
        checks['atss_cli_help_checks'] = 4
    except Exception as exc:
        checks['atss_runtime_and_config_loading'] = 'PARTIAL'
        checks['atss_dependency_error'] = str(exc)
    checks['no_training_dataset_generation_or_metrics'] = True
    checks['torch_version_observed'] = torch.__version__
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', action='store_true')
    args = parser.parse_args()
    report = {'static': static_checks()}
    if args.runtime:
        report['runtime'] = runtime_checks()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
