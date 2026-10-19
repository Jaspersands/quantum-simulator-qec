# modified for stabilizer_qec: its ldpc and PyMatching are this package's, and panqec is stabilizer_qec.panqec.
import os

import numpy as np
import pytest
from click.testing import CliRunner

from stabilizer_qec.panqec.cli import cli, read_bias_ratios, read_range_input


@pytest.fixture
def runner():
    """Click CliRunner with isolated file system."""
    _runner = CliRunner()
    with _runner.isolated_filesystem():
        yield _runner
    assert hasattr(_runner, 'invoke')


@pytest.mark.parametrize('arguments', [
    [],
    ['--help'],
    ['--version'],
])
def test_cli_basic(arguments, runner):
    result = runner.invoke(cli, arguments)
    assert result.exit_code == 0


class TestLS:

    def test_ls_all(self, runner):
        result = runner.invoke(cli, ['ls'])
        assert result.exit_code == 0
        assert 'Codes:' in result.output
        assert 'Error Models (Noise):' in result.output
        assert 'Decoders:' in result.output

    def test_ls_codes(self, runner):
        result = runner.invoke(cli, ['ls', 'codes'])
        assert result.exit_code == 0
        assert 'Codes:' in result.output
        assert 'Error Models (Noise):' not in result.output
        assert 'Decoders:' not in result.output

    def test_ls_error_models(self, runner):
        result = runner.invoke(cli, ['ls', 'error_models'])
        assert result.exit_code == 0
        assert 'Codes:' not in result.output
        assert 'Error Models (Noise):' in result.output
        assert 'Decoders:' not in result.output

    def test_ls_decoders(self, runner):
        result = runner.invoke(cli, ['ls', 'decoders'])
        assert result.exit_code == 0
        assert 'Codes:' not in result.output
        assert 'Error Models (Noise):' not in result.output
        assert 'Decoders:' in result.output


def test_read_bias_ratios():
    expected_bias_ratios = [0.5, 1, 3, 10, 30, 100, np.inf]
    eta_string = '0.5,1,3,10,30,100,inf'
    bias_ratios = read_bias_ratios(eta_string)
    assert len(bias_ratios) == len(expected_bias_ratios)
    for eta, expected_eta in zip(bias_ratios, expected_bias_ratios):
        assert eta == expected_eta
        assert isinstance(eta, type(expected_eta))


@pytest.mark.parametrize('spec,expected_values', [
    ('0:0.6:0.005', np.arange(0, 0.605, 0.005).tolist()),
    ('1,2,3', [1.0, 2.0, 3.0]),
    ('13.21', [13.21]),
    ('1e-2', [0.01]),
])
def test_read_range_input(spec, expected_values):
    values = read_range_input(spec)
    assert len(values) == len(expected_values)
    for value, expected_value in zip(values, expected_values):
        assert value == expected_value
        assert isinstance(value, type(expected_value))


def test_generate_cluster_script(runner):
    # (stabilizer_qec: the script calls `stabilizer-qec panqec ...`, which a cluster with only this
    # package installed has, where PanQEC's calls `panqec ...`; the rest is PanQEC's.)
    os.makedirs('my-experiment/inputs')
    with open('slurm_header.sbatch', 'w') as f:
        f.write('#!/bin/bash\n#SBATCH -t ${TIME}\n#SBATCH --job-name=${NAME}\n#SBATCH --array=1-${N_NODES}\n')
    result = runner.invoke(cli, [
        'generate-cluster-script', 'slurm_header.sbatch', '--output-file', 'my-experiment/run.sbatch',
        '--data-dir', 'my-experiment', '--cluster', 'slurm', '--n-nodes', '8', '--wall-time', '10:00:00',
        '--memory', '20GB', '--trials', '10000', '--n-cores', '5', '--delete-existing'
    ])
    assert result.exit_code == 0, result.output
    with open('my-experiment/run.sbatch') as f:
        text = f.read()
    data_dir = os.path.abspath('my-experiment')
    assert text == (
        '#!/bin/bash\n#SBATCH -t 10:00:00\n#SBATCH --job-name=my-experiment\n#SBATCH --array=1-8\n\n\n'
        'stabilizer-qec panqec monitor-usage my-experiment/logs/usage_${SLURM_JOB_ID}_$SLURM_ARRAY_TASK_ID.txt &\n\n'
        f'stabilizer-qec panqec run-parallel -d {data_dir} -n 8 -j $SLURM_ARRAY_TASK_ID -c 5 -t 10000 '
        '--delete-existing\n\ndate'
    )
