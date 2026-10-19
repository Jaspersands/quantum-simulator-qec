# Test basic CLI functionality to full coverage but attempting to not duplicate API tests where possible.
# Ported from qecsim 1.0b9 (qecsim/qecsim; David K. Tuckett, BSD 3-Clause; see THIRD_PARTY_NOTICES.md), modified for
# stabilizer_qec: qecsim is stabilizer_qec.qecsim.
import pytest
from click.testing import CliRunner

from stabilizer_qec.qecsim.cli import cli


# base command

@pytest.mark.parametrize('arguments', [
    pytest.param([], marks=pytest.mark.xfail(reason="click 8.2 exits with 2 for a group given no command (qecsim was written for 7)")),
    ['--help'],
    ['--version'],
])
def test_cli(arguments):
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, arguments)
        assert result.exit_code == 0


@pytest.mark.parametrize('arguments', [
    ['--blah'],  # unknown option
])
def test_cli_invalid_arguments(arguments):
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, arguments)
        assert result.exit_code != 0


# run sub-command
@pytest.mark.parametrize('arguments', [
    ['--help'],
    ['five_qubit', 'generic.depolarizing', 'generic.naive', '0.1'],  # default single run
    ['five_qubit', 'generic.depolarizing', 'generic.naive', '0.1', '0.2'],  # multiple error_probabilities
    ['-f2', 'steane', 'generic.phase_flip', 'generic.naive', '0.1'],  # max_failures
    ['-r9', 'five_qubit', 'generic.depolarizing', 'generic.naive', '0.1'],  # max_runs
    ['-f2', '-r9', 'toric(3,3)', 'generic.bit_flip', 'toric.mwpm', '0.1'],  # max_failures, max_runs
    ['-s5', 'toric(3,3)', 'generic.bit_flip', 'toric.mwpm', '0.1'],  # random_seed
    # no maximum random_seed
    ['-s174975946453286449385846588109544149806', 'toric(3,3)', 'generic.bit_flip', 'toric.mwpm', '0.1'],
    ['-otmp_data.json', 'toric(3,3)', 'generic.bit_flip', 'toric.mwpm', '0.1'],  # output_file
])
def test_cli_run(arguments):
    runner = CliRunner()
    with runner.isolated_filesystem():
        arguments = ['run'] + arguments
        result = runner.invoke(cli, arguments)
        assert result.exit_code == 0


@pytest.mark.parametrize('arguments', [
    ['--blah'],  # unknown option
    [],  # missing code
    ['five_qubit'],  # missing error_model
    ['five_qubit', 'generic.depolarizing'],  # missing decoder
    ['five_qubit', 'generic.depolarizing', 'generic.naive'],  # missing error_probability
    ['blah', 'generic.depolarizing', 'generic.naive', '0.1'],  # unknown code
    ['five_qubit', 'blah', 'generic.naive', '0.1'],  # unknown error_model
    ['five_qubit', 'generic.depolarizing', 'blah', '0.1'],  # unknown decoder
    ['five_qubit', 'generic.depolarizing', 'generic.naive(', '0.1'],  # invalid constructor
    ['five_qubit', 'generic.depolarizing', 'generic.naive(f o o)', '0.1'],  # invalid constructor args
    ['-r1.1', 'five_qubit', 'generic.depolarizing', 'generic.naive', '0.1'],  # invalid max_runs
    ['-f1.1', 'five_qubit', 'generic.depolarizing', 'generic.naive', '0.1'],  # invalid max_failures
    ['-s-1', 'five_qubit', 'generic.depolarizing', 'generic.naive', '0.1'],  # invalid random_seed min
    ['five_qubit', 'generic.depolarizing', 'generic.naive', '1.1'],  # invalid error_probability
    ['five_qubit(-1)', 'generic.depolarizing', 'generic.naive', '0.1'],  # bad code args
    ['five_qubit', 'generic.depolarizing(-1)', 'generic.naive', '0.1'],  # bad error_model args
    ['five_qubit', 'generic.depolarizing', 'generic.naive(-1)', '0.1'],  # bad decoder args
])
def test_cli_run_invalid_arguments(arguments):
    runner = CliRunner()
    with runner.isolated_filesystem():
        arguments = ['run'] + arguments
        result = runner.invoke(cli, arguments)
        assert result.exit_code != 0


def test_cli_run_existing_output_file():
    runner = CliRunner()
    with runner.isolated_filesystem():
        # run to existing file
        empty_file_name = 'tmp_empty.json'
        with open(empty_file_name, 'w') as f:
            result = runner.invoke(
                cli, ['run', '-o', empty_file_name, 'toric(3,3)', 'generic.bit_flip', 'toric.mwpm', '0.1'])
            # check file is still empty
            f.seek(0, 2)  # seek to end of file
            assert f.tell() == 0  # end of file is at position zero
            assert result.exit_code != 0


@pytest.mark.parametrize('arguments', [
    # each code with each valid decoder
    ['color666(5)', 'generic.depolarizing', 'color666.mps(8)', '0.15'],
    ['five_qubit', 'generic.depolarizing', 'generic.naive', '0.1'],
    ['planar(5,5)', 'generic.depolarizing', 'planar.cmwpm', '0.15'],
    ['planar(5,5)', 'generic.depolarizing', 'planar.mps(6)', '0.15'],
    ['planar(5,5)', 'generic.depolarizing', 'planar.mwpm', '0.15'],
    ['planar(5,5)', 'generic.depolarizing', 'planar.rmps(6)', '0.15'],
    ['planar(4,5)', 'generic.depolarizing', 'planar.y', '0.15'],
    ['rotated_planar(7,7)', 'generic.depolarizing', 'rotated_planar.mps(8)', '0.45'],
    ['rotated_planar(7,7)', 'generic.depolarizing', 'rotated_planar.rmps(8)', '0.45'],
    ['rotated_planar(7,7)', 'generic.biased_depolarizing(100)', 'rotated_planar.smwpm', '0.45'],
    ['rotated_toric(6,6)', 'generic.biased_depolarizing(100)', 'rotated_toric.smwpm', '0.45'],
    ['steane', 'generic.depolarizing', 'generic.naive', '0.1'],
    ['toric(5,5)', 'generic.depolarizing', 'toric.mwpm', '0.1'],
    # each generic noise model
    ['planar(5,5)', 'generic.biased_depolarizing(10)', 'planar.mps(6)', '0.15'],
    ['planar(5,5)', 'generic.biased_y_x(10)', 'planar.mps(6)', '0.15'],
    ['planar(5,5)', 'generic.bit_flip', 'planar.mps(6)', '0.15'],
    ['planar(5,5)', 'generic.bit_phase_flip', 'planar.mps(6)', '0.15'],
    ['planar(5,5)', 'generic.center_slice((0.2,0.8,0),0.5)', 'planar.mps(6)', '0.15'],
    ['planar(5,5)', 'generic.depolarizing', 'planar.mps(6)', '0.15'],
    ['planar(5,5)', 'generic.phase_flip', 'planar.mps(6)', '0.15'],
])
def test_cli_run_all_models(arguments):
    runner = CliRunner()
    with runner.isolated_filesystem():
        arguments = ['run'] + arguments
        result = runner.invoke(cli, arguments)
        assert result.exit_code == 0


# run-ftp sub-command
@pytest.mark.parametrize('arguments', [
    ['--help'],
    ['rotated_planar(3,3)', '3', 'generic.bit_phase_flip', 'rotated_planar.smwpm', '0.05'],  # default single run
    # multiple error_probabilities
    ['rotated_planar(3,3)', '3', 'generic.bit_phase_flip', 'rotated_planar.smwpm', '0.05', '0.06'],
    ['-f2', 'rotated_planar(3,3)', '3', 'generic.bit_phase_flip', 'rotated_planar.smwpm', '0.05'],  # max_failures
    ['-r9', 'rotated_planar(3,3)', '3', 'generic.bit_phase_flip', 'rotated_planar.smwpm', '0.05'],  # max_runs
    # max_failures, max_runs
    ['-f2', '-r9', 'rotated_planar(3,3)', '3', 'generic.bit_phase_flip', 'rotated_planar.smwpm', '0.05'],
    ['-s5', 'rotated_planar(3,3)', '3', 'generic.bit_phase_flip', 'rotated_planar.smwpm', '0.05'],  # random_seed
    ['-s174975946453286449385846588109544149806', 'rotated_planar(3,3)', '3', 'generic.bit_phase_flip',
     'rotated_planar.smwpm', '0.05'],  # no maximum random_seed
    # measurement_error_probability
    ['-m0.01', 'rotated_planar(3,3)', '3', 'generic.bit_phase_flip', 'rotated_planar.smwpm', '0.05'],
    # output_file
    ['-otmp_data.json', 'rotated_planar(3,3)', '3', 'generic.bit_phase_flip', 'rotated_planar.smwpm', '0.05'],
])
def test_cli_run_ftp(arguments):
    runner = CliRunner()
    with runner.isolated_filesystem():
        arguments = ['run-ftp'] + arguments
        result = runner.invoke(cli, arguments)
        assert result.exit_code == 0


@pytest.mark.parametrize('arguments', [
    ['--blah'],  # unknown option
    [],  # missing code
    ['rotated_planar(3,3)'],  # missing time_steps
    ['rotated_planar(3,3)', '3'],  # missing error_model
    ['rotated_planar(3,3)', '3', 'generic.bit_phase_flip'],  # missing decoder
    ['rotated_planar(3,3)', '3', 'generic.bit_phase_flip', 'rotated_planar.smwpm'],  # missing error_probability
    ['blah', '3', 'generic.bit_phase_flip', 'rotated_planar.smwpm', '0.05'],  # unknown code
    ['rotated_planar(3,3)', 'a', 'generic.bit_phase_flip', 'rotated_planar.smwpm', '0.05'],  # invalid time_steps
    ['rotated_planar(3,3)', '0', 'generic.bit_phase_flip', 'rotated_planar.smwpm', '0.05'],  # invalid time_steps
    ['rotated_planar(3,3)', '0.0', 'generic.bit_phase_flip', 'rotated_planar.smwpm', '0.05'],  # invalid time_steps
    ['rotated_planar(3,3)', '3', 'blah', 'rotated_planar.smwpm', '0.05'],  # unknown error_model
    ['rotated_planar(3,3)', '3', 'generic.bit_phase_flip', 'blah', '0.05'],  # unknown decoder
    ['rotated_planar(3,3)', '3', 'generic.bit_phase_flip', 'rotated_planar.smwpm(', '0.05'],  # invalid constructor
    # invalid constructor args
    ['rotated_planar(3,3)', '3', 'generic.bit_phase_flip', 'rotated_planar.smwpm(f o o)', '0.05'],
    ['-r1.1', 'rotated_planar(3,3)', '3', 'generic.bit_phase_flip', 'rotated_planar.smwpm', '0.05'],  # invalid max_runs
    # invalid max_failures
    ['-f1.1', 'rotated_planar(3,3)', '3', 'generic.bit_phase_flip', 'rotated_planar.smwpm', '0.05'],
    # invalid random_seed min
    ['-s-1', 'rotated_planar(3,3)', '3', 'generic.bit_phase_flip', 'rotated_planar.smwpm', '0.05'],
    ['rotated_planar(3,3)', '3', 'generic.bit_phase_flip', 'rotated_planar.smwpm', '1.1'],  # invalid error_probability
    ['rotated_planar(-1)', '3', 'generic.bit_phase_flip', 'rotated_planar.smwpm', '0.05'],  # bad code args
    ['rotated_planar(3,3)', '3', 'generic.bit_phase_flip(-1)', 'rotated_planar.smwpm', '0.05'],  # bad error_model args
    ['rotated_planar(3,3)', '3', 'generic.bit_phase_flip', 'rotated_planar.smwpm(-1)', '0.05'],  # bad decoder args
])
def test_cli_run_ftp_invalid_arguments(arguments):
    runner = CliRunner()
    with runner.isolated_filesystem():
        arguments = ['run-ftp'] + arguments
        result = runner.invoke(cli, arguments)
        assert result.exit_code != 0


def test_cli_run_ftp_existing_output_file():
    runner = CliRunner()
    with runner.isolated_filesystem():
        # run to existing file
        empty_file_name = 'tmp_empty.json'
        with open(empty_file_name, 'w') as f:
            result = runner.invoke(
                cli, ['run-ftp', '-o', empty_file_name, 'rotated_planar(3,3)', '3', 'generic.bit_phase_flip',
                      'rotated_planar.smwpm', '0.05'])
            # check file is still empty
            f.seek(0, 2)  # seek to end of file
            assert f.tell() == 0  # end of file is at position zero
            assert result.exit_code != 0


@pytest.mark.parametrize('arguments', [
    # each code with each valid decoder
    ['rotated_planar(3,3)', '3', 'generic.bit_phase_flip', 'rotated_planar.smwpm', '0.05'],
    ['rotated_toric(2,2)', '2', 'generic.bit_phase_flip', 'rotated_toric.smwpm', '0.05'],
])
def test_cli_run_ftp_all_models(arguments):
    runner = CliRunner()
    with runner.isolated_filesystem():
        arguments = ['run-ftp'] + arguments
        result = runner.invoke(cli, arguments)
        assert result.exit_code == 0


# merge sub-command
def test_cli_merge_help():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ['merge', '--help'])
        assert result.exit_code == 0


def test_cli_merge_files():
    runner = CliRunner()
    with runner.isolated_filesystem():
        data_file_names = ['tmp_data1.json', 'tmp_data2.json']
        # run for each data file
        for f in data_file_names:
            result = runner.invoke(
                cli, ['run', '-o', f, 'toric(3,3)', 'generic.bit_flip', 'toric.mwpm', '0.1'])
            assert result.exit_code == 0
        # merge data files
        result = runner.invoke(cli, ['merge'] + data_file_names)
        assert result.exit_code == 0


def test_cli_merge_files_output_file():
    runner = CliRunner()
    with runner.isolated_filesystem():
        data_file_names = ['tmp_data1.json', 'tmp_data2.json']
        # run for each data file
        for f in data_file_names:
            result = runner.invoke(
                cli, ['run', '-o', f, 'toric(3,3)', 'generic.bit_flip', 'toric.mwpm', '0.1'])
            assert result.exit_code == 0
        # merge data files
        result = runner.invoke(cli, ['merge', '-o', 'tmp_merge.json'] + data_file_names)
        assert result.exit_code == 0


def test_cli_merge_missing_file():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ['merge', 'blah.json'])
        assert result.exit_code != 0


def test_cli_merge_invalid_file():
    runner = CliRunner()
    with runner.isolated_filesystem():
        empty_file_name = 'tmp_empty.json'
        with open(empty_file_name, 'a'):
            result = runner.invoke(cli, ['merge', empty_file_name])
            assert result.exit_code != 0


def test_cli_merge_files_existing_output_file():
    runner = CliRunner()
    with runner.isolated_filesystem():
        data_file_name = 'tmp_data.json'
        # run to data file
        result = runner.invoke(
            cli, ['run', '-o', data_file_name, 'toric(3,3)', 'generic.bit_flip', 'toric.mwpm', '0.1'])
        assert result.exit_code == 0
        # merge to existing file
        empty_file_name = 'tmp_empty.json'
        with open(empty_file_name, 'w') as f:
            result = runner.invoke(cli, ['merge', '-o', empty_file_name, data_file_name])
            # check file is still empty
            f.seek(0, 2)  # seek to end of file
            assert f.tell() == 0  # end of file is at position zero
            assert result.exit_code != 0


# unknown sub-command
def test_cli_unknown_subcommand():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ['blah'])
        assert result.exit_code != 0


# (stabilizer_qec: extensions, as qecsim's README and qecsimext have them: an installed distribution's
# entry points in the qecsim.cli.run.* and qecsim.cli.run_ftp.* groups add codes, error models and decoders.)
_EXTENSION = '''
from stabilizer_qec.qecsim.model import cli_description
from stabilizer_qec.qecsim.models.basic import BasicCode
from stabilizer_qec.qecsim.models.generic import BitFlipErrorModel, NaiveDecoder


@cli_description('3-qubit repetition (an extension)')
class ThreeQubitCode(BasicCode):
    def __init__(self):
        super().__init__(('ZZI', 'IZZ'), ('XXX',), ('IIZ',), (3, 1, 1), '3-qubit')


@cli_description('3-qubit bit-flip (an extension)')
class ThreeQubitBitFlipErrorModel(BitFlipErrorModel):
    @property
    def label(self):
        return '3-qubit bit-flip'


@cli_description('3-qubit lookup (an extension)')
class ThreeQubitLookupDecoder(NaiveDecoder):
    @property
    def label(self):
        return '3-qubit lookup'
'''


def test_cli_run_extension_entry_points(tmp_path):
    import json
    import os
    import subprocess
    import sys

    (tmp_path / 'myqecext.py').write_text(_EXTENSION)
    for dist, entry_points in [
        ('myqecext-0.1', '[qecsim.cli.run.codes]\next_3qubit = myqecext:ThreeQubitCode\n'
                         '[qecsim.cli.run.error_models]\next_3qubit.bit_flip = myqecext:ThreeQubitBitFlipErrorModel\n'
                         '[qecsim.cli.run.decoders]\next_3qubit.lookup = myqecext:ThreeQubitLookupDecoder\n'
                         'ext_broken = no_such_module:Decoder\n'
                         '[qecsim.cli.run_ftp.codes]\next_rotated_planar = stabilizer_qec.qecsim.models.rotatedplanar:RotatedPlanarCode\n'),
        # The real qecsim's own entry points name its classes: the port keeps its own.
        ('qecsim-1.0b9', '[qecsim.cli.run.codes]\nfive_qubit = qecsim.models.basic:FiveQubitCode\n'),
    ]:
        info = tmp_path / (dist + '.dist-info')
        info.mkdir()
        name, version = dist.split('-')
        (info / 'METADATA').write_text('Metadata-Version: 2.1\nName: {}\nVersion: {}\n'.format(name, version))
        (info / 'entry_points.txt').write_text(entry_points)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path)] + [p for p in [os.environ.get('PYTHONPATH')] if p]))

    def run(*args):
        r = subprocess.run([sys.executable, '-m', 'stabilizer_qec.qecsim', *args], capture_output=True, text=True,
                           env=env, cwd=str(tmp_path))
        assert r.returncode == 0, r.stderr
        return r

    r = run('run', '-r20', '-s3', 'ext_3qubit', 'ext_3qubit.bit_flip', 'ext_3qubit.lookup', '0.3')
    data = json.loads(r.stdout)
    assert [(d['code'], d['error_model'], d['decoder'], d['n_run']) for d in data] == [
        ('3-qubit', '3-qubit bit-flip', '3-qubit lookup', 20)]
    assert 'SKIPPING ENTRY POINT qecsim.cli.run.decoders: ext_broken' in r.stderr and 'five_qubit' not in r.stderr
    page = run('run', '--help').stdout
    assert '3-qubit repetition (an extension)' in page and 'five_qubit' in page and 'ext_broken' not in page
    assert json.loads(run('run', '-r2', 'five_qubit', 'generic.bit_flip', 'generic.naive', '0.1').stdout)[0]['code'] == '5-qubit'
    assert 'ext_rotated_planar' in run('run-ftp', '--help').stdout
