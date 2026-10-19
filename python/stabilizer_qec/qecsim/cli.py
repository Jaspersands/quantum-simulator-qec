# Ported from qecsim 1.0b9 (qecsim/qecsim; David K. Tuckett, BSD 3-Clause; see THIRD_PARTY_NOTICES.md), modified for
# stabilizer_qec: qecsim is stabilizer_qec.qecsim, and its command line `stabilizer-qec qecsim`.
"""
This module contains the qecsim command line interface (CLI).

Components are integrated into the CLI via entries in the ``[options.entry-points]`` section of ``setup.cfg``. The
format of entries is ``<short_name> = <module_path>:<class_name>``. Codes, error models and decoders appear under the
keys ``qecsim.cli.run.codes``, ``qecsim.cli.run.error_models`` and ``qecsim.cli.run.decoders``, respectively.
Fault-tolerant compatible codes, error models and decoders appear under the keys ``qecsim.cli.run_ftp.codes``,
``qecsim.cli.run_ftp.error_models`` and ``qecsim.cli.run_ftp.decoders``, respectively.

For example, the 5-qubit code appears in ``setup.cfg`` as follows:

    .. code-block:: text

        [options.entry_points]
        qecsim.cli.run.codes =
            five_qubit = qecsim.models.basic:FiveQubitCode


Optionally, one-line descriptions for CLI help messages can be provided by decorating implementation classes with
:func:`qecsim.model.cli_description`. For example, see :class:`qecsim.models.basic.FiveQubitCode`.
"""
import ast
import inspect
import json
import logging
import re

import click
import importlib

from stabilizer_qec import qecsim as qecsim
from stabilizer_qec.qecsim import app
from stabilizer_qec.qecsim import util
from stabilizer_qec.qecsim.model import ATTR_CLI_DESCRIPTION


# qecsim's entry points (setup.cfg [options.entry_points]), as a table: this package does not
# register them with the installer. Installed distributions' entry points in the same groups
# (qecsim's extensions, e.g. qecsimext's) are read too: see _constructors.
_ENTRY_POINTS = {
    'qecsim.cli.run.codes': [
        ('color666', 'stabilizer_qec.qecsim.models.color', 'Color666Code'),
        ('five_qubit', 'stabilizer_qec.qecsim.models.basic', 'FiveQubitCode'),
        ('planar', 'stabilizer_qec.qecsim.models.planar', 'PlanarCode'),
        ('rotated_planar', 'stabilizer_qec.qecsim.models.rotatedplanar', 'RotatedPlanarCode'),
        ('rotated_toric', 'stabilizer_qec.qecsim.models.rotatedtoric', 'RotatedToricCode'),
        ('steane', 'stabilizer_qec.qecsim.models.basic', 'SteaneCode'),
        ('toric', 'stabilizer_qec.qecsim.models.toric', 'ToricCode'),
    ],
    'qecsim.cli.run.error_models': [
        ('generic.biased_depolarizing', 'stabilizer_qec.qecsim.models.generic', 'BiasedDepolarizingErrorModel'),
        ('generic.biased_y_x', 'stabilizer_qec.qecsim.models.generic', 'BiasedYXErrorModel'),
        ('generic.bit_flip', 'stabilizer_qec.qecsim.models.generic', 'BitFlipErrorModel'),
        ('generic.bit_phase_flip', 'stabilizer_qec.qecsim.models.generic', 'BitPhaseFlipErrorModel'),
        ('generic.center_slice', 'stabilizer_qec.qecsim.models.generic', 'CenterSliceErrorModel'),
        ('generic.depolarizing', 'stabilizer_qec.qecsim.models.generic', 'DepolarizingErrorModel'),
        ('generic.file', 'stabilizer_qec.qecsim.models.generic', 'FileErrorModel'),
        ('generic.phase_flip', 'stabilizer_qec.qecsim.models.generic', 'PhaseFlipErrorModel'),
    ],
    'qecsim.cli.run.decoders': [
        ('color666.mps', 'stabilizer_qec.qecsim.models.color', 'Color666MPSDecoder'),
        ('generic.naive', 'stabilizer_qec.qecsim.models.generic', 'NaiveDecoder'),
        ('planar.cmwpm', 'stabilizer_qec.qecsim.models.planar', 'PlanarCMWPMDecoder'),
        ('planar.mps', 'stabilizer_qec.qecsim.models.planar', 'PlanarMPSDecoder'),
        ('planar.mwpm', 'stabilizer_qec.qecsim.models.planar', 'PlanarMWPMDecoder'),
        ('planar.rmps', 'stabilizer_qec.qecsim.models.planar', 'PlanarRMPSDecoder'),
        ('planar.y', 'stabilizer_qec.qecsim.models.planar', 'PlanarYDecoder'),
        ('rotated_planar.mps', 'stabilizer_qec.qecsim.models.rotatedplanar', 'RotatedPlanarMPSDecoder'),
        ('rotated_planar.rmps', 'stabilizer_qec.qecsim.models.rotatedplanar', 'RotatedPlanarRMPSDecoder'),
        ('rotated_planar.smwpm', 'stabilizer_qec.qecsim.models.rotatedplanar', 'RotatedPlanarSMWPMDecoder'),
        ('rotated_toric.smwpm', 'stabilizer_qec.qecsim.models.rotatedtoric', 'RotatedToricSMWPMDecoder'),
        ('toric.mwpm', 'stabilizer_qec.qecsim.models.toric', 'ToricMWPMDecoder'),
    ],
    'qecsim.cli.run_ftp.codes': [
        ('rotated_planar', 'stabilizer_qec.qecsim.models.rotatedplanar', 'RotatedPlanarCode'),
        ('rotated_toric', 'stabilizer_qec.qecsim.models.rotatedtoric', 'RotatedToricCode'),
    ],
    'qecsim.cli.run_ftp.error_models': [
        ('generic.biased_depolarizing', 'stabilizer_qec.qecsim.models.generic', 'BiasedDepolarizingErrorModel'),
        ('generic.biased_y_x', 'stabilizer_qec.qecsim.models.generic', 'BiasedYXErrorModel'),
        ('generic.bit_flip', 'stabilizer_qec.qecsim.models.generic', 'BitFlipErrorModel'),
        ('generic.bit_phase_flip', 'stabilizer_qec.qecsim.models.generic', 'BitPhaseFlipErrorModel'),
        ('generic.center_slice', 'stabilizer_qec.qecsim.models.generic', 'CenterSliceErrorModel'),
        ('generic.depolarizing', 'stabilizer_qec.qecsim.models.generic', 'DepolarizingErrorModel'),
        ('generic.phase_flip', 'stabilizer_qec.qecsim.models.generic', 'PhaseFlipErrorModel'),
    ],
    'qecsim.cli.run_ftp.decoders': [
        ('rotated_planar.smwpm', 'stabilizer_qec.qecsim.models.rotatedplanar', 'RotatedPlanarSMWPMDecoder'),
        ('rotated_toric.smwpm', 'stabilizer_qec.qecsim.models.rotatedtoric', 'RotatedToricSMWPMDecoder'),
    ],
}

logger = logging.getLogger(__name__)


def _installed_entry_points(group):
    """Installed distributions' entry points in `group` (as qecsim's pkg_resources.iter_entry_points), but for
    the real qecsim's own, which name its classes where the port's table names this package's."""
    import importlib.metadata as md

    eps = md.entry_points()
    eps = eps.select(group=group) if hasattr(eps, 'select') else eps.get(group, [])  # (Python 3.9: a dict)
    seen, out = set(), []
    for ep in eps:
        if ep.value.split(':')[0].split('.')[0] == 'qecsim' or (ep.name, ep.value) in seen:
            continue
        seen.add((ep.name, ep.value))
        out.append(ep)
    return out


def _constructors(entry_point_id):
    """Each name in the entry point group `entry_point_id` (e.g. qecsim.cli.run_ftp.codes) and its class: the
    port's own and installed extensions' (whose name wins over the port's), sorted by name as qecsim's are. An
    extension that does not load is skipped with a warning (qecsim would fail to start)."""
    table = {name: (module, attr) for name, module, attr in _ENTRY_POINTS.get(entry_point_id, [])}
    constructors = {name: getattr(importlib.import_module(module), attr) for name, (module, attr) in table.items()}
    for ep in sorted(_installed_entry_points(entry_point_id), key=lambda ep: ep.name):
        try:
            constructors[ep.name] = ep.load()
        except Exception as ex:  # noqa: BLE001 - a broken extension should not stop the others
            logger.warning('SKIPPING ENTRY POINT {}: {} = {} ({!r}).'.format(entry_point_id, ep.name, ep.value, ex))
    return dict(sorted(constructors.items()))


class _ConstructorParamType(click.ParamType):
    """
    Constructor param type that accepts parameters in the format ``name(<args>)``.
    """
    name = 'constructor'

    def __init__(self, constructors):
        """
        Initialise new constructor parameter type.

        :param constructors: Map of constructor names to constructor functions.
        :type constructors: dict of str to function
        """
        self._constructors = constructors

    def get_metavar(self, param):
        """See ``click.ParamType.get_metavar``"""
        return '[{}]'.format('|'.join(sorted(self._constructors.keys())))

    def get_missing_message(self, param, ctx=None):  # (stabilizer_qec: click 8.2 passes ctx too)
        """See ``click.ParamType.get_missing_message``"""
        return '(choose from {})'.format(', '.join(sorted(self._constructors.keys())))

    def convert(self, value, param, ctx):
        """
        Convert value to model instance.

        If the value is correctly formatted as ``name`` or ``name(<args>)`` then:

        * constructor is resolved using the constructors map.
        * arguments is resolved to a tuple using a literal evaluation of args.
        * instance is constructed using constructor(*arguments).

        See ``click.ParamType.convert`` for more details.

        :param value: Parameter value.
        :type value: str
        :param param: Parameter.
        :type param: click.Parameter
        :param ctx: Context.
        :type ctx: click.Context
        :return: Model instance
        :rtype: object
        :raises BadParameter: if the value cannot be parsed or does not correspond to valid constructor or arguments.
        """
        # constructor regex match
        constructor_match = re.fullmatch(r'''
            # match 'toric(3,3)' as {'constructor_name': 'toric', 'constructor_args': '3,3'}
            (?P<constructor_name>[\w.]+)  # capture constructor_name, e.g. 'toric'
            (?:\(\s*                      # skip opening parenthesis and leading whitespace
                (?P<constructor_args>.*?) # capture constructor_args, e.g. '3,3'
            ,?\s*\))?                     # skip trailing comma, trailing whitespace and closing parenthesis
        ''', value, re.VERBOSE)

        # check format
        if constructor_match is None:
            self.fail('{} (format as name(<args>))'.format(value), param, ctx)

        # convert constructor_name to constructor
        constructor_name = constructor_match.group('constructor_name')
        if constructor_name in self._constructors.keys():
            # select constructor from map
            constructor = self._constructors[constructor_name]
        else:
            self.fail('{} (choose from {})'.format(value, ', '.join(sorted(self._constructors.keys()))), param, ctx)

        # convert constructor_args to arguments tuple
        constructor_args = constructor_match.group('constructor_args')
        if constructor_args:
            try:
                # eval args as literal (add comma to force tuple)
                arguments = ast.literal_eval(constructor_args + ',')
            except Exception as ex:
                self.fail('{} (failed to parse arguments "{}")'.format(value, ex), param, ctx)
        else:
            # no args -> empty tuple
            arguments = tuple()

        # instantiate model
        try:
            return constructor(*arguments)
        except Exception as ex:
            self.fail('{} (failed to construct "{}")'.format(value, ex), param, ctx)

    def __repr__(self):
        return '{}({!r})'.format(type(self).__name__, self._constructors)


# custom argument decorators
def _model_argument(model_type):
    """
    Model argument function decorator.

    Notes:
    * This decorator is applied to a run command to add an argument of the given model type, i.e. code, error_model, or
      decoder.
    * The possible argument values and corresponding model constructors are loaded from setuptools entry-points, under
      the key `qecsim.cli.<run-command>.<model-type>s`, e.g. `qecsim.cli.run_ftp.codes`, with value
      `["<model-name> = <model-package>:<model-class>", ...]`, e.g.` ["steane = qecsim.models.basic:SteaneCode", ...]`.
    * The doc-string of the run command is updated as follows:

        * The placeholder `#<MODEL-TYPE>_PARAMETERS#`, e.g. `#CODE_PARAMETERS#`, is replaced by a definition list
          consisting of `<model-name>` and `<cli-description>`, as specified by the model class decorator
          :func:`qecsim.model.cli_description`, e.g. `planar` and `Planar (rows INT >= 2, cols INT >= 2)`.

    :param model_type: The model type, i.e code, error_model, or decoder.
    :type model_type: str
    :return: Model argument function decorator.
    :rtype: function
    """

    def _decorator(func):
        # extract name and class from entry-point, e.g. {'five_qubit': FiveQubitCode, ...}
        entry_point_id = 'qecsim.cli.{}.{}s'.format(func.__name__, model_type)  # e.g. qecsim.cli.run_ftp.codes
        constructors = _constructors(entry_point_id)
        # add argument decorator
        func = click.argument(model_type, type=_ConstructorParamType(constructors), metavar=model_type.upper())(func)
        # extract name and cli_help, e.g. [('five_qubit', '5-qubit'), ...]
        model_definition_list = [(name, getattr(cls, ATTR_CLI_DESCRIPTION, '')) for name, cls in constructors.items()]
        # update __doc__
        formatter = click.HelpFormatter()
        formatter.indent()
        if model_definition_list:
            formatter.write_dl(model_definition_list)
        model_doc_placeholder = '#{}_PARAMETERS#'.format(model_type.upper())  # e.g. #CODE_PARAMETERS#
        func.__doc__ = inspect.getdoc(func).replace(model_doc_placeholder, formatter.getvalue())
        return func

    return _decorator


# custom parameter validators
def _validate_error_probability(ctx, param, value):
    if not (0 <= value <= 1):
        raise click.BadParameter('{} is not in [0.0, 1.0]'.format(value), ctx, param)
    return value


def _validate_error_probabilities(ctx, param, value):
    for v in value:
        _validate_error_probability(ctx, param, v)
    return value


def _validate_measurement_error_probability(ctx, param, value):
    if not (value is None or (0 <= value <= 1)):
        raise click.BadParameter('{} is not in [0.0, 1.0]'.format(value), ctx, param)
    return value


@click.group()
@click.version_option(version=qecsim.__version__, prog_name='qecsim')
def cli():
    """
    qecsim - quantum error correction simulator using stabilizer codes.

    See stabilizer-qec qecsim COMMAND --help for command-specific help.
    """
    util.init_logging()


@cli.command()
@_model_argument('code')
@_model_argument('error_model')
@_model_argument('decoder')
@click.argument('error_probabilities', required=True, nargs=-1, type=float, metavar='ERROR_PROBABILITY...',
                callback=_validate_error_probabilities)
@click.option('--max-failures', '-f', type=click.IntRange(min=1), metavar='INT',
              help='Maximum number of failures for each probability.')
@click.option('--max-runs', '-r', type=click.IntRange(min=1), metavar='INT',
              help='Maximum number of runs for each probability.  [default: 1 if max-failures unspecified]')
@click.option('--output', '-o', default='-', type=click.Path(allow_dash=True), metavar='FILENAME',
              help='Output file. (Writes to log if file exists).')
@click.option('--random-seed', '-s', type=click.IntRange(min=0), metavar='INT',
              help='Random seed for qubit error generation. (Re-applied for each probability).')
def run(code, error_model, decoder, error_probabilities, max_failures, max_runs, output, random_seed):
    """
    Simulate quantum error correction.

    Arguments:

    \b
     CODE                  Stabilizer code in format name(<args>)
    #CODE_PARAMETERS#

    \b
     ERROR_MODEL           Error model in format name(<args>)
    #ERROR_MODEL_PARAMETERS#

    \b
     DECODER               Decoder in format name(<args>)
    #DECODER_PARAMETERS#

    \b
     ERROR_PROBABILITY...  One or more probabilities as FLOAT in [0.0, 1.0]

    Examples:

     stabilizer-qec qecsim run -r10 "five_qubit" "generic.depolarizing" "generic.naive" 0.1

     stabilizer-qec qecsim run -f5 -r50 -s13 "steane" "generic.phase_flip" "generic.naive" 0.1

     stabilizer-qec qecsim run -r20 "planar(7,7)" "generic.bit_flip" "planar.mps(6)" 0.101 0.102 0.103

     stabilizer-qec qecsim run -r10 "color666(7)" "generic.bit_flip" "color666.mps(16)" 0.09 0.10

     stabilizer-qec qecsim run -o"data.json" -f9 "toric(3,3)" "generic.bit_flip" "toric.mwpm" 0.1
    """
    # INPUT
    code.validate()

    logger.info('RUN STARTING: code={}, error_model={}, decoder={}, error_probabilities={}, max_failures={}, '
                'max_runs={}, random_seed={}.'
                .format(code, error_model, decoder, error_probabilities, max_failures, max_runs, random_seed))

    # RUN
    data = []
    for error_probability in error_probabilities:
        runs_data = app.run(code, error_model, decoder, error_probability,
                            max_runs=max_runs, max_failures=max_failures, random_seed=random_seed)
        data.append(runs_data)

    logger.info('RUN COMPLETE: data={}'.format(data))

    # OUTPUT
    _write_data(output, data)


@cli.command()
@_model_argument('code')
@click.argument('time_steps', type=click.IntRange(min=1), metavar='TIME_STEPS')
@_model_argument('error_model')
@_model_argument('decoder')
@click.argument('error_probabilities', required=True, nargs=-1, type=float, metavar='ERROR_PROBABILITY...',
                callback=_validate_error_probabilities)
@click.option('--max-failures', '-f', type=click.IntRange(min=1), metavar='INT',
              help='Maximum number of failures for each probability.')
@click.option('--max-runs', '-r', type=click.IntRange(min=1), metavar='INT',
              help='Maximum number of runs for each probability. [default: 1 if max_failures unspecified]')
@click.option('--measurement-error-probability', '-m', type=float, default=None,
              callback=_validate_measurement_error_probability,
              help='Measurement error probability [default: 0.0 if TIME_STEPS == 1 else ERROR_PROBABILITY].')
@click.option('--output', '-o', default='-', type=click.Path(allow_dash=True), metavar='FILENAME',
              help='Output file. (Writes to log if file exists).')
@click.option('--random-seed', '-s', type=click.IntRange(min=0), metavar='INT',
              help='Random seed for qubit error generation. (Re-applied for each probability).')
def run_ftp(code, time_steps, error_model, decoder, error_probabilities, max_failures, max_runs,
            measurement_error_probability, output, random_seed):
    """
    Simulate fault-tolerant (time-periodic) quantum error correction.

    Arguments:

    \b
     CODE                  Stabilizer code in format name(<args>)
    #CODE_PARAMETERS#

    \b
     TIME_STEPS            Number of time steps as INT >= 1

    \b
     ERROR_MODEL           Error model in format name(<args>)
    #ERROR_MODEL_PARAMETERS#

    \b
     DECODER               Decoder in format name(<args>)
    #DECODER_PARAMETERS#

    \b
     ERROR_PROBABILITY...  One or more probabilities as FLOAT in [0.0, 1.0]

    Examples:

     stabilizer-qec qecsim run-ftp -r5 "rotated_planar(13,13)" 13 "generic.bit_phase_flip" "rotated_planar.smwpm" 0.1 0.2

     stabilizer-qec qecsim run-ftp -r5 -m0.05 "rotated_toric(6,6)" 4 "generic.bit_phase_flip" "rotated_toric.smwpm" 0.1

     stabilizer-qec qecsim run-ftp -r5 -o"data.json" "rotated_planar(7,7)" 7 "generic.depolarizing" "rotated_planar.smwpm" 0.1
    """
    # INPUT
    code.validate()

    logger.info('RUN STARTING: code={}, time_steps={}, error_model={}, decoder={}, error_probabilities={}, '
                'max_failures={}, max_runs={}, measurement_error_probability={}, random_seed={}.'
                .format(code, time_steps, error_model, decoder, error_probabilities, max_failures, max_runs,
                        measurement_error_probability, random_seed))

    # RUN
    data = []
    for error_probability in error_probabilities:
        runs_data = app.run_ftp(code, time_steps, error_model, decoder, error_probability,
                                measurement_error_probability=measurement_error_probability,
                                max_runs=max_runs, max_failures=max_failures, random_seed=random_seed)
        data.append(runs_data)

    logger.info('RUN COMPLETE: data={}'.format(data))

    # OUTPUT
    _write_data(output, data)


@cli.command()
@click.argument('data_file', required=True, nargs=-1, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option('--output', '-o', default='-', type=click.Path(allow_dash=True), metavar='FILENAME',
              help='Output file. (Writes to log if file exists).')
def merge(data_file, output):
    """
    Merge simulation data files.

    Arguments:

    \b
     DATA_FILE...          One or more data files

    Examples:

     stabilizer-qec qecsim merge "data1.json" "data2.json" "data3.json"

     stabilizer-qec qecsim merge -o"merged.json" data*.json
    """
    # INPUT
    input_data = []

    # extract data from input files
    for input_file in data_file:
        try:
            with open(input_file, 'r') as f:
                input_data.append(json.load(f))
        except ValueError as ex:
            raise click.ClickException('{} (failed to parse JSON data "{}")'.format(input_file, ex))

    # MERGE
    data = app.merge(*input_data)

    # OUTPUT
    _write_data(output, data)


def _write_data(output, data):
    """
    Write data in JSON format (sorted keys) to the given output.

    Note: If the data cannot be written to the given output, for example if the file already exists, then the data is
    written to stderr and an exception is raised.

    :param output: Output file path or '-' for stdout.
    :type output: str
    :param data: Data (convertible to JSON).
    :type data: list of dict
    :raises ClickException: if the data cannot be written to the given path.
    """
    if output == '-':
        # write to stdout
        click.echo(json.dumps(data, sort_keys=True))
    else:
        try:
            # attempt to save to output filename (mode='x' -> fail if file exists)
            with open(output, 'x') as f:
                json.dump(data, f, sort_keys=True)
        except IOError as ex:
            logger.error('recovered data:\n{}'.format(json.dumps(data, sort_keys=True)))
            raise click.ClickException('{} (failed to open output file "{}")'.format(output, ex))
