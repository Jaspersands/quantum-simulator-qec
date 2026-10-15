// clap's help for MWPF 0.2.12's command line, as its `--help` and `-h` print it (captured from MWPF's
// wheels; the command line is parsed in src/mwpf/cli.rs).

pub const ROOT_LONG: &str = r###"Minimum-Weight Parity Factor Algorithm for Quantum Error Correction Decoding

Usage: mwpf <COMMAND>

Commands:
  benchmark      benchmark the speed (and also correctness if enabled)
  matrix-speed   benchmark the matrix speed
  decoder-speed  decoder speed
  custom-test    custom test command, for porting failed python test
  custom-test2   custom test command, for comparing with ported failed python test
  test           built-in tests
  help           Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
"###;

pub const ROOT_SHORT: &str = ROOT_LONG;

pub const BENCHMARK_LONG: &str = r###"benchmark the speed (and also correctness if enabled)

Usage: mwpf benchmark [OPTIONS] <D> <P>

Arguments:
  <D>
          code distance

  <P>
          physical error rate: the probability of each edge to

Options:
  -e, --pe <PE>
          rounds of noisy measurement, valid only when multiple rounds
          
          [default: 0]

  -n, --noisy-measurements <NOISY_MEASUREMENTS>
          rounds of noisy measurement, valid only when multiple rounds
          
          [default: 0]

  -c, --code-type <CODE_TYPE>
          example code type
          
          [default: code-capacity-tailored-code]

          Possible values:
          - code-capacity-repetition-code: quantum repetition code with perfect stabilizer measurement
          - code-capacity-planar-code:     planar code with perfect stabilizer measurement, only one type of stabilizer's decoding graph (thus normal graph)
          - code-capacity-color-code:      color code with perfect stabilizer measurement
          - code-capacity-tailored-code:   tailored surface code, which is essentially rotated planar code with depolarizing noise model
          - error-pattern-reader:          read from error pattern file, generated using option `--solver-type error-pattern-logger`
          - qec-playground-code:           code constructed by QEC-Playground, pass configurations using `--code-config`

      --code-config <CODE_CONFIG>
          the configuration of the code builder
          
          [default: {}]

      --enable-visualizer
          logging to the default visualizer file at visualize/data/visualizer.json

      --visualizer-json-filepath <VISUALIZER_JSON_FILEPATH>
          generate json from the visualizer; if not provided, we will use 256MB max memory to store the JSON in memory, and move it to a temporary file (which is deleted when the program exit) if the size of the JSON exceeds 256MB

      --visualizer-html-filepath <VISUALIZER_HTML_FILEPATH>
          generate html from the visualizer

      --print-syndrome-pattern
          print syndrome patterns

      --print-error-pattern
          print error patterns

      --verifier <VERIFIER>
          the method to verify the correctness of the decoding result
          
          [default: actual-error]

          Possible values:
          - none:                disable verifier
          - fusion-serial:       use the serial version of fusion blossom library to verify
          - actual-error:        if the actual error has smaller weight than the lower bound of the weight range then fail
          - strict-actual-error: if the actual error has smaller weight than the solved subgraph then fail

  -r, --total-rounds <TOTAL_ROUNDS>
          the number of iterations to run
          
          [default: 1000]

  -p, --solver-type <SOLVER_TYPE>
          select the combination of primal and dual module
          
          [default: union-find]

          Possible values:
          - union-find:           the solver from Union-Find decoder
          - single-hair:          the single-hair solver
          - joint-single-hair:    joint single-hair solver
          - error-pattern-logger: log error into a file for later fetch

      --solver-config <SOLVER_CONFIG>
          the configuration of primal and dual module
          
          [default: {}]

      --pb-message <PB_MESSAGE>
          message on the progress bar
          
          [default: ]

      --use-deterministic-seed
          use deterministic seed for debugging purpose (round number is the seed)

      --benchmark-profiler-output <BENCHMARK_PROFILER_OUTPUT>
          the benchmark profile output file path

      --starting-iteration <STARTING_ITERATION>
          skip some iterations, useful when debugging
          
          [default: 0]

      --apply-deterministic-seed <APPLY_DETERMINISTIC_SEED>
          apply a deterministic seed for debugging purposes

      --single-seed <SINGLE_SEED>
          only execute a single seed for debugging purposes

      --use-bp
          to use bp or not

      --bp-application-ratio <BP_APPLICATION_RATIO>
          

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
"###;

pub const BENCHMARK_SHORT: &str = r###"benchmark the speed (and also correctness if enabled)

Usage: mwpf benchmark [OPTIONS] <D> <P>

Arguments:
  <D>  code distance
  <P>  physical error rate: the probability of each edge to

Options:
  -e, --pe <PE>
          rounds of noisy measurement, valid only when multiple rounds [default: 0]
  -n, --noisy-measurements <NOISY_MEASUREMENTS>
          rounds of noisy measurement, valid only when multiple rounds [default: 0]
  -c, --code-type <CODE_TYPE>
          example code type [default: code-capacity-tailored-code] [possible values: code-capacity-repetition-code, code-capacity-planar-code, code-capacity-color-code, code-capacity-tailored-code, error-pattern-reader, qec-playground-code]
      --code-config <CODE_CONFIG>
          the configuration of the code builder [default: {}]
      --enable-visualizer
          logging to the default visualizer file at visualize/data/visualizer.json
      --visualizer-json-filepath <VISUALIZER_JSON_FILEPATH>
          generate json from the visualizer; if not provided, we will use 256MB max memory to store the JSON in memory, and move it to a temporary file (which is deleted when the program exit) if the size of the JSON exceeds 256MB
      --visualizer-html-filepath <VISUALIZER_HTML_FILEPATH>
          generate html from the visualizer
      --print-syndrome-pattern
          print syndrome patterns
      --print-error-pattern
          print error patterns
      --verifier <VERIFIER>
          the method to verify the correctness of the decoding result [default: actual-error] [possible values: none, fusion-serial, actual-error, strict-actual-error]
  -r, --total-rounds <TOTAL_ROUNDS>
          the number of iterations to run [default: 1000]
  -p, --solver-type <SOLVER_TYPE>
          select the combination of primal and dual module [default: union-find] [possible values: union-find, single-hair, joint-single-hair, error-pattern-logger]
      --solver-config <SOLVER_CONFIG>
          the configuration of primal and dual module [default: {}]
      --pb-message <PB_MESSAGE>
          message on the progress bar [default: ]
      --use-deterministic-seed
          use deterministic seed for debugging purpose (round number is the seed)
      --benchmark-profiler-output <BENCHMARK_PROFILER_OUTPUT>
          the benchmark profile output file path
      --starting-iteration <STARTING_ITERATION>
          skip some iterations, useful when debugging [default: 0]
      --apply-deterministic-seed <APPLY_DETERMINISTIC_SEED>
          apply a deterministic seed for debugging purposes
      --single-seed <SINGLE_SEED>
          only execute a single seed for debugging purposes
      --use-bp
          to use bp or not
      --bp-application-ratio <BP_APPLICATION_RATIO>
          
  -h, --help
          Print help (see more with '--help')
  -V, --version
          Print version
"###;

pub const MATRIX_SPEED_LONG: &str = r###"benchmark the matrix speed

Usage: mwpf matrix-speed [OPTIONS]

Options:
  -c, --matrix-type <MATRIX_TYPE>
          [default: echelon-tail-tight] [possible values: echelon-tail-tight, echelon-tight, echelon]
      --width <WIDTH>
          [default: 50]
      --height <HEIGHT>
          [default: 50]
      --one-density <ONE_DENSITY>
          [default: 0.1]
  -r, --total-rounds <TOTAL_ROUNDS>
          [default: 100000]
      --deterministic-seed <DETERMINISTIC_SEED>
          
  -h, --help
          Print help
  -V, --version
          Print version
"###;

pub const MATRIX_SPEED_SHORT: &str = MATRIX_SPEED_LONG;

pub const DECODER_SPEED_LONG: &str = r###"decoder speed

Usage: mwpf decoder-speed [OPTIONS] --file-path <FILE_PATH>

Options:
  -f, --file-path <FILE_PATH>
          

  -p, --solver-type <SOLVER_TYPE>
          select the combination of primal and dual module
          
          [default: joint-single-hair]

          Possible values:
          - union-find:           the solver from Union-Find decoder
          - single-hair:          the single-hair solver
          - joint-single-hair:    joint single-hair solver
          - error-pattern-logger: log error into a file for later fetch

      --solver-config <SOLVER_CONFIG>
          the configuration of primal and dual module
          
          [default: {}]

      --use-bp
          to use bp or not

      --bp-application-ratio <BP_APPLICATION_RATIO>
          

      --bp-max-iter <BP_MAX_ITER>
          

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
"###;

pub const DECODER_SPEED_SHORT: &str = r###"decoder speed

Usage: mwpf decoder-speed [OPTIONS] --file-path <FILE_PATH>

Options:
  -f, --file-path <FILE_PATH>
          
  -p, --solver-type <SOLVER_TYPE>
          select the combination of primal and dual module [default: joint-single-hair] [possible values: union-find, single-hair, joint-single-hair, error-pattern-logger]
      --solver-config <SOLVER_CONFIG>
          the configuration of primal and dual module [default: {}]
      --use-bp
          to use bp or not
      --bp-application-ratio <BP_APPLICATION_RATIO>
          
      --bp-max-iter <BP_MAX_ITER>
          
  -h, --help
          Print help (see more with '--help')
  -V, --version
          Print version
"###;

pub const CUSTOM_TEST_LONG: &str = r###"custom test command, for porting failed python test

Usage: mwpf custom-test

Options:
  -h, --help     Print help
  -V, --version  Print version
"###;

pub const CUSTOM_TEST_SHORT: &str = CUSTOM_TEST_LONG;

pub const CUSTOM_TEST2_LONG: &str = r###"custom test command, for comparing with ported failed python test

Usage: mwpf custom-test2

Options:
  -h, --help     Print help
  -V, --version  Print version
"###;

pub const CUSTOM_TEST2_SHORT: &str = CUSTOM_TEST2_LONG;

pub const TEST_LONG: &str = r###"built-in tests

Usage: mwpf test <COMMAND>

Commands:
  common         test common cases
  code-capacity  test various codes using code capacity noise model
  help           Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
"###;

pub const TEST_SHORT: &str = TEST_LONG;

pub const TEST_COMMON_LONG: &str = r###"test common cases

Usage: mwpf test common

Options:
  -h, --help     Print help
  -V, --version  Print version
"###;

pub const TEST_COMMON_SHORT: &str = TEST_COMMON_LONG;

pub const TEST_CODE_CAPACITY_LONG: &str = r###"test various codes using code capacity noise model

Usage: mwpf test code-capacity [OPTIONS]

Options:
  -c, --print-command
          print out the command to test

  -v, --enable-visualizer
          enable visualizer

  -u, --use-strict
          use strict verifier to check whether the result is always optimal

  -s, --print-syndrome-pattern
          enable print syndrome pattern

  -p, --solver-type <SOLVER_TYPE>
          select the combination of primal and dual module
          
          [default: union-find]

          Possible values:
          - union-find:           the solver from Union-Find decoder
          - single-hair:          the single-hair solver
          - joint-single-hair:    joint single-hair solver
          - error-pattern-logger: log error into a file for later fetch

      --solver-config <SOLVER_CONFIG>
          the configuration of primal and dual module
          
          [default: {}]

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
"###;

pub const TEST_CODE_CAPACITY_SHORT: &str = r###"test various codes using code capacity noise model

Usage: mwpf test code-capacity [OPTIONS]

Options:
  -c, --print-command                  print out the command to test
  -v, --enable-visualizer              enable visualizer
  -u, --use-strict                     use strict verifier to check whether the result is always optimal
  -s, --print-syndrome-pattern         enable print syndrome pattern
  -p, --solver-type <SOLVER_TYPE>      select the combination of primal and dual module [default: union-find] [possible values: union-find, single-hair, joint-single-hair, error-pattern-logger]
      --solver-config <SOLVER_CONFIG>  the configuration of primal and dual module [default: {}]
  -h, --help                           Print help (see more with '--help')
  -V, --version                        Print version
"###;

pub const HELP_HELP: &str = r###"Print this message or the help of the given subcommand(s)

Usage: mwpf help [COMMAND]...

Arguments:
  [COMMAND]...  Print help for the subcommand(s)
"###;
