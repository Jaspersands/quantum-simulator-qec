"""Bounded ownership for loopback tests whose server startup runs blocking OS calls.

Keep the entire test in one child so Bottle's shared default routes still behave normally.
Only report its result after subprocess.run has reaped it; timeout kills the whole owned
process, including delayed server threads and any listener they created.
"""
import functools
import json
import os
import pathlib
import subprocess
import sys

import pytest

_GUARD = 'STABILIZER_QEC_OWNED_SERVER_TEST'
_RESULT = 'QEC_SERVER_TEST_RESULT='
PROCESS_TIMEOUT = 20


def in_owned_server_process():
    return bool(os.environ.get(_GUARD))


def run_owned(command, *, timeout=PROCESS_TIMEOUT, env=None):
    child_env = dict(os.environ) if env is None else dict(env)
    child_env.setdefault(_GUARD, 'standalone-server-control')
    # subprocess.run kills and waits for this exact child before raising TimeoutExpired.
    # Server workers are threads in this process, so they cannot survive its reaping.
    return subprocess.run(command, env=child_env, capture_output=True, text=True, timeout=timeout)


_WORKER = '''
import faulthandler, json, sys
sys.path.insert(0, sys.argv[2])
import pytest
class Result:
    skipped = None
    def pytest_runtest_logreport(self, report):
        if report.skipped:
            self.skipped = str(report.longrepr)
result = Result()
faulthandler.dump_traceback_later(float(sys.argv[3]) - 2)
code = pytest.main(['-q', '-p', 'no:cacheprovider', sys.argv[1]], plugins=[result])
faulthandler.cancel_dump_traceback_later()
print('QEC_SERVER_TEST_RESULT=' + json.dumps({'code': int(code), 'skipped': result.skipped}), flush=True)
raise SystemExit(code)
'''


def isolated_server_test(test):
    @functools.wraps(test)
    def owned(*args, **kwargs):
        # pytest publishes the parametrized node as "path::test[id] (call)". Resolve
        # the file independently so this also works from env.sh's different cwd.
        current = os.environ['PYTEST_CURRENT_TEST']
        case = current.split('::', 1)[1].rsplit(' (', 1)[0]
        node = str(pathlib.Path(test.__code__.co_filename).resolve()) + '::' + case
        if os.environ.get(_GUARD) == node:
            return test(*args, **kwargs)
        env = dict(os.environ, **{_GUARD: node})
        try:
            result = run_owned([sys.executable, '-c', _WORKER, node, str(pathlib.Path(__file__).parent), str(PROCESS_TIMEOUT)], env=env)
        except subprocess.TimeoutExpired as error:
            pytest.fail('owned server process timed out and was killed/reaped: ' + str(error.output) + '\n' + str(error.stderr), pytrace=False)
        assert result.returncode == 0, result.stdout + '\n' + result.stderr
        records = [line[len(_RESULT):] for line in result.stdout.splitlines() if line.startswith(_RESULT)]
        assert len(records) == 1, result.stdout + '\n' + result.stderr
        report = json.loads(records[0])
        assert report['code'] == 0
        if report['skipped'] is not None:
            pytest.skip(report['skipped'])
    return owned
