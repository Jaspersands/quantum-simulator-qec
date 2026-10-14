"""ldpc's tests need scipy, as stabilizer_qec.ldpc does: without it they are not collected."""

import importlib.util

collect_ignore_glob = [] if importlib.util.find_spec("scipy") else ["test_*.py"]
