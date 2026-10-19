"""The analog-information simulators take MQT QECC's decoding-method names, "bposd" and "matching".

(This package's tests, not MQT's: the import rename once turned the literal "bposd" into
"stabilizer_qec.bposd", so ``decoding_method="bposd"`` was refused while the default still worked.)"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from stabilizer_qec.ldpc.bposd_decoder import BpOsdDecoder
from stabilizer_qec.mqt_qecc import QssSimulator
from stabilizer_qec.mqt_qecc.analog_information_decoding.simulators import memory_experiment_v2
from stabilizer_qec.mqt_qecc.analog_information_decoding.utils.data_utils import BpParams


def _qss(tmp_path, **kw):
    return QssSimulator(
        pcm=np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int32),
        per=0.01,
        ser=0.01,
        logicals=np.array([[1, 1, 1]], dtype=np.int32),
        bias=np.array([1.0, 1.0, 1.0]),
        codename="rep3",
        bp_params=BpParams(osd_order=0, osd_method="osd0"),
        code_params={"n": 3, "k": 1, "d": 3},
        repetitions=2,
        rounds=1,
        outpath=str(tmp_path / "qss.json"),
        **kw,
    )


def test_defaults_are_mqts_names() -> None:
    assert inspect.signature(QssSimulator).parameters["decoding_method"].default == "bposd"
    assert inspect.signature(memory_experiment_v2.decode_multiround).parameters["decoding_method"].default == "bposd"


def test_qss_takes_bposd_by_name(tmp_path) -> None:
    sim = _qss(tmp_path, decoding_method="bposd")
    assert isinstance(sim.decoder, BpOsdDecoder)
    assert sim.run(2)["code_N"] == 3


def test_qss_refuses_other_names(tmp_path) -> None:
    with pytest.raises(ValueError, match="bposd or matching"):
        _qss(tmp_path, decoding_method="stabilizer_qec.bposd")


def test_get_updated_decoder_bposd_updates_the_channel() -> None:
    pcm = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    decoder = BpOsdDecoder(pcm, error_rate=0.1, max_iter=3, bp_method="ms", osd_method="osd0")
    probs = np.array([0.2, 0.3, 0.4])
    out = memory_experiment_v2.get_updated_decoder("bposd", decoder, probs)
    assert out is decoder
    assert np.allclose(decoder.channel_probs, probs)
