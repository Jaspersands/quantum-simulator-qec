"""``python -m stabilizer_qec.mqt_qecc.cc_decoder``: MQT QECC's ``mqt.qecc.cc-decoder`` command (also
``stabilizer-qec mqt.qecc.cc-decoder``). This package registers no ``mqt.qecc.cc-decoder`` script of
its own, so as not to shadow MQT QECC's."""

from .cli import cli

if __name__ == "__main__":
    cli(prog="python -m stabilizer_qec.mqt_qecc.cc_decoder")
