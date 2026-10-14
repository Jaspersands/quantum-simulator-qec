# Ported from sinter 1.16.0 (Copyright Google LLC, Apache License 2.0; see THIRD_PARTY_NOTICES.md),
# modified for stabilizer_qec: Stim and PyMatching are this package's, and scipy is not needed.
from stabilizer_qec.sinter._decoding._decoding_decoder_class import Decoder, CompiledDecoder


class PyMatchingCompiledDecoder(CompiledDecoder):
    def __init__(self, matcher: 'pymatching.Matching', use_correlated_decoding: bool):
        self.matcher = matcher
        self.use_correlated_decoding = use_correlated_decoding

    def decode_shots_bit_packed(
            self,
            *,
            bit_packed_detection_event_data: 'np.ndarray',
    ) -> 'np.ndarray':
        kwargs = {}
        if self.use_correlated_decoding:
            kwargs['enable_correlations'] = True
        return self.matcher.decode_batch(
            shots=bit_packed_detection_event_data,
            bit_packed_shots=True,
            bit_packed_predictions=True,
            return_weights=False,
            **kwargs,
        )


class PyMatchingDecoder(Decoder):
    """Use PyMatching (this package's) to predict observables from detection events."""

    def __init__(self, use_correlated_decoding: bool = False):
        self.use_correlated_decoding = use_correlated_decoding

    def compile_decoder_for_dem(self, *, dem: 'stim.DetectorErrorModel') -> CompiledDecoder:
        from stabilizer_qec import pymatching

        kwargs = {}
        if self.use_correlated_decoding:
            kwargs['enable_correlations'] = True
        return PyMatchingCompiledDecoder(
            pymatching.Matching.from_detector_error_model(dem, **kwargs),
            use_correlated_decoding=self.use_correlated_decoding,
        )

    def decode_via_files(self,
                         *,
                         num_shots: int,
                         num_dets: int,
                         num_obs: int,
                         dem_path: 'pathlib.Path',
                         dets_b8_in_path: 'pathlib.Path',
                         obs_predictions_b8_out_path: 'pathlib.Path',
                         tmp_dir: 'pathlib.Path',
                       ) -> None:
        from stabilizer_qec import pymatching

        if num_dets == 0:
            with open(obs_predictions_b8_out_path, 'wb') as f:
                f.write(b'\0' * (num_obs * num_shots))
            return

        args = [
            "predict",
            "--dem", str(dem_path),
            "--in", str(dets_b8_in_path),
            "--in_format", "b8",
            "--out", str(obs_predictions_b8_out_path),
            "--out_format", "b8",
        ]
        if self.use_correlated_decoding:
            args.append("--enable_correlations")

        result = pymatching.cli(command_line_args=args)
        if result:
            raise ValueError("pymatching.cli returned a non-zero exit code")
