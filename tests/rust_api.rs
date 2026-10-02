//! The crate's public API, used as a dependent crate would use it.

use stabilizer_qec::{
    lattice_surgery, memory_circuit, Basis, BeliefMatching, BitTable, BivariateBicycleCode, BpDecoder, BpMethod, BpOptions,
    BpOsd, BpOsdDecoder, Circuit, DemOptions, DetectorErrorModel, Error, GrossOperator, Matching, Noise, OsdMethod,
    SurfaceCode, WindowMatching, WindowMode, WindowOptions,
};

fn d5() -> Circuit {
    memory_circuit(SurfaceCode::Rotated, 5, 5, Noise::Sd6 { p: 0.004 }, Basis::Z).unwrap()
}

fn failure_rate(predictions: &[stabilizer_qec::Prediction], observables: &BitTable) -> f64 {
    let wrong = predictions.iter().enumerate().filter(|(s, p)| p.flips(0) != observables.get(*s, 0)).count();
    wrong as f64 / predictions.len() as f64
}

#[test]
fn decoders_and_samplers_cross_threads() {
    fn shareable<T: Send + Sync>() {}
    shareable::<Circuit>();
    shareable::<DetectorErrorModel>();
    shareable::<Matching>();
    shareable::<BeliefMatching>();
    shareable::<BpOsd>();
    shareable::<WindowMatching>();
    shareable::<BpDecoder>();
    shareable::<BpOsdDecoder>();
    fn sendable<T: Send>() {}
    sendable::<stabilizer_qec::DetectorSampler>();
    sendable::<stabilizer_qec::MeasurementConverter>();
    fn error<E: std::error::Error + Send + Sync + 'static>() {}
    error::<Error>();
}

#[test]
fn circuits_parse_print_and_count() {
    let c: Circuit = "R 0 1\nH 0\nCX 0 1\nM 0 1\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]".parse().unwrap();
    assert_eq!((c.num_qubits(), c.num_measurements(), c.num_detectors(), c.num_observables()), (2, 2, 1, 1));
    assert_eq!(c.to_string().parse::<Circuit>().unwrap(), c);
    let looped: Circuit = "REPEAT 1000000000 {\n M 0\n DETECTOR rec[-1]\n}\nCZ sweep[3] 0".parse().unwrap();
    assert_eq!((looped.num_detectors(), looped.num_sweep_bits()), (1_000_000_000, 4));
    let err = "FOO 0".parse::<Circuit>().unwrap_err();
    assert!(err.message().contains("unsupported instruction"), "{err}");
}

#[test]
fn error_models_round_trip() {
    let dem = d5().detector_error_model(&DemOptions::new().decompose_errors(true)).unwrap();
    let again: DetectorErrorModel = dem.to_string().parse().unwrap();
    assert_eq!((again.num_detectors(), again.num_observables(), again.num_errors()), (120, 1, dem.num_errors()));
    let disjoint: Circuit = "R 0 1\nPAULI_CHANNEL_2(0.01, 0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) 0 1\nM 0 1\nDETECTOR rec[-1]".parse().unwrap();
    assert!(disjoint.detector_error_model(&DemOptions::new()).is_err());
    assert!(disjoint.detector_error_model(&DemOptions::new().approximate_disjoint_errors(Some(1.0))).is_ok());
}

#[test]
fn a_seed_gives_the_same_shots_on_any_number_of_threads() {
    let c = d5();
    let one = c.detector_sampler(42).unwrap().sample(3000, 1);
    assert_eq!(c.detector_sampler(42).unwrap().sample(3000, 7), one);
    assert_ne!(c.detector_sampler(43).unwrap().sample(3000, 1), one);
    let mut s = c.detector_sampler(5).unwrap();
    let (a, b) = (s.sample(64, 1), s.sample(64, 1));
    let whole = c.detector_sampler(5).unwrap().sample(128, 1);
    assert_eq!([a.detectors.as_bytes(), b.detectors.as_bytes()].concat(), whole.detectors.as_bytes());
}

#[test]
fn raw_measurements_convert() {
    let c: Circuit = "R 0 1\nCX sweep[0] 0\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]".parse().unwrap();
    let conv = c.measurement_converter().unwrap();
    let meas = BitTable::from_rows(2, &[[true, false], [false, false]]).unwrap();
    let sweeps = BitTable::from_rows(1, &[[true], [false]]).unwrap();
    let out = conv.convert(&meas, Some(&sweeps)).unwrap();
    assert!(!out.detectors.get(0, 0) && !out.detectors.get(1, 0)); // the sweep explained shot 0
    assert!(conv.convert(&BitTable::zeros(2, 3), None).is_err());
}

#[test]
fn matching_decodes_batches_and_shots() {
    let c = d5();
    let dem = c.detector_error_model(&DemOptions::new().decompose_errors(true)).unwrap();
    let samples = c.detector_sampler(1).unwrap().sample(20_000, 0);
    let plain = Matching::new(&dem).unwrap();
    let batch = plain.decode_batch(&samples.detectors, 0).unwrap();
    assert_eq!(batch, plain.decode_batch(&samples.detectors, 1).unwrap());
    assert_eq!(plain.decode(&samples.detectors.ones(17)).unwrap(), batch[17]);
    let p = failure_rate(&batch, &samples.observables);
    let corr = failure_rate(&Matching::with_correlations(&dem).unwrap().decode_batch(&samples.detectors, 0).unwrap(), &samples.observables);
    assert!(p > 0.0 && p < 0.1 && corr < p + 3.0 * (p / 20_000.0).sqrt(), "{p} {corr}");
    let unmatchable = Matching::new(&"error(0.1) D0 D1\nerror(0.1) D1 D2 L0".parse().unwrap()).unwrap();
    let shots = BitTable::from_rows(3, &[[false, false, false], [true, false, false]]).unwrap();
    assert!(unmatchable.decode_batch(&shots, 1).unwrap_err().message().contains("shot 1"));
    assert!(plain.decode(&[10_000]).is_err());
}

#[test]
fn belief_matching_bposd_and_windows() {
    let c = memory_circuit(SurfaceCode::Rotated, 3, 30, Noise::Sd6 { p: 0.005 }, Basis::Z).unwrap();
    let dem = c.detector_error_model(&DemOptions::new().decompose_errors(true)).unwrap();
    let samples = c.detector_sampler(2).unwrap().sample(2000, 0);
    let glob = failure_rate(&Matching::new(&dem).unwrap().decode_batch(&samples.detectors, 0).unwrap(), &samples.observables);
    let bm = BeliefMatching::new(&dem, BpOptions::new(20, BpMethod::ProductSum)).unwrap();
    let b = bm.decode_batch(&samples.detectors, 0).unwrap();
    assert!(failure_rate(&b, &samples.observables) < glob + 0.02);
    assert!(b.iter().all(|p| p.bp_converged.is_some()));
    let undecomposed = c.detector_error_model(&DemOptions::new()).unwrap();
    let osd = BpOsd::new(&undecomposed, BpOptions::new(30, BpMethod::MinimumSum { scaling_factor: 0.0 }), OsdMethod::CombinationSweep(4)).unwrap();
    assert!(failure_rate(&osd.decode_batch(&samples.detectors, 0).unwrap(), &samples.observables) < 1.5 * glob + 0.02);
    for mode in [WindowMode::Sliding, WindowMode::Parallel] {
        let w = WindowMatching::new(&dem, WindowOptions::new(4, 4, mode)).unwrap();
        assert!(!w.windows().is_empty());
        assert!(failure_rate(&w.decode_batch(&samples.detectors, 0).unwrap(), &samples.observables) < glob + 0.03);
    }
}

#[test]
fn check_matrix_decoders() {
    // The [7, 4] Hamming code's checks, by column.
    let columns: Vec<Vec<u32>> = (1..=7u32).map(|j| (0..3).filter(|r| (j >> r) & 1 == 1).collect()).collect();
    let priors = vec![0.05; 7];
    let bp = BpDecoder::new(3, &columns, &priors, BpOptions::new(10, BpMethod::ProductSum)).unwrap();
    let osd = BpOsdDecoder::new(3, &columns, &priors, BpOptions::new(10, BpMethod::MinimumSum { scaling_factor: 0.0 }), OsdMethod::Osd0).unwrap();
    for e in 0..7 {
        let syndrome: Vec<bool> = (0..3).map(|r| columns[e].contains(&r)).collect();
        let c = osd.decode(&syndrome).unwrap().correction;
        let explained: Vec<bool> = (0..3u32).map(|r| columns.iter().zip(&c).filter(|(col, &x)| x && col.contains(&r)).count() % 2 == 1).collect();
        assert_eq!(explained, syndrome);
        assert_eq!(bp.decode(&syndrome).unwrap().log_prob_ratios.len(), 7);
    }
    assert!(BpDecoder::new(3, &columns, &[0.05; 6], BpOptions::new(10, BpMethod::ProductSum)).is_err());
    assert!(bp.decode(&[true]).is_err());
}

#[test]
fn generated_circuits() {
    assert!(memory_circuit(SurfaceCode::Xzzx, 3, 3, Noise::Biased { p: 0.001, eta: 10.0 }, Basis::X).is_ok());
    assert!(memory_circuit(SurfaceCode::Rotated, 13, 3, Noise::Sd6 { p: 0.001 }, Basis::Z).is_err());
    assert!(memory_circuit(SurfaceCode::Rotated, 3, 3, Noise::Sd6 { p: f64::NAN }, Basis::Z).is_err());
    let gross = BivariateBicycleCode::gross();
    let (hx, hz) = gross.check_matrices();
    assert_eq!((hx.len(), hz.len(), gross.logicals().0.len()), (72, 72, 12));
    let m = gross.logical_measurement_circuit(GrossOperator::F, Basis::X, 1, 2, 1, 0.003, false).unwrap();
    assert_eq!(m.num_observables(), 13);
    assert!(BivariateBicycleCode::bb72().logical_measurement_circuit(GrossOperator::F, Basis::X, 1, 2, 1, 0.003, false).is_err());
    assert_eq!(lattice_surgery::cnot(3, 3, 0.002, Basis::X).unwrap().num_observables(), 2);
    assert_eq!(lattice_surgery::line(3, 4, 2, 0.002).unwrap().num_observables(), 7);
    assert!(lattice_surgery::line(3, 40, 2, 0.002).is_err());
    assert!(lattice_surgery::cnot(1001, 3, 0.002, Basis::Z).is_err());
    assert!(lattice_surgery::repeated_zz(3, 2, 2, 0.002).is_ok());
    assert!(lattice_surgery::xx_measurement(3, 3, 0.002, Basis::Z).is_ok());
}

#[test]
fn stim_tags_and_pauli_observables() {
    let text = "R[init] 0 1\nH 1\nX_ERROR[noise](0.1) 0\nOBSERVABLE_INCLUDE[l](0) Z0 !X1\nM 0\nDETECTOR[d] rec[-1]";
    let c: Circuit = text.parse().unwrap();
    assert_eq!(c.to_string().trim(), text);
    assert_eq!((c.num_qubits(), c.num_observables()), (2, 1));
    let dem = c.detector_error_model(&DemOptions::new()).unwrap();
    let printed = dem.to_string();
    for line in ["detector[d] D0", "logical_observable[l] L0", "error[noise](0.1) D0 L0"] {
        assert!(printed.contains(line), "{printed}");
    }
    assert_eq!(printed.parse::<DetectorErrorModel>().unwrap().to_string(), printed);
    let samples = c.detector_sampler(3).unwrap().sample(4000, 1);
    let both = (0..4000).filter(|&s| samples.detectors.get(s, 0) && samples.observables.get(s, 0)).count();
    assert!((300..500).contains(&both), "{both}");
    // A Pauli observable the state does not fix is refused, as Stim refuses it.
    let random: Circuit = "R 0\nOBSERVABLE_INCLUDE(0) X0\nM 0\nDETECTOR rec[-1]".parse().unwrap();
    assert!(random.detector_error_model(&DemOptions::new()).unwrap_err().message().contains("not deterministic"));
}
