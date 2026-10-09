//! The crate's public API, used as a dependent crate would use it.

use stabilizer_qec::{
    lattice_surgery, memory_circuit, Basis, BeliefMatching, BitTable, BivariateBicycleCode, BpDecoder, BpLsd, BpLsdDecoder, BpMethod,
    BpOptions, BpOsd, BpOsdDecoder, ColorMatching, LsdOptions, RelayBp, RelayBpDecoder, RelayOptions, SearchDecoder, SearchOptions, DetectorOrder, Circuit, DemOptions, DetectorErrorModel, Error, GrossOperator, Matching, Noise, OsdMethod, Pauli,
    SurfaceCode, Target, WindowMatching, WindowMode, WindowOptions,
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
    shareable::<BpLsd>();
    shareable::<BpLsdDecoder>();
    shareable::<RelayBp>();
    shareable::<RelayBpDecoder>();
    shareable::<ColorMatching>();
    shareable::<SearchDecoder>();
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
    let relay = RelayBp::new(&undecomposed, RelayOptions::new().legs(5).solutions(Some(1))).unwrap();
    let r = relay.decode_batch(&samples.detectors, 0).unwrap();
    assert!(failure_rate(&r, &samples.observables) < 1.5 * glob + 0.02);
    let first: Vec<u32> = (0..samples.detectors.num_bits()).filter(|&d| samples.detectors.get(0, d)).map(|d| d as u32).collect();
    assert_eq!(r[0], relay.decode(&first).unwrap());
    let search = SearchDecoder::new(&undecomposed, SearchOptions::new().generated_orders(DetectorOrder::Index, 2, 1)).unwrap();
    let found = search.decode_batch(&samples.detectors, 0).unwrap();
    assert!(failure_rate(&found, &samples.observables) < 1.5 * glob + 0.02);
    assert!(found.iter().all(|p| p.weight.is_some()));
    let lsd = BpLsd::new(&undecomposed, BpOptions::new(30, BpMethod::MinimumSum { scaling_factor: 0.625 }), LsdOptions::new(OsdMethod::Osd0)).unwrap();
    assert!(failure_rate(&lsd.decode_batch(&samples.detectors, 0).unwrap(), &samples.observables) < 1.5 * glob + 0.02);
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
    let lsd = BpLsdDecoder::new(3, &columns, &priors, BpOptions::new(1, BpMethod::ProductSum), LsdOptions::new(OsdMethod::Osd0).with_always_run(true)).unwrap();
    for e in 0..7 {
        let syndrome: Vec<bool> = (0..3).map(|r| columns[e].contains(&r)).collect();
        let c = lsd.decode(&syndrome).unwrap().correction;
        let explained: Vec<bool> = (0..3u32).map(|r| columns.iter().zip(&c).filter(|(col, &x)| x && col.contains(&r)).count() % 2 == 1).collect();
        assert_eq!(explained, syndrome);
    }
    let relay = RelayBpDecoder::new(3, &columns, &priors, RelayOptions::new().pre_iterations(1).gammas(vec![vec![0.3; 7]])).unwrap();
    for e in 0..7 {
        let syndrome: Vec<bool> = (0..3).map(|r| columns[e].contains(&r)).collect();
        let out = relay.decode(&syndrome).unwrap();
        if out.converged {
            let explained: Vec<bool> = (0..3u32).map(|r| columns.iter().zip(&out.correction).filter(|(col, &x)| x && col.contains(&r)).count() % 2 == 1).collect();
            assert_eq!(explained, syndrome);
            assert!(out.weight.is_finite() && out.legs >= 1);
        }
    }
    assert!(RelayBpDecoder::new(3, &columns, &priors, RelayOptions::new().gamma_range(0.5, 0.1)).is_err());
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
    let text = "R[init] 0 1\nH 1\nX_ERROR[noise](0.125) 0\nOBSERVABLE_INCLUDE[l](0) Z0 !X1\nM 0\nDETECTOR[d] rec[-1]";
    let c: Circuit = text.parse().unwrap();
    assert_eq!(c.to_string().trim(), text);
    assert_eq!((c.num_qubits(), c.num_observables()), (2, 1));
    let dem = c.detector_error_model(&DemOptions::new()).unwrap();
    let printed = dem.to_string();
    for line in ["detector[d] D0", "logical_observable[l] L0", "error[noise](0.125) D0 L0"] {
        assert!(printed.contains(line), "{printed}");
    }
    assert_eq!(printed.parse::<DetectorErrorModel>().unwrap().to_string(), printed);
    let samples = c.detector_sampler(3).unwrap().sample(4000, 1);
    let both = (0..4000).filter(|&s| samples.detectors.get(s, 0) && samples.observables.get(s, 0)).count();
    assert!((400..600).contains(&both), "{both}");
    // A Pauli observable the state does not fix is refused, as Stim refuses it.
    let random: Circuit = "R 0\nOBSERVABLE_INCLUDE(0) X0\nM 0\nDETECTOR rec[-1]".parse().unwrap();
    assert!(random.detector_error_model(&DemOptions::new()).unwrap_err().message().contains("not deterministic"));
}

#[test]
fn circuits_built_in_code() {
    use Target::{Combiner, Inverted, Qubit, Rec, Sweep};
    let mut c = Circuit::new();
    c.append("R", &[Qubit(0), Qubit(1), Qubit(2)], &[]).unwrap();
    c.append_tagged("H", &[Qubit(0)], &[], "prep").unwrap();
    c.append("CX", &[Qubit(0), Qubit(1), Sweep(3), Qubit(2)], &[]).unwrap();
    c.append("MPP", &[Target::Pauli { pauli: Pauli::X, qubit: 0, inverted: false }, Combiner, Target::Pauli { pauli: Pauli::X, qubit: 1, inverted: true }], &[0.01])
        .unwrap();
    c.append("M", &[Inverted(2)], &[]).unwrap();
    c.append("DETECTOR", &[Rec(2)], &[1.0, 2.0]).unwrap();
    c.append("OBSERVABLE_INCLUDE", &[Rec(1)], &[0.0]).unwrap();
    assert_eq!(
        c.to_string(),
        "R 0 1 2\nH[prep] 0\nCX 0 1 sweep[3] 2\nMPP(0.01) X0*!X1\nM !2\nDETECTOR(1, 2) rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
    );
    assert_eq!((c.num_qubits(), c.num_measurements(), c.num_detectors(), c.num_observables(), c.num_sweep_bits()), (3, 2, 1, 1, 4));
    // A failed append leaves the circuit as it was.
    let before = c.clone();
    for (name, targets) in [("FOO", vec![Qubit(0)]), ("H\nM", vec![Qubit(0)]), ("CX", vec![Qubit(0)]), ("MPP", vec![Combiner]), ("REPEAT", vec![])] {
        assert!(c.append(name, &targets, &[]).is_err(), "{name}");
    }
    assert!(c.append("X_ERROR", &[Qubit(0)], &[f64::NAN]).is_err());
    assert!(c.append_tagged("H", &[Qubit(0)], &[], "a]b").is_err());
    assert_eq!(c, before);
    // A round reads the last round's record; the whole, not the piece, is what runs.
    let mut round = Circuit::new();
    round.append_text("CX 0 1\nMR 1\nDETECTOR rec[-1] rec[-2]").unwrap();
    assert!(round.detector_sampler(1).is_err() && round.detector_error_model(&DemOptions::new()).is_err());
    let first: Circuit = "R 0 1\nCX 0 1\nMR 1".parse().unwrap();
    let memory = &first + &(&round * 20);
    assert_eq!((memory.num_measurements(), memory.num_detectors()), (21, 20));
    assert_eq!(memory.to_string().parse::<Circuit>().unwrap(), memory);
    assert!(memory.detector_error_model(&DemOptions::new()).is_ok());
    // Repeating a loop multiplies its count, as in Stim.
    assert_eq!((&round * 20) * 3, &round * 60);
    let mut grown = first.clone();
    grown += &round;
    grown *= 1;
    assert_eq!(grown.num_detectors(), 1);
    assert_eq!(round.repeated(0).unwrap(), Circuit::new());
}

#[test]
fn gross_code_automorphisms_gauging_and_streams() {
    use stabilizer_qec::{stream_memory, Automorphism};
    let gross = BivariateBicycleCode::gross();
    let autos: Vec<Automorphism> = gross.automorphisms().unwrap();
    assert_eq!(autos.len(), 144);
    // Each action is invertible over GF(2): its 24 rows are independent.
    for a in &autos {
        let mut rows: Vec<u32> = a.action.iter().map(|r| r.iter().fold(0u32, |m, &i| m | 1 << i)).collect();
        let mut rank = 0;
        for bit in 0..24 {
            if let Some(i) = (rank..rows.len()).find(|&i| rows[i] >> bit & 1 == 1) {
                rows.swap(rank, i);
                for j in 0..rows.len() {
                    if j != rank && rows[j] >> bit & 1 == 1 {
                        rows[j] ^= rows[rank];
                    }
                }
                rank += 1;
            }
        }
        assert_eq!(rank, 24, "{:?}", a.shift);
    }
    for (op, expanded) in [(GrossOperator::F, false), (GrossOperator::FTimesGh, true)] {
        let g = gross.gauging(op, expanded).unwrap();
        let n = 144 + g.incidence.len();
        assert!(g.hx.iter().chain(&g.hz).flatten().all(|&q| q < n));
        // Each edge meets the support an even number of times (twice, or four times for f·gh).
        assert!(g.incidence.iter().all(|e| !e.is_empty() && e.len() % 2 == 0 && e.iter().all(|&v| v < g.support.len())));
        if expanded {
            assert!(g.worst_cut.0 >= g.worst_cut.1, "{:?}", g.worst_cut);
        }
    }
    assert!(BivariateBicycleCode::bb72().gauging(GrossOperator::F, false).is_err());
    // A stream's failures depend on its seed, not its threads.
    let opts = WindowOptions::new(2, 2, WindowMode::Parallel);
    let one = stream_memory(SurfaceCode::Rotated, 3, 300, 0.004, opts, 256, 9, 1).unwrap();
    let four = stream_memory(SurfaceCode::Rotated, 3, 300, 0.004, opts, 256, 9, 4).unwrap();
    assert_eq!((one.failures, one.shots), (four.failures, four.shots));
    assert_eq!(one.window_seconds.len(), 4);
    assert!(one.window_seconds.iter().all(|w| w.len() == one.windows.len()));
    assert!(stream_memory(SurfaceCode::Rotated, 3, 0, 0.004, opts, 64, 1, 1).is_err());
}

#[test]
fn coherent_exact_and_twirled() {
    use stabilizer_qec::{Circuit, CoherentOptions};
    let c = Circuit::parse("R 0\nI_ERROR[R_X(theta=0.3)] 0\nM 0\nDETECTOR rec[-1]").unwrap();
    // The twirl: X with probability sin²(0.3).
    let t = c.twirled(false).unwrap();
    assert!(t.to_string().contains("X_ERROR("), "{t}");
    // Exact shots and weighted shots both fire at sin²(0.3).
    let p = 0.3f64.sin().powi(2);
    let e = c.exact_sampler(1).unwrap().sample(20_000, 0);
    let hits = (0..20_000).filter(|&k| e.detectors.get(k, 0)).count() as f64 / 20_000.0;
    assert!((hits - p).abs() < 0.02);
    let mut s = c.coherent_sampler(2, CoherentOptions::new().order(2)).unwrap();
    assert_eq!(s.num_locations(), 1);
    let w = s.sample(20_000, 0);
    let (hit, total) = w.weights.iter().enumerate().fold((0.0, 0.0), |(h, t), (k, &x)| (h + if w.detectors.get(k, 0) { x } else { 0.0 }, t + x));
    assert!((hit / total - p).abs() < 0.02);
}
