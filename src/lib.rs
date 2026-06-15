pub mod tableau;
pub mod simulator;
pub mod decoder;
pub mod surface_code;

use pyo3::prelude::*;

#[pyclass(name = "RotatedSurfaceCode")]
pub struct PyRotatedSurfaceCode {
    code: surface_code::RotatedSurfaceCode,
}

#[pymethods]
impl PyRotatedSurfaceCode {
    #[new]
    fn new(d: usize) -> Self {
        PyRotatedSurfaceCode {
            code: surface_code::RotatedSurfaceCode::new(d),
        }
    }

    fn simulate(&self, num_rounds: usize, p: f64) -> bool {
        self.code.simulate_phenomenological_noise(num_rounds, p)
    }

    fn simulate_data_noise(&self, p: f64) -> bool {
        self.code.simulate_data_noise(p)
    }

    #[getter]
    fn d(&self) -> usize {
        self.code.d
    }
}

#[pymodule]
fn stabilizer_qec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRotatedSurfaceCode>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::simulator::StabilizerSimulator;

    #[test]
    fn test_bell_state() {
        for _ in 0..100 {
            let mut sim = StabilizerSimulator::new(2);
            sim.apply_h(0);
            sim.apply_cnot(0, 1);

            let m0 = sim.measure_z(0);
            let m1 = sim.measure_z(1);

            // In a Bell state, measuring both qubits in Z basis must yield identical outcomes (00 or 11)
            assert_eq!(m0, m1);
        }
    }

    #[test]
    fn test_ghz_state() {
        for _ in 0..100 {
            let mut sim = StabilizerSimulator::new(3);
            sim.apply_h(0);
            sim.apply_cnot(0, 1);
            sim.apply_cnot(1, 2);

            let m0 = sim.measure_z(0);
            let m1 = sim.measure_z(1);
            let m2 = sim.measure_z(2);

            // GHZ state must result in either 000 or 111
            assert_eq!(m0, m1);
            assert_eq!(m1, m2);
        }
    }

    #[test]
    fn test_teleportation_x_basis() {
        for _ in 0..100 {
            let mut sim = StabilizerSimulator::new(3);
            // Qubit 0: message to teleport. Set to |+> state
            sim.apply_h(0);

            // Qubits 1 & 2: EPR pair
            sim.apply_h(1);
            sim.apply_cnot(1, 2);

            // Bell measurement on 0 & 1
            sim.apply_cnot(0, 1);
            sim.apply_h(0);

            let m1 = sim.measure_z(1);
            let m0 = sim.measure_z(0);

            // Active feedback on 2
            if m1 == 1 {
                sim.apply_x(2);
            }
            if m0 == 1 {
                sim.apply_z(2);
            }

            // Qubit 2 should now be in the |+> state.
            // Measuring it in X basis must always yield 0 (which means +1 eigenstate).
            let m2 = sim.measure_x(2);
            assert_eq!(m2, 0);
        }
    }

    #[test]
    fn test_teleportation_y_basis() {
        for _ in 0..100 {
            let mut sim = StabilizerSimulator::new(3);
            // Qubit 0: message to teleport. Set to |i> state (Y-eigenstate)
            sim.apply_h(0);
            sim.apply_s(0);

            // Qubits 1 & 2: EPR pair
            sim.apply_h(1);
            sim.apply_cnot(1, 2);

            // Bell measurement on 0 & 1
            sim.apply_cnot(0, 1);
            sim.apply_h(0);

            let m1 = sim.measure_z(1);
            let m0 = sim.measure_z(0);

            // Active feedback on 2
            if m1 == 1 {
                sim.apply_x(2);
            }
            if m0 == 1 {
                sim.apply_z(2);
            }

            // Qubit 2 should now be in the |i> state.
            // Measuring it in Y basis must always yield 0 (+1 eigenstate).
            let m2 = sim.measure_y(2);
            assert_eq!(m2, 0);
        }
    }

    #[test]
    fn test_single_qubit_gates() {
        let mut sim = StabilizerSimulator::new(1);
        
        // Z gate on |0> does nothing (phase remains +1)
        sim.apply_z(0);
        assert_eq!(sim.measure_z(0), 0);

        // X gate flips |0> to |1>
        sim.apply_x(0);
        assert_eq!(sim.measure_z(0), 1);

        // Y gate flips |1> to |0> (ignoring global phase)
        sim.apply_y(0);
        assert_eq!(sim.measure_z(0), 0);
    }

    #[test]
    fn test_surface_code_zero_noise() {
        let code = crate::surface_code::RotatedSurfaceCode::new(3);
        for _ in 0..10 {
            let logical_err = code.simulate_phenomenological_noise(3, 0.0);
            assert!(!logical_err);
        }
    }

    #[test]
    fn test_surface_code_low_noise() {
        let code = crate::surface_code::RotatedSurfaceCode::new(3);
        let mut error_count = 0;
        let num_runs = 500;
        for _ in 0..num_runs {
            if code.simulate_phenomenological_noise(3, 0.005) {
                error_count += 1;
            }
        }
        let error_rate = (error_count as f64) / (num_runs as f64);
        println!("d=3, p=0.005: logical error rate = {}", error_rate);
        assert!(error_rate < 0.12, "Logical error rate {} too high for p=0.005", error_rate);
    }
}
