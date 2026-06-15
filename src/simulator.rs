use crate::tableau::Tableau;

pub struct StabilizerSimulator {
    tableau: Tableau,
}

impl StabilizerSimulator {
    pub fn new(n: usize) -> Self {
        StabilizerSimulator {
            tableau: Tableau::new(n),
        }
    }

    pub fn n(&self) -> usize {
        self.tableau.n
    }

    pub fn apply_h(&mut self, qubit: usize) {
        self.tableau.apply_h(qubit);
    }

    pub fn apply_s(&mut self, qubit: usize) {
        self.tableau.apply_s(qubit);
    }

    pub fn apply_s_dag(&mut self, qubit: usize) {
        // S^dag = S^3
        self.tableau.apply_s(qubit);
        self.tableau.apply_s(qubit);
        self.tableau.apply_s(qubit);
    }

    pub fn apply_cnot(&mut self, control: usize, target: usize) {
        self.tableau.apply_cnot(control, target);
    }

    pub fn apply_x(&mut self, qubit: usize) {
        // X gate on qubit k: r_i = r_i ^ z_{i,k} for all rows i
        for i in 0..(2 * self.tableau.n) {
            let z_val = self.tableau.get_z(i, qubit);
            self.tableau.set_r(i, self.tableau.get_r(i) ^ (z_val as u8));
        }
    }

    pub fn apply_z(&mut self, qubit: usize) {
        // Z gate on qubit k: r_i = r_i ^ x_{i,k} for all rows i
        for i in 0..(2 * self.tableau.n) {
            let x_val = self.tableau.get_x(i, qubit);
            self.tableau.set_r(i, self.tableau.get_r(i) ^ (x_val as u8));
        }
    }

    pub fn apply_y(&mut self, qubit: usize) {
        // Y = i X Z. Up to global phase (which does not affect stabilizers), Y is X then Z.
        // r_i = r_i ^ x_{i,k} ^ z_{i,k}
        for i in 0..(2 * self.tableau.n) {
            let x_val = self.tableau.get_x(i, qubit);
            let z_val = self.tableau.get_z(i, qubit);
            self.tableau.set_r(i, self.tableau.get_r(i) ^ (x_val as u8) ^ (z_val as u8));
        }
    }

    pub fn measure_z(&mut self, qubit: usize) -> u8 {
        self.tableau.measure(qubit)
    }

    pub fn measure_x(&mut self, qubit: usize) -> u8 {
        // Measure X: H, measure Z, H
        self.apply_h(qubit);
        let outcome = self.measure_z(qubit);
        self.apply_h(qubit);
        outcome
    }

    pub fn measure_y(&mut self, qubit: usize) -> u8 {
        // Measure Y: S^dag, H, measure Z, H, S
        self.apply_s_dag(qubit);
        self.apply_h(qubit);
        let outcome = self.measure_z(qubit);
        self.apply_h(qubit);
        self.apply_s(qubit);
        outcome
    }

    // Helper for debugging: prints the stabilizer rows
    pub fn print_stabilizers(&self) {
        let n = self.tableau.n;
        println!("Stabilizer Generators:");
        for i in n..(2 * n) {
            let phase = if self.tableau.get_r(i) == 1 { "-" } else { "+" };
            print!("Row {}: {} ", i, phase);
            for qubit in 0..n {
                let x = self.tableau.get_x(i, qubit);
                let z = self.tableau.get_z(i, qubit);
                match (x, z) {
                    (false, false) => print!("I"),
                    (true, false) => print!("X"),
                    (false, true) => print!("Z"),
                    (true, true) => print!("Y"),
                }
            }
            println!();
        }
    }
}
