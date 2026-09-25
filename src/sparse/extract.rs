//! The matching, read off the final regions.
//!
//! Every top region is matched to another or to the boundary. A matched
//! blossom comes apart recursively: the child holding the matched edge's
//! endpoint takes the match, and the rest of the cycle pairs off along its own
//! edges. The prediction is the XOR of every matched edge's observables. The
//! integer weight is the sum of every region's radius, which by LP duality is
//! the weight of the matching.

use crate::dem_decoder::{Prediction, SCALE};

use super::state::{BOUNDARY, NONE};
use super::Solver;

impl<'a> Solver<'a> {
    pub(crate) fn extract(&self) -> Prediction {
        let now = self.s.now;
        let mut iweight = 0i64;
        let mut observables = 0u64;
        for (r, reg) in self.s.regions.iter().enumerate() {
            if reg.dead {
                continue;
            }
            iweight += reg.radius.at(now);
            if reg.blossom_parent != NONE {
                continue;
            }
            let (partner, e) = reg.matched.expect("every top region is matched at the end");
            if partner == BOUNDARY || (r as u32) < partner {
                observables ^= e.obs;
                self.expand(r as u32, e.a, &mut observables);
                if partner != BOUNDARY {
                    self.expand(partner, e.b, &mut observables);
                }
            }
        }
        Prediction { observables, weight: iweight as f64 / SCALE, iweight }
    }

    fn expand(&self, r: u32, a: u32, observables: &mut u64) {
        let children = &self.s.regions[r as usize].children;
        if children.is_empty() {
            return;
        }
        let k = children.len();
        let i = self.child_index(r, children, a);
        let mut j = (i + 1) % k;
        while j != i {
            let next = (j + 1) % k;
            let e = children[j].1;
            *observables ^= e.obs;
            self.expand(children[j].0, e.a, observables);
            self.expand(children[next].0, e.b, observables);
            j = (next + 1) % k;
        }
        self.expand(children[i].0, a, observables);
    }
}
