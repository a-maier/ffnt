use ffnt::Z32;

use num_traits::identities::{Zero, One};

const P: u32 = 1073741789;

fn main() {
    let fac = (1..P).map(Z32::<P>::from)
        .reduce(|acc, t| acc * t)
        .unwrap();
    assert!((fac + Z32::<P>::one()).is_zero());
}
