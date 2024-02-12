use ffnt::Z64;

use num_traits::identities::{One, Zero};

const P: u64 = 2147483647;

fn main() {
    let fac = (1..P).map(Z64::<P>::from).reduce(|acc, t| acc * t).unwrap();
    assert!((fac + Z64::<P>::one()).is_zero());
}
