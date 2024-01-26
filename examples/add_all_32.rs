use ffnt::Z32;

use num_traits::identities::Zero;

const P: u32 = 1073741789;

fn main() {
    let sum = (1..P).map(Z32::<P>::from)
        .reduce(|acc, t| acc + t)
        .unwrap();
    assert!(sum.is_zero());
}
