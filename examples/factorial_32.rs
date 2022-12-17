use galois_fields::Z32;

use num_traits::identities::Zero;

const P: u32 = 1073741789;

fn main() {
    let fac = (1..P).map(Z32::<P>::new_unchecked)
        .reduce(|acc, t| acc * t)
        .unwrap();
    assert!((fac + Z32::<P>::new_unchecked(1)).is_zero());
}
