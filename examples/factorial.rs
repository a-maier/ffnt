use galois_fields::Z64;

use num_traits::identities::Zero;

const P: u64 = 2147483647;

fn main() {
    let fac = (1..P).map(Z64::<P>::new_unchecked)
        .reduce(|acc, t| acc * t)
        .unwrap();
    assert!((fac + Z64::<P>::new_unchecked(1)).is_zero());
}
