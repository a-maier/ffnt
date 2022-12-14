use galois_fields::Z64;

use num_traits::identities::Zero;

const P: u64 = 8589934583;

fn main() {
    let sum = (1..P).map(Z64::<P>::new_unchecked)
        .reduce(|acc, t| acc + t)
        .unwrap();
    assert!(sum.is_zero());
}
