use std::iter::repeat;

use galois_fields::Z64;

use num_traits::identities::Zero;
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;

const P: u64 = 8589934583;

fn main() {
    let mut rng = rand_xoshiro::Xoshiro256StarStar::seed_from_u64(0);
    let step: Z64<P> = rng.gen();
    let sum = repeat(step)
        .take(P as usize)
        .reduce(|acc, step| acc + step)
        .unwrap();
    assert!(sum.is_zero());
}
