use std::iter::repeat;

use galois_fields::Z32;

use num_traits::identities::Zero;
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;

const P: u32 = 1073741789;

fn main() {
    let mut rng = rand_xoshiro::Xoshiro256StarStar::seed_from_u64(0);
    let step: Z32<P> = rng.gen();
    let sum = repeat(step)
        .take(P as usize)
        .reduce(|acc, step| acc + step)
        .unwrap();
    assert!(sum.is_zero());
}
