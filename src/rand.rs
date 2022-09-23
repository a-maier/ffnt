use rand::distributions::{Distribution, Standard};
use rand::Rng;

use crate::Z64;

impl<const P: u64> Distribution<Z64<P>> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Z64<P> {
        Z64::new_unchecked(rng.gen_range(0..P))
    }
}

#[cfg(test)]
mod tests {

    use rand::{Rng, SeedableRng};

    const PRIMES: [u64; 3] = [3, 443619635352171979, 1152921504606846883];

    #[test]
    fn gen() {
        use crate::Z64;
        let mut rng = rand_xoshiro::Xoshiro256StarStar::seed_from_u64(0);

        for _ in 0..1000 {
            let _z: Z64<{ PRIMES[0] }> = rng.gen();
            let _z: Z64<{ PRIMES[1] }> = rng.gen();
            let _z: Z64<{ PRIMES[2] }> = rng.gen();
        }
    }
}
