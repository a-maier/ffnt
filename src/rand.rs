use rand::distributions::{Distribution, Standard, uniform::{SampleBorrow, SampleUniform, UniformInt, UniformSampler}};
use rand::Rng;

use crate::Z64;

impl<const P: u64> Distribution<Z64<P>> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Z64<P> {
        Z64::new_unchecked(rng.gen_range(0..P))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct UniformZ64<const P: u64>(UniformInt<u64>);

impl<const P: u64> UniformSampler for UniformZ64<P> {
    type X = Z64<P>;

    fn new<B1, B2>(low: B1, high: B2) -> Self
        where B1: SampleBorrow<Self::X> + Sized,
              B2: SampleBorrow<Self::X> + Sized
    {
        UniformZ64(UniformInt::<u64>::new(low.borrow().0, high.borrow().0))
    }
    fn new_inclusive<B1, B2>(low: B1, high: B2) -> Self
        where B1: SampleBorrow<Self::X> + Sized,
              B2: SampleBorrow<Self::X> + Sized
    {
        UniformZ64(UniformInt::<u64>::new_inclusive(
            low.borrow().0,
            high.borrow().0,
        ))
    }
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::X {
        Z64(self.0.sample(rng))
    }
}

impl<const P: u64> SampleUniform for Z64<P> {
    type Sampler = UniformZ64<P>;
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

    #[test]
    fn gen_range() {
        use crate::Z64;
        let mut rng = rand_xoshiro::Xoshiro256StarStar::seed_from_u64(0);

        const P: u64 = PRIMES[2];
        const MIN: Z64<P> = Z64::new_unchecked(7);
        const MAX: Z64<P> = Z64::new_unchecked(153);

        for _ in 0..1000 {
            let z = rng.gen_range(MIN..MAX);
            assert!(z >= MIN);
            assert!(z < MAX);
        }
    }
}
