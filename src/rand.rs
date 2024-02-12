use crate::{Z32, Z64};

use paste::paste;
use rand::distributions::{
    uniform::{SampleBorrow, SampleUniform, UniformInt, UniformSampler},
    Distribution, Standard,
};
use rand::Rng;

macro_rules! impl_rand {
    ( $($z:literal), *) => {
        $(
            paste!{
                impl<const P: [<u $z>]> Distribution<[<Z $z>]<P>> for Standard {
                    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> [<Z $z>]<P> {
                        let num = rng.gen_range(0..P);
                        unsafe { [<Z $z>]::new_unchecked(num) }
                    }
                }

                #[allow(missing_docs)]
                #[derive(Clone, Copy, Debug)]
                pub struct [<UniformZ $z>]<const P: [<u $z>]>(UniformInt<[<u $z>]>);

                impl<const P: [<u $z>]> UniformSampler for [<UniformZ $z>]<P> {
                    type X = [<Z $z>]<P>;

                    fn new<B1, B2>(low: B1, high: B2) -> Self
                    where B1: SampleBorrow<Self::X> + Sized,
                          B2: SampleBorrow<Self::X> + Sized
                    {
                        [<UniformZ $z>](
                            UniformInt::<[<u $z>]>::new(
                                low.borrow().repr(),
                                high.borrow().repr()
                            )
                        )
                    }
                    fn new_inclusive<B1, B2>(low: B1, high: B2) -> Self
                    where B1: SampleBorrow<Self::X> + Sized,
                          B2: SampleBorrow<Self::X> + Sized
                    {
                        [<UniformZ $z>](UniformInt::<[<u $z>]>::new_inclusive(
                            low.borrow().repr(),
                            high.borrow().repr(),
                        ))
                    }
                    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::X {
                        unsafe { [<Z $z>]::new_unchecked(self.0.sample(rng)) }
                    }
                }

                impl<const P: [<u $z>]> SampleUniform for [<Z $z>]<P> {
                    type Sampler = [<UniformZ $z>]<P>;
                }
            }
        )*
    }
}

impl_rand! {32, 64}

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
        const MIN: Z64<P> = Z64::new(7);
        const MAX: Z64<P> = Z64::new(153);

        for _ in 0..1000 {
            let z = rng.gen_range(MIN..MAX);
            assert!(z >= MIN);
            assert!(z < MAX);
        }
    }
}
