use std::ops::Rem;

use crate::Z64;

use num_traits::{AsPrimitive, Bounded, One, Inv, Pow, Unsigned, Num, Zero};

impl<const P: u64> Bounded for Z64<P> {
    fn min_value() -> Self {
        Self::zero()
    }

    fn max_value() -> Self {
        Self::new_unchecked(P - 1)
    }
}

impl<const P: u64> Zero for Z64<P> {
    fn zero() -> Self {
        Self::new_unchecked(0)
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<const P: u64> One for Z64<P> {
    fn one() -> Self {
        Self::new_unchecked(1)
    }

    fn is_one(&self) -> bool {
        self.0.is_one()
    }
}

impl<const P: u64> Inv for Z64<P> {
    type Output = Self;

    fn inv(self) -> Self {
        Z64::<P>::inv(&self)
    }
}

impl<const P: u64> Unsigned for Z64<P> { }

impl<const P: u64> Num for Z64<P> {
    type FromStrRadixErr = <u64 as Num>::FromStrRadixErr;

    fn from_str_radix(
        str: &str,
        radix: u32
    ) -> Result<Self, Self::FromStrRadixErr> {
        Ok(Self::new(i64::from_str_radix(str, radix)?))
    }
}

impl<const P: u64> Rem for Z64<P> {
    type Output = Self;

    fn rem(self, _modulus: Self) -> Self::Output {
        Self::zero()
    }
}

macro_rules! impl_pow {
    ( $( $x:ty ),* ) => {
        $(
            impl<const P: u64> Pow<$x> for Z64<P> {
                type Output = Self;

                fn pow(self, _rhs: $x) -> Self::Output {
                    todo!()
                }
            }
        )*
    };
}

impl_pow!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

macro_rules! impl_as_primitive {
    ( $( $x:ty ),* ) => {
        $(
            impl<const P: u64> AsPrimitive<$x> for Z64<P> {
                fn as_(self) -> $x {
                    self.0 as $x
                }
            }
        )*
    };
}

impl_as_primitive!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

#[cfg(test)]
mod tests {

    use super::*;

    const PRIMES: [u64; 3] = [3, 443619635352171979, 1152921504606846883];

    #[test]
    fn bounded() {
        use crate::Z64;
        assert_eq!(Z64::<{PRIMES[0]}>::min_value(), Z64::<{PRIMES[0]}>::new(0));
        assert_eq!(Z64::<{PRIMES[0]}>::max_value(), Z64::<{PRIMES[0]}>::new(-1));
        assert_eq!(Z64::<{PRIMES[1]}>::min_value(), Z64::<{PRIMES[1]}>::new(0));
        assert_eq!(Z64::<{PRIMES[1]}>::max_value(), Z64::<{PRIMES[1]}>::new(-1));
        assert_eq!(Z64::<{PRIMES[2]}>::min_value(), Z64::<{PRIMES[2]}>::new(0));
        assert_eq!(Z64::<{PRIMES[2]}>::max_value(), Z64::<{PRIMES[2]}>::new(-1));
    }

}
