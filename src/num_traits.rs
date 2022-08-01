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

impl<const P: u64> Pow<i8> for Z64<P> {
    type Output = Self;

    fn pow(self, exp: i8) -> Self::Output {
        if exp < 0 {
            self.pow((-exp) as u8).inv()
        } else {
            self.pow(exp as u8)
        }
    }
}

impl<const P: u64> Pow<i16> for Z64<P> {
    type Output = Self;

    fn pow(self, exp: i16) -> Self::Output {
        if exp < 0 {
            self.pow((-exp) as u16).inv()
        } else {
            self.pow(exp as u16)
        }
    }
}

impl<const P: u64> Pow<i32> for Z64<P> {
    type Output = Self;

    fn pow(self, exp: i32) -> Self::Output {
        if exp < 0 {
            self.pow((-exp) as u32).inv()
        } else {
            self.pow(exp as u32)
        }
    }
}

impl<const P: u64> Pow<i64> for Z64<P> {
    type Output = Self;

    fn pow(self, exp: i64) -> Self::Output {
        if exp < 0 {
            self.pow((-exp) as u64).inv()
        } else {
            self.pow(exp as u64)
        }
    }
}

impl<const P: u64> Pow<i128> for Z64<P> {
    type Output = Self;

    fn pow(self, exp: i128) -> Self::Output {
        if exp < 0 {
            self.pow((-exp) as u128).inv()
        } else {
            self.pow(exp as u128)
        }
    }
}

// TODO: code duplication with Self::powu
macro_rules! impl_powu {
    ( $( $x:ty ),* ) => {
        $(
            impl<const P: u64> Pow<$x> for Z64<P> {
                type Output = Self;

                fn pow(mut self, mut exp: $x) -> Self::Output {
                    let mut res = Self::new_unchecked(1);
                    while exp > 0 {
                        if exp & 1 != 0 {
                            res *= self
                        };
                        self *= self;
                        exp /= 2;
                    }
                    res
                }
            }
        )*
    };
}

impl_powu!(u8, u16, u32, u64, u128);

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
