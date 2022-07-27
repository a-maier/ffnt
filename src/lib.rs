use std::{ops::{Add, Sub, Neg, AddAssign, SubAssign, MulAssign, DivAssign, Mul, Div}, fmt::{Display, self}, cell::UnsafeCell};

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Z64<const P: u64> (
    u64
);

impl<const P: u64> Z64<P> {
    pub fn new(z: i64) -> Self {
        let res = remi(z, P, Self::info().red_struct);
        debug_assert!(res >= 0);
        let res = res as u64;
        debug_assert!(res < P);
        Self(res)
    }

    pub fn inv(&self) -> Self {
        let res = extended_gcd(self.0, Self::modulus());
        assert_eq!(res.gcd, 1, "inverse undefined for {}", self.0);
        let s = res.bezout[0];
        if s < 0 {
            debug_assert!(s + Self::modulus() as i64 >= 0);
            Self((s + Self::modulus() as i64) as u64)
        } else {
            Self(s as u64)
        }
    }

    fn info() -> Z64Info {
        // TODO: make this static as soon as possible
        // the problem is that rust 1.62 always returns the same instance,
        // even for different values for P
        thread_local!(static INFO: UnsafeCell<Z64Info> = Default::default());
        INFO.with(|f| {
            unsafe{
                let info = f.get();
                if (*info).p != P {
                    *info = Z64Info::new(P);
                };
                *info
            }
        })
    }

    pub const fn modulus() -> u64 {
        P
    }

    pub fn modulus_inv() -> SpInverse64 {
        Self::info().p_inv
    }
}

impl<const P: u64> From<Z64<P>> for u64 {
    fn from(i: Z64<P>) -> Self {
        i.0
    }
}

impl<const P: u64> From<Z64<P>> for i64 {
    fn from(i: Z64<P>) -> Self {
        i.0 as i64
    }
}

impl<const P: u64> From<u64> for Z64<P> {
    fn from(u: u64) -> Self {
        Self(remu(u, Self::modulus(), Self::info().red_struct) as u64)
    }
}

impl<const P: u64> From<i64> for Z64<P> {
    fn from(i: i64) -> Self {
        Self::new(i)
    }
}

impl<const P: u64> From<i32> for Z64<P> {
    fn from(i: i32) -> Self {
        Self::from(i as i64)
    }
}

impl<const P: u64> From<u32> for Z64<P> {
    fn from(u: u32) -> Self {
        Self::from(u as u64)
    }
}

impl<const P: u64> From<i16> for Z64<P> {
    fn from(i: i16) -> Self {
        Self::from(i as i64)
    }
}

impl<const P: u64> From<u16> for Z64<P> {
    fn from(u: u16) -> Self {
        Self::from(u as u64)
    }
}

impl<const P: u64> From<i8> for Z64<P> {
    fn from(i: i8) -> Self {
        Self::from(i as i64)
    }
}

impl<const P: u64> From<u8> for Z64<P> {
    fn from(u: u8) -> Self {
        Self::from(u as u64)
    }
}

impl<const P: u64> Display for Z64<P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<const P: u64>  AddAssign for Z64<P> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u64>  SubAssign for Z64<P> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u64>  MulAssign for Z64<P> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const P: u64>  DivAssign for Z64<P> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<const P: u64> Add for Z64<P> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let res = correct_excess((self.0 + rhs.0) as i64, Self::modulus());
        debug_assert!(res >= 0);
        let res = res as u64;
        debug_assert!(res < Self::modulus());
        Self(res)
    }
}

impl<const P: u64> Sub for Z64<P> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let res = correct_deficit(self.0 as i64 - rhs.0 as i64, Self::modulus());
        debug_assert!(res >= 0);
        let res = res as u64;
        debug_assert!(res < Self::modulus());
        Self(res)
    }
}

impl<const P: u64> Neg for Z64<P> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::default() - self
    }
}

impl<const P: u64> Mul for Z64<P> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(mul_mod(self.0, rhs.0, Self::modulus(), Self::modulus_inv()))
    }
}

impl<const P: u64> Div for Z64<P> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inv()
    }
}

fn mul_mod(a: u64, b: u64, n: u64, ninv: SpInverse64) -> u64 {
    let res = normalised_mul_mod(
        a,
        (b as i64) << ninv.shamt,
        ((n as i64) << ninv.shamt) as u64,
        ninv.inv
    ) >> ninv.shamt;
    res as u64
}

fn normalised_mul_mod(a: u64, b: i64, n: u64, ninv: u64) -> i64 {
    let u = a as u128 * b as u128;
    let h = (u >> (SP_NBITS - 2)) as u64;
    let q = u128_mul_high(h, ninv);
    // no shift needed, NTL_POST_SHIFT is 0
    let l = u as u64;
    let r = l.wrapping_sub(q.wrapping_mul(n));
    debug_assert!(r < 2 * n);
    correct_excess(r as i64, n)
}

fn remu(z: u64, p: u64, red: ReduceStruct) -> i64 {
    let q = u128_mul_high(z, red.ninv);
    let qp = q.wrapping_mul(p);
    let r = z as i64 - qp as i64;
    correct_excess(r as i64, p)
}

fn remi(z: i64, p: u64, red: ReduceStruct) -> i64 {
    let zu = (z as u64) & ((1u64 << (u64::BITS - 1)) - 1);
    let r = remu(zu, p, red);
    let s = z.sign_mask() & (red.sgn as i64);
    correct_deficit(r - s, p)
}

fn u128_mul_high(a: u64, b: u64) -> u64 {
    u128_get_high(a as u128 * b as u128)
}

fn u128_get_high(u: u128) -> u64 {
    (u >> u64::BITS) as u64
}

fn correct_excess(a: i64, p: u64) -> i64 {
    let n = p as i64;
    (a - n) + ((a - n).sign_mask() & n)
}

fn correct_deficit(a: i64, p: u64) -> i64 {
    a + (a.sign_mask() & (p as i64))
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct ExtendedGCDResult {
    gcd: u64,
    bezout: [i64; 2],
}

fn extended_gcd(a: u64, b: u64) -> ExtendedGCDResult {
    let mut old_r = a;
    let mut r = b;
    let mut old_s = 1;
    let mut s = 0;
    let mut old_t = 0;
    let mut t = 1;

    while r != 0 {
        let quotient = old_r / r;
        (old_r, r) = (r, old_r - quotient * r);
        (old_s, s) = (s, old_s - quotient as i64 * s);
        (old_t, t) = (t, old_t - quotient as i64 * t);
    }
    ExtendedGCDResult { gcd: old_r, bezout: [old_s, old_t] }
}

const SP_NBITS: u32 = u64::BITS - 4;
const PRE_SHIFT2: u32 = 2 * SP_NBITS + 2;

// // NTL: absolute maximum root bound for FFT primes
// const FFT_MAX_ROOT_BND: u32 = SP_NBITS - 2;

// const FFT_MAX_ROOT: u32 = 25;

const fn used_bits(z: u64) -> u32 {
    u64::BITS - z.leading_zeros()
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct Z64Info {
    p: u64,
    p_inv: SpInverse64,
    red_struct: ReduceStruct,
    u128_red_struct: U128ReduceStruct,
}

impl Z64Info {
    fn new(p: u64) -> Self {
        assert!(p > 1);
        assert!(used_bits(p) <= SP_NBITS);

        let p_inv = prep_mul_mod(p);
        let red_struct = prep_rem(p);
        let u128_red_struct = U128ReduceStruct::new(p);
        // TODO: ZZ_red_struct?
        Self{
            p,
            p_inv,
            red_struct,
            u128_red_struct
        }
    }
}

fn prep_mul_mod(p: u64) -> SpInverse64 {
    let shamt = p.leading_zeros() - (u64::BITS - SP_NBITS);
    let inv = normalized_prep_mul_mod(p << shamt);
    SpInverse64{inv, shamt}
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct ReduceStruct {
    ninv: u64,
    sgn: u64,
}

fn prep_rem(p: u64) -> ReduceStruct {
    let mut q = (1 << (u64::BITS - 1)) / p;
    // r = 2^63 % p
    let r = (1 << (u64::BITS - 1)) - q * p;

    q *= 2;
    let _ = correct_excess_quo(&mut q, 2 * r as i64, p as i64);

    ReduceStruct { ninv: q, sgn: r }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct U128ReduceStruct {
    inv: u64,
    nbits: u32,
}

impl U128ReduceStruct {
    fn new(n: u64) -> Self {
        let nbits = used_bits(n);
        let inv = ((1u128 << (nbits + u64::BITS)) - 1) / (n as u128);
        U128ReduceStruct { inv: inv as u64, nbits }
    }
}

fn correct_excess_quo(q: &mut u64, a: i64, n: i64) -> i64 {
    if a >= n {
        *q += 1;
        a - n
    } else {
        a
    }
}

trait SignMask{
    fn sign_mask(self) -> i64;
}

impl SignMask for u64 {
    fn sign_mask(self) -> i64 {
        (self as i64).sign_mask()
    }
}

impl SignMask for i64 {
    fn sign_mask(self) -> i64 {
        self >> (u64::BITS - 1)
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SpInverse64 {
    inv: u64,
    shamt: u32,
}

fn normalized_prep_mul_mod(n: u64) -> u64 {
    let ninv = 1. / (n as f64);

    // NOTE: this is an initial approximation
    //       the true quotient is <= 2^SP_NBITS
    let init_quot_approx =
           ((1u64 << (SP_NBITS - 1)) as f64 * (1u64 << SP_NBITS) as f64 * ninv) as u64;

    let wot = 1u128 << (2 * SP_NBITS - 1);
    let approx_rem = wot.wrapping_sub(n as u128 * init_quot_approx as u128);

    let approx_rem = (approx_rem << (PRE_SHIFT2 - 2 * SP_NBITS + 1)) - 1;

    // now compute a floating point approximation to the remainder,
    // but avoiding unsigned -> float conversions,
    // as these are not as well supported in hardware as
    // signed -> float conversions

    let approx_rem_low = approx_rem as u64;
    let approx_rem_high = (approx_rem >> u64::BITS) as u64
        + (approx_rem_low >> (u64::BITS - 1));

    let approx_rem_low = approx_rem_low as i64;
    let approx_rem_high = approx_rem_high as i64;

    let bpl = (1u64 << SP_NBITS) as f64 * (1u64 << (u64::BITS - SP_NBITS)) as f64;

    let fr = approx_rem_low as f64 + approx_rem_high as f64 * bpl;

    // now convert fr*ninv to a long
    // but we have to be careful: fr may be negative.
    // the result should still give floor(r/n) pm 1,
    // and is computed in a way that avoids branching

    let mut q1 = (fr * ninv) as i64;
    if q1 < 0 {
        // This counteracts the round-to-zero behavior of conversion
        // to i64.  It should be compiled into branch-free code.
        q1 -= 1
    }

    let mut q1 = q1 as u64;
    let approx_rem_low = approx_rem_low as u64;
    let sub = q1.wrapping_mul(n);

    let approx_rem = approx_rem_low.wrapping_sub(sub);

    q1 += (1 + approx_rem.sign_mask() + approx_rem.wrapping_sub(n).sign_mask()) as u64;

    (init_quot_approx << (PRE_SHIFT2 - 2 * SP_NBITS + 1)).wrapping_add(q1)

    // NTL_PRE_SHIFT1 is 0, so no further shift required
}

#[cfg(test)]
mod tests {

    use once_cell::sync::Lazy;
    use rand::{SeedableRng, Rng};

    use super::*;

    const PRIMES: [u64; 3] = [
        3,
        443619635352171979,
        1152921504606846883,
    ];

    #[test]
    fn z64_info() {
        let tst = Z64::<{PRIMES[0]}>::info();
        assert_eq!(Z64::<{PRIMES[0]}>::modulus(), PRIMES[0]);
        assert_eq!(tst.p_inv.inv, 6148914691236517205);
        assert_eq!(tst.p_inv.shamt, 58);
        assert_eq!(tst.red_struct.ninv, 6148914691236517205);
        assert_eq!(tst.red_struct.sgn, 2);
        assert_eq!(tst.u128_red_struct.inv, 6148914691236517205);
        assert_eq!(tst.u128_red_struct.nbits, 2);
        // TODO: ZZ_red_struct

        let tst = Z64::<{PRIMES[1]}>::info();
        assert_eq!(Z64::<{PRIMES[1]}>::modulus(), PRIMES[1]);
        assert_eq!(tst.p_inv.inv, 5992647258409536587);
        assert_eq!(tst.p_inv.shamt, 1);
        assert_eq!(tst.red_struct.ninv, 41);
        assert_eq!(tst.red_struct.sgn, 350979329811336228);
        assert_eq!(tst.u128_red_struct.inv, 5523844959928594733);
        assert_eq!(tst.u128_red_struct.nbits, 59);

        let tst = Z64::<{PRIMES[2]}>::info();
        assert_eq!(Z64::<{PRIMES[2]}>::modulus(), PRIMES[2]);
        assert_eq!(tst.p_inv.inv, 4611686018427388276);
        assert_eq!(tst.p_inv.shamt, 0);
        assert_eq!(tst.red_struct.ninv, 16);
        assert_eq!(tst.red_struct.sgn, 744);
        assert_eq!(tst.u128_red_struct.inv, 1488);
        assert_eq!(tst.u128_red_struct.nbits, 60);
    }

    #[test]
    fn z64_constr() {
        let z: Z64<3> = 2.into();
        assert_eq!(u64::from(z), 2);
        let z: Z64<3> = (-1).into();
        assert_eq!(u64::from(z), 2);
        let z: Z64<3> = 5.into();
        assert_eq!(u64::from(z), 2);

        let z: Z64<3> = 0.into();
        assert_eq!(u64::from(z), 0);
        let z: Z64<3> = 3.into();
        assert_eq!(u64::from(z), 0);

        let z: Z64<3> = 2u32.into();
        assert_eq!(u64::from(z), 2);
        let z: Z64<3> = 5u32.into();
        assert_eq!(u64::from(z), 2);

        let z: Z64<3> = 0u32.into();
        assert_eq!(u64::from(z), 0);
        let z: Z64<3> = 3u32.into();
        assert_eq!(u64::from(z), 0);
    }

    static POINTS: Lazy<[i64; 1000]> = Lazy::new(|| {
        let mut pts = [0; 1000];
        let mut rng = rand_xoshiro::Xoshiro256StarStar::seed_from_u64(0);
        for pt in &mut pts {
            *pt = rng.gen();
        }
        pts
    });

    #[test]
    fn tst_conv() {
        // separate for loops are faster because we don't have to recalculate
        // the thread local Z64Info in each iteration
        for pt in *POINTS {
            let z: Z64<{PRIMES[0]}> = pt.into();
            let z: i64 = z.into();
            assert_eq!(z, pt.rem_euclid(PRIMES[0] as i64));
        }

        for pt in *POINTS {
            let z: Z64<{PRIMES[1]}> = pt.into();
            let z: i64 = z.into();
            assert_eq!(z, pt.rem_euclid(PRIMES[1] as i64));
        }

        for pt in *POINTS {
            let z: Z64<{PRIMES[2]}> = pt.into();
            let z: i64 = z.into();
            assert_eq!(z, pt.rem_euclid(PRIMES[2] as i64));
        }
    }

    #[test]
    fn tst_add() {
        for pt1 in *POINTS {
            let z1: Z64<{PRIMES[0]}> = pt1.into();
            let pt1 = pt1 as i128;
            for pt2 in *POINTS {
                let z2: Z64<{PRIMES[0]}> = pt2.into();
                let pt2 = pt2 as i128;
                let sum1: i64 = (z1 + z2).into();
                let sum2 = (pt1 + pt2).rem_euclid(PRIMES[0] as i128) as i64;
                assert_eq!(sum1, sum2);
            }
        }

        for pt1 in *POINTS {
            let z1: Z64<{PRIMES[1]}> = pt1.into();
            let pt1 = pt1 as i128;
            for pt2 in *POINTS {
                let z2: Z64<{PRIMES[1]}> = pt2.into();
                let pt2 = pt2 as i128;
                let sum1: i64 = (z1 + z2).into();
                let sum2 = (pt1 + pt2).rem_euclid(PRIMES[1] as i128) as i64;
                assert_eq!(sum1, sum2);
            }
        }

        for pt1 in *POINTS {
            let z1: Z64<{PRIMES[2]}> = pt1.into();
            let pt1 = pt1 as i128;
            for pt2 in *POINTS {
                let z2: Z64<{PRIMES[2]}> = pt2.into();
                let pt2 = pt2 as i128;
                let sum1: i64 = (z1 + z2).into();
                let sum2 = (pt1 + pt2).rem_euclid(PRIMES[2] as i128) as i64;
                assert_eq!(sum1, sum2);
            }
        }
    }

    #[test]
    fn tst_sub() {
        for pt1 in *POINTS {
            let z1: Z64<{PRIMES[0]}> = pt1.into();
            let pt1 = pt1 as i128;
            for pt2 in *POINTS {
                let z2: Z64<{PRIMES[0]}> = pt2.into();
                let pt2 = pt2 as i128;
                let sum1: i64 = (z1 - z2).into();
                let sum2 = (pt1 - pt2).rem_euclid(PRIMES[0] as i128) as i64;
                assert_eq!(sum1, sum2);
            }
        }

        for pt1 in *POINTS {
            let z1: Z64<{PRIMES[1]}> = pt1.into();
            let pt1 = pt1 as i128;
            for pt2 in *POINTS {
                let z2: Z64<{PRIMES[1]}> = pt2.into();
                let pt2 = pt2 as i128;
                let sum1: i64 = (z1 - z2).into();
                let sum2 = (pt1 - pt2).rem_euclid(PRIMES[1] as i128) as i64;
                assert_eq!(sum1, sum2);
            }
        }

        for pt1 in *POINTS {
            let z1: Z64<{PRIMES[2]}> = pt1.into();
            let pt1 = pt1 as i128;
            for pt2 in *POINTS {
                let z2: Z64<{PRIMES[2]}> = pt2.into();
                let pt2 = pt2 as i128;
                let sum1: i64 = (z1 - z2).into();
                let sum2 = (pt1 - pt2).rem_euclid(PRIMES[2] as i128) as i64;
                assert_eq!(sum1, sum2);
            }
        }
    }

    #[test]
    fn tst_mul() {
        for pt1 in *POINTS {
            let z1: Z64<{PRIMES[0]}> = pt1.into();
            let pt1 = pt1 as i128;
            for pt2 in *POINTS {
                let z2: Z64<{PRIMES[0]}> = pt2.into();
                let pt2 = pt2 as i128;
                let prod1: i64 = (z1 * z2).into();
                let prod2 = (pt1 * pt2).rem_euclid(PRIMES[0] as i128) as i64;
                assert_eq!(prod1, prod2);
            }
        }

        for pt1 in *POINTS {
            let z1: Z64<{PRIMES[1]}> = pt1.into();
            let pt1 = pt1 as i128;
            for pt2 in *POINTS {
                let z2: Z64<{PRIMES[1]}> = pt2.into();
                let pt2 = pt2 as i128;
                let prod1: i64 = (z1 * z2).into();
                let prod2 = (pt1 * pt2).rem_euclid(PRIMES[1] as i128) as i64;
                assert_eq!(prod1, prod2);
            }
        }

        for pt1 in *POINTS {
            let z1: Z64<{PRIMES[2]}> = pt1.into();
            let pt1 = pt1 as i128;
            for pt2 in *POINTS {
                let z2: Z64<{PRIMES[2]}> = pt2.into();
                let pt2 = pt2 as i128;
                let prod1: i64 = (z1 * z2).into();
                let prod2 = (pt1 * pt2).rem_euclid(PRIMES[2] as i128) as i64;
                assert_eq!(prod1, prod2);
            }
        }
    }

    #[test]
    fn tst_div() {
        for pt1 in *POINTS {
            let z1: Z64<{PRIMES[0]}> = pt1.into();
            for pt2 in *POINTS {
                let z2: Z64<{PRIMES[0]}> = pt2.into();
                if i64::from(z2) == 0 {
                    continue;
                }
                let div = z1 / z2;
                assert_eq!(z1, div * z2);
            }
        }

        for pt1 in *POINTS {
            let z1: Z64<{PRIMES[1]}> = pt1.into();
            for pt2 in *POINTS {
                let z2: Z64<{PRIMES[1]}> = pt2.into();
                if i64::from(z2) == 0 {
                    continue;
                }
                let div = z1 / z2;
                assert_eq!(z1, div * z2);
            }
        }

        for pt1 in *POINTS {
            let z1: Z64<{PRIMES[2]}> = pt1.into();
            for pt2 in *POINTS {
                let z2: Z64<{PRIMES[2]}> = pt2.into();
                if i64::from(z2) == 0 {
                    continue;
                }
                let div = z1 / z2;
                assert_eq!(z1, div * z2);
            }
        }
    }
}
