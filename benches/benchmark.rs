// run with `cargo criterion --features rand,num-traits`
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

use ffnt::{Z32, Z64};
use num_traits::Zero;
use rand::{Rng, SeedableRng};

const PRIMES_32: [u32; 3] = [3, 65521, 1073741789];
const PRIMES: [u64; 3] = [3, 443619635352171979, 1152921504606846883];

// TODO: refactor as soon as
// https://github.com/rust-lang/rust/issues/35853
// is resolved

macro_rules! bench_new {
    ( $c:ident, $rng:ident, $( $x:literal ),* ) => {
        $(
            {
                let rng = &mut $rng;
                $c.bench_function(
                    &format!("new mod {}", PRIMES[$x]),
                    move |b| {
                        b.iter_batched(
                            || rng.gen(),
                            |z| Z64::<{ PRIMES[$x] }>::new(z),
                            BatchSize::SmallInput,
                        )
                    });
            }
        )*
    };
}

macro_rules! bench_op {
    ( $c:ident, $rng:ident, $op:tt, $( $x:literal ),* ) => {
        $(
            {
                let rng = &mut $rng;
                $c.bench_function(
                    &format!("{} mod {}", stringify!($op), PRIMES[$x]),
                    move |b| {
                        b.iter_batched(
                            || {
                                let res: [Z64<{ PRIMES[$x] }>; 2] = [
                                    rng.gen(),
                                    rng.gen()
                                ];
                                res
                            },
                            |z| z[0] $op z[1],
                            BatchSize::SmallInput,
                        )
                    });
            }
        )*
    };
}

macro_rules! bench_div {
    ( $c:ident, $rng:ident, $( $x:literal ),* ) => {
        $(
            {
                let rng = &mut $rng;
                $c.bench_function(
                    &format!("/ mod {}", PRIMES[$x]),
                    move |b| {
                        b.iter_batched(
                            || {
                                let res: [Z64<{ PRIMES[$x] }>; 2] = [
                                    rng.gen(),
                                    {
                                        let mut den: Z64<{ PRIMES[$x] }> = Zero::zero();
                                        while den.is_zero() {
                                            den = rng.gen()
                                        }
                                        den
                                    }
                                ];
                                res
                            },
                            |z| z[0] / z[1],
                            BatchSize::SmallInput,
                        )
                    });
            }
        )*
    };
}

macro_rules! bench_pow {
    ( $c:ident, $rng:ident, $( $x:literal ),* ) => {
        $(
            {
                let rng = &mut $rng;
                $c.bench_function(
                    &format!("^ mod {}", PRIMES[$x]),
                    move |b| {
                        b.iter_batched(
                            || {
                                let mut base: Z64<{ PRIMES[$x] }> = Zero::zero();
                                let mut exp: i64 = -1;
                                while base.is_zero() && exp < 0 {
                                    base = rng.gen();
                                    exp = rng.gen();
                                }
                                (base, exp)
                            },
                            |(base, exp)|  base.powi(exp),
                            BatchSize::SmallInput,
                        )
                    });
            }
        )*
    };
}

macro_rules! bench_new_32 {
    ( $c:ident, $rng:ident, $( $x:literal ),* ) => {
        $(
            {
                let rng = &mut $rng;
                $c.bench_function(
                    &format!("new mod {} (32 bits)", PRIMES_32[$x]),
                    move |b| {
                        b.iter_batched(
                            || rng.gen(),
                            |z| Z32::<{ PRIMES_32[$x] }>::new(z),
                            BatchSize::SmallInput,
                        )
                    });
            }
        )*
    };
}

macro_rules! bench_op_32 {
    ( $c:ident, $rng:ident, $op:tt, $( $x:literal ),* ) => {
        $(
            {
                let rng = &mut $rng;
                $c.bench_function(
                    &format!("{} mod {} (32 bits)", stringify!($op), PRIMES_32[$x]),
                    move |b| {
                        b.iter_batched(
                            || {
                                let res: [Z32<{ PRIMES_32[$x] }>; 2] = [
                                    rng.gen(),
                                    rng.gen()
                                ];
                                res
                            },
                            |z| z[0] $op z[1],
                            BatchSize::SmallInput,
                        )
                    });
            }
        )*
    };
}

macro_rules! bench_div_32 {
    ( $c:ident, $rng:ident, $( $x:literal ),* ) => {
        $(
            {
                let rng = &mut $rng;
                $c.bench_function(
                    &format!("/ mod {} (32 bits)", PRIMES_32[$x]),
                    move |b| {
                        b.iter_batched(
                            || {
                                let res: [Z32<{ PRIMES_32[$x] }>; 2] = [
                                    rng.gen(),
                                    {
                                        let mut den: Z32<{ PRIMES_32[$x] }> = Zero::zero();
                                        while den.is_zero() {
                                            den = rng.gen()
                                        }
                                        den
                                    }
                                ];
                                res
                            },
                            |z| z[0] / z[1],
                            BatchSize::SmallInput,
                        )
                    });
            }
        )*
    };
}

macro_rules! bench_pow_32 {
    ( $c:ident, $rng:ident, $( $x:literal ),* ) => {
        $(
            {
                let rng = &mut $rng;
                $c.bench_function(
                    &format!("^ mod {} (32 bits)", PRIMES_32[$x]),
                    move |b| {
                        b.iter_batched(
                            || {
                                let mut base: Z32<{ PRIMES_32[$x] }> = Zero::zero();
                                let mut exp: i64 = -1;
                                while base.is_zero() && exp < 0 {
                                    base = rng.gen();
                                    exp = rng.gen();
                                }
                                (base, exp)
                            },
                            |(base, exp)|  base.powi(exp),
                            BatchSize::SmallInput,
                        )
                    });
            }
        )*
    };
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = rand_xoshiro::Xoshiro256StarStar::seed_from_u64(0);
    bench_new_32!(c, rng, 0, 1, 2);
    bench_new!(c, rng, 0, 1, 2);
    bench_op_32!(c, rng, +, 0, 1, 2);
    bench_op!(c, rng, +, 0, 1, 2);
    bench_op_32!(c, rng, -, 0, 1, 2);
    bench_op!(c, rng, -, 0, 1, 2);
    bench_op_32!(c, rng, *, 0, 1, 2);
    bench_op!(c, rng, *, 0, 1, 2);
    bench_div_32!(c, rng, 0, 1, 2);
    bench_div!(c, rng, 0, 1, 2);
    bench_pow_32!(c, rng, 0, 1, 2);
    bench_pow!(c, rng, 0, 1, 2);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
