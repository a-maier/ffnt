use criterion::{criterion_group, criterion_main, Criterion, BatchSize};

use galois_fields::Z64;
use rand::{Rng, SeedableRng};

const PRIMES: [u64; 3] = [3, 443619635352171979, 1152921504606846883];

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
                                    rng.gen::<i64>().into(),
                                    rng.gen::<i64>().into()
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
                                    rng.gen::<i64>().into(),
                                    {
                                        let mut den: Z64<{ PRIMES[$x] }> = 0.into();
                                        while den == 0.into() {
                                            den = rng.gen::<i64>().into()
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


pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = rand_xoshiro::Xoshiro256StarStar::seed_from_u64(0);
    bench_new!(c, rng, 0, 1, 2);
    bench_op!(c, rng, +, 0, 1, 2);
    bench_op!(c, rng, -, 0, 1, 2);
    bench_op!(c, rng, *, 0, 1, 2);
    bench_div!(c, rng, 0, 1, 2);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
