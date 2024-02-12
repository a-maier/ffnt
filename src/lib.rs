//! This crate provides prime fields with a characteristic that fits
//! inside a 64 bit (or 32 bit) integer.
//!
//! It is mostly intended for number-theoretical applications. All
//! algorithms are taken from the [NTL library](https://libntl.org/),
//! but for performance and safety reasons the field characteristic is
//! set at compile time.
//!
//! *This crate is not suitable for applications in cryptography*
//!
//! # Usage
//!
//! Add this to your Cargo.toml:
//!
//! ```toml
//! [dependencies]
//! ffnt = "0.5"
//! ```
//!
//! # Example
//!
//! ```rust
//! use ffnt::Z64; // or Z32 if the characteristic fits inside 32 bits
//!
//! // the field characteristic
//! const P: u64 = 113;
//!
//! // sum up all elements of the field
//! let sum = (1..P).map(Z64::<P>::from)
//!    .reduce(|acc, t| acc + t)
//!    .unwrap();
//!
//! // check that the elements sum to 0
//! // if `num-traits` is enabled it is even better to use `sum.is_zero()`
//! assert_eq!(sum, Z64::<P>::from(0));
//! ```
//!
//! For more examples see [the examples
//! directory](https://github.com/a-maier/ffnt/tree/master/examples).
//!
//! # Features
//!
//! - `rand`: support for [random number generation](https://crates.io/crates/rand)
//! - `num-traits`: [numeric traits](https://crates.io/crates/num-traits)
//! - `serde`: [serialisation and deserialisation](https://crates.io/crates/serde)
#![warn(missing_docs)]
#[cfg(feature = "num-traits")]
/// Numeric traits
pub mod num_traits;
#[cfg(feature = "rand")]
/// Random field element generation
pub mod rand;
/// Fields with a 32 bit characteristic
pub mod z32;
/// Fields with a 64 bit characteristic
pub mod z64;

pub use z32::Z32;
pub use z64::TryDiv;
pub use z64::Z64;
