use std::{
    fmt::{self, Display},
    num::IntErrorKind,
};

/// An error which can be returned when parsing a finite-field integer.
///
/// This error type closely mirrors `std::num::ParseIntError`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParseIntError(IntErrorKind);

impl ParseIntError {
    /// Outputs the detailed cause of parsing an integer failing.
    pub fn kind(&self) -> &IntErrorKind {
        &self.0
    }

    fn description(&self) -> &str {
        match self.0 {
            IntErrorKind::Empty => "cannot parse integer from empty string",
            IntErrorKind::InvalidDigit => "invalid digit found in string",
            IntErrorKind::PosOverflow => {
                "number too large to fit in target type"
            }
            IntErrorKind::NegOverflow => {
                "number too small to fit in target type"
            }
            IntErrorKind::Zero => "number would be zero for non-zero type",
            _ => "unknown error",
        }
    }
}

impl Display for ParseIntError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.description().fmt(f)
    }
}

impl From<std::num::ParseIntError> for ParseIntError {
    fn from(source: std::num::ParseIntError) -> Self {
        Self(source.kind().clone())
    }
}

impl From<IntErrorKind> for ParseIntError {
    fn from(source: IntErrorKind) -> Self {
        Self(source)
    }
}

impl std::error::Error for ParseIntError {}
