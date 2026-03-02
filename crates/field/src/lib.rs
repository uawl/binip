pub mod traits;
pub mod gf2_32;
pub mod gf2_64;
pub mod gf2_128;

pub use gf2_32::GF2_32;
pub use gf2_64::GF2_64;
pub use gf2_128::GF2_128;
pub use traits::FieldElem;
