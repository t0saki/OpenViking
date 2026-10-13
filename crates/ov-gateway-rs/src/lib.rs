//! Experimental Rust port of the OpenViking Gateway hot path (Anthropic Messages,
//! streaming), written to estimate what a Rust worker kernel would gain. Not a product:
//! see BENCH.md for what is ported, what is stubbed, and the measurements.

pub mod anthropic;
pub mod canon;
pub mod kernel;
pub mod ov;
pub mod proxy;
pub mod store;
pub mod text;
