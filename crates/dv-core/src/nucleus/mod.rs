//! Ports of the `third_party/nucleus/util/*` Python utility modules.
//!
//! Each submodule is a clean-room reimplementation of the public API surface
//! of its upstream counterpart, with tests reauthored from upstream's
//! `*_test.py` fixtures.

pub mod cigar;
pub mod ranges;
pub mod sequence_utils;
pub mod variant_utils;
pub mod variantcall_utils;
