//! Phred / probability conversions.
//!
//! Direct port of `third_party/nucleus/util/math.{h,cc}`. Same numerical
//! semantics — pError vs pTrue, real-space vs log10-space vs Phred-scale.

/// `phred = -10 * log10(perror)`
#[inline]
pub fn phred_to_perror(phred: i32) -> f64 {
    debug_assert!(phred >= 0);
    10f64.powf(-(phred as f64) / 10.0)
}

#[inline]
pub fn phred_to_log10_perror(phred: i32) -> f64 {
    debug_assert!(phred >= 0);
    -(phred as f64) / 10.0
}

#[inline]
pub fn perror_to_log10_perror(perror: f64) -> f64 {
    debug_assert!(perror > 0.0 && perror <= 1.0);
    perror.log10()
}

#[inline]
pub fn log10_perror_to_phred(log10_perror: f64) -> f64 {
    debug_assert!(log10_perror <= 0.0);
    -10.0 * log10_perror
}

#[inline]
pub fn log10_perror_to_rounded_phred(log10_perror: f64) -> i32 {
    log10_perror_to_phred(log10_perror).round().abs() as i32
}

#[inline]
pub fn perror_to_phred(perror: f64) -> f64 {
    log10_perror_to_phred(perror_to_log10_perror(perror))
}

#[inline]
pub fn perror_to_rounded_phred(perror: f64) -> i32 {
    log10_perror_to_rounded_phred(perror_to_log10_perror(perror))
}

/// Maximum confidence (matches Python `_MAX_CONFIDENCE = 1.0 - 1.25e-10`,
/// giving a phred ceiling of ~99).
pub const MAX_CONFIDENCE: f64 = 1.0 - 1.25e-10;

/// Phred-scaled `1 - ptrue`, capped at `_MAX_CONFIDENCE`.
#[inline]
pub fn ptrue_to_bounded_phred(ptrue: f64) -> f64 {
    debug_assert!((0.0..=1.0).contains(&ptrue));
    perror_to_phred(1.0 - ptrue.min(MAX_CONFIDENCE))
}

/// `log10(p)` with `p` floor-clamped at `1 - MAX_CONFIDENCE`.
#[inline]
pub fn perror_to_bounded_log10_perror(perror: f64) -> f64 {
    debug_assert!((0.0..=1.0).contains(&perror));
    let min_prob = 1.0 - MAX_CONFIDENCE;
    perror_to_log10_perror(perror.max(min_prob))
}

/// Subtract the max from each likelihood so the most likely is 0
/// (improves numerical resolution downstream).
pub fn zero_shift_log10_likelihoods(likelihoods: &[f64]) -> Vec<f64> {
    let max = likelihoods.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    likelihoods.iter().map(|x| x - max).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn phred_to_perror_basics() {
        assert!(close(phred_to_perror(0), 1.0, 1e-12));
        assert!(close(phred_to_perror(10), 0.1, 1e-12));
        assert!(close(phred_to_perror(20), 0.01, 1e-12));
        assert!(close(phred_to_perror(30), 0.001, 1e-12));
    }

    #[test]
    fn perror_to_phred_inverse() {
        for &phred in &[0, 1, 10, 20, 30, 50, 99] {
            let p = phred_to_perror(phred);
            assert!(close(perror_to_phred(p), phred as f64, 1e-9));
        }
    }

    #[test]
    fn ptrue_to_bounded_phred_caps_at_99() {
        let phred = ptrue_to_bounded_phred(1.0);
        assert!(phred >= 98.0 && phred <= 100.0, "phred={phred}");
    }

    #[test]
    fn ptrue_to_bounded_phred_zero() {
        let phred = ptrue_to_bounded_phred(0.0);
        assert!(close(phred, 0.0, 1e-9));
    }

    #[test]
    fn zero_shift_centers_max_at_zero() {
        let shifted = zero_shift_log10_likelihoods(&[-3.0, -1.0, -2.0]);
        assert_eq!(shifted, vec![-2.0, 0.0, -1.0]);
    }
}
