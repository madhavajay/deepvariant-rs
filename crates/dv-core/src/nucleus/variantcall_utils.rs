//! Port of `third_party/nucleus/util/variantcall_utils.py`. Helpers that
//! get/set named info fields on a `VariantCall` proto using the
//! Value/ListValue convention.

use dv_proto::nucleus_v1::{value, ListValue, Value, VariantCall};

pub fn set_int(call: &mut VariantCall, key: &str, n: i32) {
    call.info.insert(
        key.to_string(),
        ListValue {
            values: vec![Value {
                kind: Some(value::Kind::IntValue(n)),
            }],
        },
    );
}

pub fn set_int_list(call: &mut VariantCall, key: &str, ns: &[i32]) {
    call.info.insert(
        key.to_string(),
        ListValue {
            values: ns
                .iter()
                .map(|&n| Value {
                    kind: Some(value::Kind::IntValue(n)),
                })
                .collect(),
        },
    );
}

pub fn set_float(call: &mut VariantCall, key: &str, x: f64) {
    call.info.insert(
        key.to_string(),
        ListValue {
            values: vec![Value {
                kind: Some(value::Kind::NumberValue(x)),
            }],
        },
    );
}

pub fn set_float_list(call: &mut VariantCall, key: &str, xs: &[f64]) {
    call.info.insert(
        key.to_string(),
        ListValue {
            values: xs
                .iter()
                .map(|&x| Value {
                    kind: Some(value::Kind::NumberValue(x)),
                })
                .collect(),
        },
    );
}

pub fn set_string(call: &mut VariantCall, key: &str, s: &str) {
    call.info.insert(
        key.to_string(),
        ListValue {
            values: vec![Value {
                kind: Some(value::Kind::StringValue(s.to_string())),
            }],
        },
    );
}

pub fn get_int(call: &VariantCall, key: &str) -> Option<i32> {
    call.info.get(key).and_then(|lv| {
        lv.values.first().and_then(|v| match &v.kind {
            Some(value::Kind::IntValue(n)) => Some(*n),
            _ => None,
        })
    })
}

pub fn get_int_list(call: &VariantCall, key: &str) -> Vec<i32> {
    call.info
        .get(key)
        .map(|lv| {
            lv.values
                .iter()
                .filter_map(|v| match &v.kind {
                    Some(value::Kind::IntValue(n)) => Some(*n),
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default()
}

pub fn get_string(call: &VariantCall, key: &str) -> Option<String> {
    call.info.get(key).and_then(|lv| {
        lv.values.first().and_then(|v| match &v.kind {
            Some(value::Kind::StringValue(s)) => Some(s.clone()),
            _ => None,
        })
    })
}

pub fn set_gt(call: &mut VariantCall, gt: &[i32]) {
    call.genotype = gt.to_vec();
}

pub fn set_gq(call: &mut VariantCall, gq: i32) {
    set_int(call, "GQ", gq);
}

pub fn set_gl(call: &mut VariantCall, log10_gls: &[f64]) {
    call.genotype_likelihood = log10_gls.to_vec();
}

pub fn set_model_id(call: &mut VariantCall, model_id: &str) {
    set_string(call, "MID", model_id);
}

pub fn get_model_id(call: &VariantCall) -> Option<String> {
    get_string(call, "MID")
}

pub fn get_ad(call: &VariantCall) -> Vec<i32> {
    get_int_list(call, "AD")
}

pub fn get_dp(call: &VariantCall) -> Option<i32> {
    get_int(call, "DP")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_call() -> VariantCall {
        VariantCall::default()
    }

    #[test]
    fn int_round_trip() {
        let mut c = empty_call();
        set_int(&mut c, "DP", 42);
        assert_eq!(get_int(&c, "DP"), Some(42));
        assert_eq!(get_int(&c, "missing"), None);
    }

    #[test]
    fn int_list_round_trip() {
        let mut c = empty_call();
        set_int_list(&mut c, "AD", &[10, 20, 30]);
        assert_eq!(get_int_list(&c, "AD"), vec![10, 20, 30]);
    }

    #[test]
    fn string_round_trip() {
        let mut c = empty_call();
        set_string(&mut c, "MID", "deepvariant");
        assert_eq!(get_string(&c, "MID").as_deref(), Some("deepvariant"));
    }

    #[test]
    fn gt_and_gq() {
        let mut c = empty_call();
        set_gt(&mut c, &[0, 1]);
        set_gq(&mut c, 35);
        assert_eq!(c.genotype, vec![0, 1]);
        assert_eq!(get_int(&c, "GQ"), Some(35));
    }

    #[test]
    fn gl_sets_top_level_field() {
        let mut c = empty_call();
        set_gl(&mut c, &[-3.0, 0.0, -10.0]);
        assert_eq!(c.genotype_likelihood, vec![-3.0, 0.0, -10.0]);
    }

    #[test]
    fn model_id_aliases_mid_string() {
        let mut c = empty_call();
        set_model_id(&mut c, "small_model");
        assert_eq!(get_model_id(&c).as_deref(), Some("small_model"));
    }

    #[test]
    fn ad_dp_helpers() {
        let mut c = empty_call();
        set_int_list(&mut c, "AD", &[25, 30]);
        set_int(&mut c, "DP", 55);
        assert_eq!(get_ad(&c), vec![25, 30]);
        assert_eq!(get_dp(&c), Some(55));
    }
}
