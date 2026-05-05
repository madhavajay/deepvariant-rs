//! Options for the pileup image generator. Defaults mirror upstream
//! `pileup_image.py`'s `default_options()` for WGS.

#[derive(Debug, Clone, Copy)]
pub struct PileupOptions {
    pub base_color_offset_a_and_g: i32,
    pub base_color_offset_t_and_c: i32,
    pub base_color_stride: i32,
    pub reference_base_quality: i32,
    pub base_quality_cap: i32,
    pub mapping_quality_cap: i32,
    pub positive_strand_color: i32,
    pub negative_strand_color: i32,
    pub allele_supporting_read_alpha: f32,
    pub allele_unsupporting_read_alpha: f32,
    pub other_allele_supporting_read_alpha: f32,
    pub reference_matching_read_alpha: f32,
    pub reference_mismatching_read_alpha: f32,
    pub reference_alpha: f32,
    pub width: usize,
    pub height: usize,
}

impl Default for PileupOptions {
    fn default() -> Self {
        Self {
            base_color_offset_a_and_g: 40,
            base_color_offset_t_and_c: 30,
            base_color_stride: 70,
            reference_base_quality: 60,
            base_quality_cap: 40,
            mapping_quality_cap: 60,
            positive_strand_color: 70,
            negative_strand_color: 240,
            allele_supporting_read_alpha: 1.0,
            allele_unsupporting_read_alpha: 0.6,
            other_allele_supporting_read_alpha: 0.6,
            reference_matching_read_alpha: 0.2,
            reference_mismatching_read_alpha: 1.0,
            reference_alpha: 0.4,
            width: 221,
            height: 100,
        }
    }
}
