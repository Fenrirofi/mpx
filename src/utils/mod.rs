#[inline(always)]
pub const fn align_up(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) & !(alignment - 1)
}

#[inline(always)]
pub const fn align_uniform(value: u64) -> u64 {
    align_up(value, 256)
}