fn convert_base(mut number: u32, base: u32, add: u32) -> [u8; 4] {
    let mut out = [0; 4];
    out[0] = (number % base + add) as u8;
    number /= base;
    out[1] = (number % base + add) as u8;
    number /= base;
    out[2] = (number % base + add) as u8;
    number /= base;
    out[3] = (number % base + add) as u8;
    out
}
pub fn convert_ascii(number: u32) -> [u8; 4] {
    convert_base(number, 0x80, 0)
}
pub fn convert_all(number: u32) -> [u8; 4] {
    number.to_le_bytes()
}
pub fn convert_password(number: u32) -> [u8; 4] {
    convert_base(number, 0x5d, 0x21)
}
pub fn convert_alphanumeric(number: u32) -> [u8; 4] {
    let mut out = convert_base(number, 0x3e, 0x31);
    out.iter_mut().for_each(|o| {
        if *o > 0x39 {
            *o += 7;
        }
        if *o > 0x5a {
            *o += 6;
        }
    });
    out
}
pub fn convert_alpha(number: u32) -> [u8; 4] {
    let mut out = convert_base(number, 0x34, 0x41);
    out.iter_mut().for_each(|o| {
        if *o > 0x5a {
            *o += 6;
        }
    });
    out
}
pub fn convert_numeric(number: u32) -> [u8; 4] {
    convert_base(number, 0xa, 0x31)
}
pub fn convert_upper(number: u32) -> [u8; 4] {
    convert_base(number, 0x1a, 0x41)
}
pub fn convert_lower(number: u32) -> [u8; 4] {
    convert_base(number, 0x1a, 0x61)
}
pub fn convert_hex(number: u32) -> [u8; 4] {
    let mut out = convert_base(number, 0x10, 0x11);
    out.iter_mut().for_each(|o| {
        if *o > 0x39 {
            *o += 7;
        }
    });
    out
}
