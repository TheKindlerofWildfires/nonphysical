use core::hash::Hasher;

#[derive(Debug, Clone)]
pub struct FallHasher {
    buffer: u64,
    pad: u64,
    extra_keys: [u64; 2],
}

impl FallHasher {
    pub fn new(u: u64, v: u64, w: u64, x: u64) -> Self {
        let u = u ^ 0x243f_6a88_85a3_08d3;
        let v = v ^ 0x1319_8a2e_0370_7344;
        let w = w ^ 0xa409_3822_299f_31d0;
        let x = x ^ 0x082e_fa98_ec4e_6c89;
        Self {
            buffer: u,
            pad: v,
            extra_keys: [w, x],
        }
    }
    pub fn update(&mut self, y: u64){
        let result = ((y^self.buffer) as u128).wrapping_mul(6364136223846793005);
        self.buffer =  ((result & 0xffff_ffff_ffff_ffff) as u64) ^ ((result >> 64) as u64)
    }
    pub fn large_update(&mut self, y: u64, z: u64){
        let result = ((y^self.extra_keys[0]) as u128).wrapping_mul((z^self.extra_keys[1]) as u128);
        let combined = ((result & 0xffff_ffff_ffff_ffff) as u64) ^ ((result >> 64) as u64);
        self.buffer = (self.buffer.wrapping_add(self.pad)^combined).rotate_left(23);
    }
}

impl Hasher for FallHasher{
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.update(i as u64);
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.update(i as u64);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.update(i as u64);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.update(i as u64);
    }

    #[inline]
    fn write_u128(&mut self, i: u128) {
        self.large_update(i as u64, (i>>64) as u64);
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }
    fn write(&mut self, input: &[u8]) {
        let mut data = input;
        let length = data.len() as u64;
        //Needs to be an add rather than an xor because otherwise it could be canceled with carefully formed input.
        self.buffer = self.buffer.wrapping_add(length).wrapping_mul(6364136223846793005);
        //A 'binary search' on sizes reduces the number of comparisons.
        if data.len() > 8 {
            if data.len() > 16 {
                let (_,tb)= data.split_at(data.len()-16);
                let y: [u8;8] = tb[..8].try_into().unwrap();
                let z: [u8;8] = tb[8..].try_into().unwrap();
                self.large_update(u64::from_le_bytes(y),u64::from_le_bytes(z));
                while data.len() > 16 {
                    let (tb,rest)= data.split_at(16);
                    let y: [u8;8] = tb[..8].try_into().unwrap();
                    let z: [u8;8] = tb[8..].try_into().unwrap();
                    self.large_update(u64::from_le_bytes(y),u64::from_le_bytes(z));
                    data = rest;
                }
            } else {
                let (tb,_)= data.split_at(8);
                let y: [u8;8] = tb.try_into().unwrap();
                let (tb,_)= data.split_at(data.len()-8);
                let z: [u8;8] = tb.try_into().unwrap();
                self.large_update(u64::from_le_bytes(y),u64::from_le_bytes(z));
            }
        } else {
            let yz = if data.len() >= 2 {
                if data.len() >= 4 {
                    let (tby,_)= data.split_at(4);
                    let y: [u8;4] = tby.try_into().unwrap();
                    let (_,tbz)= data.split_at(data.len()-4);
                    let z: [u8;4] = tbz.try_into().unwrap();
                    [u32::from_le_bytes(y)as u64, u32::from_le_bytes(z) as u64]
                } else {
                    let (tby,_)= data.split_at(2);
                    let y: [u8;2] = tby.try_into().unwrap();
                    [u16::from_le_bytes(y)as u64, data[data.len() - 1] as u64]
                }
            } else {
                if data.len() > 0 {
                    [data[0] as u64, data[0] as u64]
                } else {
                    [0, 0]
                }
            };
            self.large_update(yz[0],yz[1]);
        }
    }
    fn finish(&self) -> u64 {
        let result = ((self.buffer) as u128).wrapping_mul((self.pad) as u128);
        ((result & 0xffff_ffff_ffff_ffff) as u64) ^ ((result >> 64) as u64).rotate_left((self.buffer & 63) as u32)
    }
}
