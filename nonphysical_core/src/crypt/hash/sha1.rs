use core::num::Wrapping;

pub struct SHA1 {
    block_len: usize,
    state: [Wrapping<u32>; 5],
}
macro_rules! rounds4 {
    ($h0:ident, $h1:ident, $wk:expr, $i:expr) => {
        SHA1::sha1_digest_round_x4($h0, SHA1::sha1_first_half($h1, $wk), $i)
    };
}

macro_rules! schedule {
    ($v0:expr, $v1:expr, $v2:expr, $v3:expr) => {
        SHA1::sha1msg2(SHA1::xor(SHA1::sha1msg1($v0, $v1), $v2), $v3)
    };
}

macro_rules! schedule_rounds4 {
    (
        $h0:ident, $h1:ident,
        $w0:expr, $w1:expr, $w2:expr, $w3:expr, $w4:expr,
        $i:expr
    ) => {
        $w4 = schedule!($w0, $w1, $w2, $w3);
        $h1 = rounds4!($h0, $h1, $w4, $i);
    };
}

impl SHA1 {
    const K: [Wrapping<u32>; 4] = [
        Wrapping(0x5A827999),
        Wrapping(0x6ED9EBA1),
        Wrapping(0x8F1BBCDC),
        Wrapping(0xCA62C1D6),
    ];
    const BLOCK_LEN: usize = 64;
    pub fn new() -> Self {
        let block_len = 0;
        let state = [
            Wrapping(0x67452301),
            Wrapping(0xEFCDAB89),
            Wrapping(0x98BADCFE),
            Wrapping(0x10325476),
            Wrapping(0xC3D2E1F0),
        ];
        Self { block_len, state }
    }
    #[inline(always)]
    pub fn digest(&mut self, input: &[u8]) -> [u8; 20] {
        let chunks = input.chunks_exact(Self::BLOCK_LEN);
        let remainder = chunks.remainder();
        chunks.for_each(|chunk| {
            self.block_len += 1;
            let mut block = [Wrapping(0u32); 16];
            block
                .iter_mut()
                .zip(chunk.chunks_exact(4))
                .for_each(|(b, c)| {
                    b.0 = u32::from_be_bytes(c.try_into().unwrap());
                });
            self.sha1_digest_block_u32(&block);
        });

        self.finalize(remainder);

        let mut output = [0; 20];
        output
            .chunks_exact_mut(4)
            .zip(self.state)
            .for_each(|(o, s)| {
                o.iter_mut()
                    .zip(s.0.to_be_bytes().iter())
                    .for_each(|(o, a)| *o = *a);
            });
        output
    }
    #[inline(always)]
    fn finalize(&mut self, remainder: &[u8]) {
        let len = Wrapping((Self::BLOCK_LEN * self.block_len + remainder.len()) as u64);
        let bit_len = Wrapping(8) * len;
        let mut final_value = [0; Self::BLOCK_LEN];
        remainder
            .iter()
            .chain([0x80].iter())
            .zip(final_value.iter_mut())
            .for_each(|(b, i)| *i = *b);
        bit_len
            .0
            .to_be_bytes()
            .iter()
            .zip(final_value.iter_mut().skip(Self::BLOCK_LEN - 8))
            .for_each(|(b, i)| *i = *b);
        let mut block = [Wrapping(0u32); 16];
        block
            .iter_mut()
            .zip(final_value.chunks_exact(4))
            .for_each(|(b, c)| {
                b.0 = u32::from_be_bytes(c.try_into().unwrap());
            });
        self.sha1_digest_block_u32(&block);
    }

    #[inline(always)]
    fn sha1_digest_block_u32(&mut self, block: &[Wrapping<u32>; 16]) {
        let mut w0 = [block[0], block[1], block[2], block[3]];
        let mut w1 = [block[4], block[5], block[6], block[7]];
        let mut w2 = [block[8], block[9], block[10], block[11]];
        let mut w3 = [block[12], block[13], block[14], block[15]];
        let mut w4;

        let mut h0 = [self.state[0], self.state[1], self.state[2], self.state[3]];
        let mut h1 = Self::sha1_first_add(self.state[4], w0);

        // Rounds 0..20
        h1 = Self::sha1_digest_round_x4(h0, h1, 0);
        h0 = rounds4!(h1, h0, w1, 0);
        h1 = rounds4!(h0, h1, w2, 0);
        h0 = rounds4!(h1, h0, w3, 0);
        schedule_rounds4!(h0, h1, w0, w1, w2, w3, w4, 0);

        // Rounds 20..40
        schedule_rounds4!(h1, h0, w1, w2, w3, w4, w0, 1);
        schedule_rounds4!(h0, h1, w2, w3, w4, w0, w1, 1);
        schedule_rounds4!(h1, h0, w3, w4, w0, w1, w2, 1);
        schedule_rounds4!(h0, h1, w4, w0, w1, w2, w3, 1);
        schedule_rounds4!(h1, h0, w0, w1, w2, w3, w4, 1);

        // Rounds 40..60
        schedule_rounds4!(h0, h1, w1, w2, w3, w4, w0, 2);
        schedule_rounds4!(h1, h0, w2, w3, w4, w0, w1, 2);
        schedule_rounds4!(h0, h1, w3, w4, w0, w1, w2, 2);
        schedule_rounds4!(h1, h0, w4, w0, w1, w2, w3, 2);
        schedule_rounds4!(h0, h1, w0, w1, w2, w3, w4, 2);

        // Rounds 60..80
        schedule_rounds4!(h1, h0, w1, w2, w3, w4, w0, 3);
        schedule_rounds4!(h0, h1, w2, w3, w4, w0, w1, 3);
        schedule_rounds4!(h1, h0, w3, w4, w0, w1, w2, 3);
        schedule_rounds4!(h0, h1, w4, w0, w1, w2, w3, 3);
        schedule_rounds4!(h1, h0, w0, w1, w2, w3, w4, 3);

        let e = Wrapping(h1[0].0.rotate_left(30));
        let [a, b, c, d] = h0;

        self.state[0] += a;
        self.state[1] += b;
        self.state[2] += c;
        self.state[3] += d;
        self.state[4] += e;
    }
    #[inline(always)]
    fn add(a: [Wrapping<u32>; 4], b: [Wrapping<u32>; 4]) -> [Wrapping<u32>; 4] {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
    }
    #[inline(always)]
    fn xor(a: [Wrapping<u32>; 4], b: [Wrapping<u32>; 4]) -> [Wrapping<u32>; 4] {
        [a[0] ^ b[0], a[1] ^ b[1], a[2] ^ b[2], a[3] ^ b[3]]
    }
    #[inline(always)]
    fn sha1_first_add(e: Wrapping<u32>, w0: [Wrapping<u32>; 4]) -> [Wrapping<u32>; 4] {
        let [a, b, c, d] = w0;
        [e + a, b, c, d]
    }
    #[inline(always)]
    fn sha1msg1(a: [Wrapping<u32>; 4], b: [Wrapping<u32>; 4]) -> [Wrapping<u32>; 4] {
        let [_, _, w2, w3] = a;
        let [w4, w5, _, _] = b;
        [a[0] ^ w2, a[1] ^ w3, a[2] ^ w4, a[3] ^ w5]
    }
    #[inline(always)]
    fn sha1msg2(a: [Wrapping<u32>; 4], b: [Wrapping<u32>; 4]) -> [Wrapping<u32>; 4] {
        let [x0, x1, x2, x3] = a;
        let [_, w13, w14, w15] = b;

        let w16 = Wrapping((x0 ^ w13).0.rotate_left(1));
        let w17 = Wrapping((x1 ^ w14).0.rotate_left(1));
        let w18 = Wrapping((x2 ^ w15).0.rotate_left(1));
        let w19 = Wrapping((x3 ^ w16).0.rotate_left(1));

        [w16, w17, w18, w19]
    }
    #[inline(always)]
    fn sha1_first_half(abcd: [Wrapping<u32>; 4], msg: [Wrapping<u32>; 4]) -> [Wrapping<u32>; 4] {
        Self::sha1_first_add(Wrapping(abcd[0].0.rotate_left(30)), msg)
    }
    #[inline(always)]
    fn sha1_digest_round_x4(
        abcd: [Wrapping<u32>; 4],
        work: [Wrapping<u32>; 4],
        i: i8,
    ) -> [Wrapping<u32>; 4] {
        match i {
            0 => Self::sha1rnds4c(abcd, Self::add(work, [Self::K[0]; 4])),
            1 => Self::sha1rnds4p(abcd, Self::add(work, [Self::K[1]; 4])),
            2 => Self::sha1rnds4m(abcd, Self::add(work, [Self::K[2]; 4])),
            3 => Self::sha1rnds4p(abcd, Self::add(work, [Self::K[3]; 4])),
            _ => unreachable!(),
        }
    }
    #[inline(always)]
    fn sha1rnds4c(abcd: [Wrapping<u32>; 4], msg: [Wrapping<u32>; 4]) -> [Wrapping<u32>; 4] {
        let [mut a, mut b, mut c, mut d] = abcd;
        let [t, u, v, w] = msg;
        let mut e = Wrapping(0u32);

        macro_rules! bool3ary_202 {
            ($a:expr, $b:expr, $c:expr) => {
                $c ^ ($a & ($b ^ $c))
            };
        } // Choose, MD5F, SHA1C

        e = e + Wrapping(a.0.rotate_left(5)) + bool3ary_202!(b, c, d) + t;
        b = Wrapping(b.0.rotate_left(30));
        d = d + Wrapping(e.0.rotate_left(5)) + bool3ary_202!(a, b, c) + u;
        a = Wrapping(a.0.rotate_left(30));
        c = c + Wrapping(d.0.rotate_left(5)) + bool3ary_202!(e, a, b) + v;
        e = Wrapping(e.0.rotate_left(30));
        b = b + Wrapping(c.0.rotate_left(5)) + bool3ary_202!(d, e, a) + w;
        d = Wrapping(d.0.rotate_left(30));

        [b, c, d, e]
    }
    #[inline(always)]
    fn sha1rnds4p(abcd: [Wrapping<u32>; 4], msg: [Wrapping<u32>; 4]) -> [Wrapping<u32>; 4] {
        let [mut a, mut b, mut c, mut d] = abcd;
        let [t, u, v, w] = msg;
        let mut e = Wrapping(0u32);

        macro_rules! bool3ary_150 {
            ($a:expr, $b:expr, $c:expr) => {
                $a ^ $b ^ $c
            };
        }
        e = e + Wrapping(a.0.rotate_left(5)) + bool3ary_150!(b, c, d) + t;
        b = Wrapping(b.0.rotate_left(30));
        d = d + Wrapping(e.0.rotate_left(5)) + bool3ary_150!(a, b, c) + u;
        a = Wrapping(a.0.rotate_left(30));
        c = c + Wrapping(d.0.rotate_left(5)) + bool3ary_150!(e, a, b) + v;
        e = Wrapping(e.0.rotate_left(30));
        b = b + Wrapping(c.0.rotate_left(5)) + bool3ary_150!(d, e, a) + w;
        d = Wrapping(d.0.rotate_left(30));

        [b, c, d, e]
    }
    #[inline(always)]
    fn sha1rnds4m(abcd: [Wrapping<u32>; 4], msg: [Wrapping<u32>; 4]) -> [Wrapping<u32>; 4] {
        let [mut a, mut b, mut c, mut d] = abcd;
        let [t, u, v, w] = msg;
        let mut e = Wrapping(0u32);

        macro_rules! bool3ary_232 {
            ($a:expr, $b:expr, $c:expr) => {
                ($a & $b) ^ ($a & $c) ^ ($b & $c)
            };
        }

        e = e + Wrapping(a.0.rotate_left(5)) + bool3ary_232!(b, c, d) + t;
        b = Wrapping(b.0.rotate_left(30));
        d = d + Wrapping(e.0.rotate_left(5)) + bool3ary_232!(a, b, c) + u;
        a = Wrapping(a.0.rotate_left(30));
        c = c + Wrapping(d.0.rotate_left(5)) + bool3ary_232!(e, a, b) + v;
        e = Wrapping(e.0.rotate_left(30));
        b = b + Wrapping(c.0.rotate_left(5)) + bool3ary_232!(d, e, a) + w;
        d = Wrapping(d.0.rotate_left(30));

        [b, c, d, e]
    }
}
