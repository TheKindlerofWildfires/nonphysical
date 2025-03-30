use core::num::Wrapping;

pub struct SHA256 {
    block_len: usize,
    state: [Wrapping<u32>; 8],
}
macro_rules! rounds4 {
    ($abef:ident, $cdgh:ident, $rest:expr, $i:expr) => {{
        let t1 = SHA256::add_round_const($rest, $i);
        $cdgh = SHA256::sha256_digest_round_x2($cdgh, $abef, t1);
        let t2 = SHA256::sha256swap(t1);
        $abef = SHA256::sha256_digest_round_x2($abef, $cdgh, t2);
    }};
}

macro_rules! schedule_rounds4 {
    (
        $abef:ident, $cdgh:ident,
        $w0:expr, $w1:expr, $w2:expr, $w3:expr, $w4:expr,
        $i: expr
    ) => {{
        $w4 = SHA256::schedule($w0, $w1, $w2, $w3);
        rounds4!($abef, $cdgh, $w4, $i);
    }};
}

impl SHA256 {
    const K32: [Wrapping<u32>; 64] = [
        Wrapping(0x428a2f98),
        Wrapping(0x71374491),
        Wrapping(0xb5c0fbcf),
        Wrapping(0xe9b5dba5),
        Wrapping(0x3956c25b),
        Wrapping(0x59f111f1),
        Wrapping(0x923f82a4),
        Wrapping(0xab1c5ed5),
        Wrapping(0xd807aa98),
        Wrapping(0x12835b01),
        Wrapping(0x243185be),
        Wrapping(0x550c7dc3),
        Wrapping(0x72be5d74),
        Wrapping(0x80deb1fe),
        Wrapping(0x9bdc06a7),
        Wrapping(0xc19bf174),
        Wrapping(0xe49b69c1),
        Wrapping(0xefbe4786),
        Wrapping(0x0fc19dc6),
        Wrapping(0x240ca1cc),
        Wrapping(0x2de92c6f),
        Wrapping(0x4a7484aa),
        Wrapping(0x5cb0a9dc),
        Wrapping(0x76f988da),
        Wrapping(0x983e5152),
        Wrapping(0xa831c66d),
        Wrapping(0xb00327c8),
        Wrapping(0xbf597fc7),
        Wrapping(0xc6e00bf3),
        Wrapping(0xd5a79147),
        Wrapping(0x06ca6351),
        Wrapping(0x14292967),
        Wrapping(0x27b70a85),
        Wrapping(0x2e1b2138),
        Wrapping(0x4d2c6dfc),
        Wrapping(0x53380d13),
        Wrapping(0x650a7354),
        Wrapping(0x766a0abb),
        Wrapping(0x81c2c92e),
        Wrapping(0x92722c85),
        Wrapping(0xa2bfe8a1),
        Wrapping(0xa81a664b),
        Wrapping(0xc24b8b70),
        Wrapping(0xc76c51a3),
        Wrapping(0xd192e819),
        Wrapping(0xd6990624),
        Wrapping(0xf40e3585),
        Wrapping(0x106aa070),
        Wrapping(0x19a4c116),
        Wrapping(0x1e376c08),
        Wrapping(0x2748774c),
        Wrapping(0x34b0bcb5),
        Wrapping(0x391c0cb3),
        Wrapping(0x4ed8aa4a),
        Wrapping(0x5b9cca4f),
        Wrapping(0x682e6ff3),
        Wrapping(0x748f82ee),
        Wrapping(0x78a5636f),
        Wrapping(0x84c87814),
        Wrapping(0x8cc70208),
        Wrapping(0x90befffa),
        Wrapping(0xa4506ceb),
        Wrapping(0xbef9a3f7),
        Wrapping(0xc67178f2),
    ];
    const BLOCK_LEN: usize = 64;
    pub fn new() -> Self {
        let block_len = 0;
        let state = [
            Wrapping(0x6a09e667),
            Wrapping(0xbb67ae85),
            Wrapping(0x3c6ef372),
            Wrapping(0xa54ff53a),
            Wrapping(0x510e527f),
            Wrapping(0x9b05688c),
            Wrapping(0x1f83d9ab),
            Wrapping(0x5be0cd19),
        ];
        Self { block_len, state }
    }
    #[inline(always)]
    pub fn digest(&mut self, input: &[u8]) -> [u8; 32] {
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
            self.sha256_digest_block_u32(&block);
        });

        self.finalize(remainder);

        let mut output = [0; 32];
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
        self.sha256_digest_block_u32(&block);
    }

    #[inline(always)]
    fn shr(v: [Wrapping<u32>; 4], o: u32) -> [Wrapping<u32>; 4] {
        [
            Wrapping(v[0].0 >> o),
            Wrapping(v[1].0 >> o),
            Wrapping(v[2].0 >> o),
            Wrapping(v[3].0 >> o),
        ]
    }

    #[inline(always)]
    fn shl(v: [Wrapping<u32>; 4], o: u32) -> [Wrapping<u32>; 4] {
        [
            Wrapping(v[0].0 << o),
            Wrapping(v[1].0 << o),
            Wrapping(v[2].0 << o),
            Wrapping(v[3].0 << o),
        ]
    }

    #[inline(always)]
    fn or(a: [Wrapping<u32>; 4], b: [Wrapping<u32>; 4]) -> [Wrapping<u32>; 4] {
        [a[0] | b[0], a[1] | b[1], a[2] | b[2], a[3] | b[3]]
    }

    #[inline(always)]
    fn xor(a: [Wrapping<u32>; 4], b: [Wrapping<u32>; 4]) -> [Wrapping<u32>; 4] {
        [a[0] ^ b[0], a[1] ^ b[1], a[2] ^ b[2], a[3] ^ b[3]]
    }

    #[inline(always)]
    fn add(a: [Wrapping<u32>; 4], b: [Wrapping<u32>; 4]) -> [Wrapping<u32>; 4] {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
    }

    #[inline(always)]
    fn add_round_const(mut a: [Wrapping<u32>; 4], i: usize) -> [Wrapping<u32>; 4] {
        a[3] = a[3] + Self::K32[i * 4];
        a[2] = a[2] + Self::K32[i * 4 + 1];
        a[1] = a[1] + Self::K32[i * 4 + 2];
        a[0] = a[0] + Self::K32[i * 4 + 3];
        a
    }
    #[inline(always)]
    fn sha256load(v2: [Wrapping<u32>; 4], v3: [Wrapping<u32>; 4]) -> [Wrapping<u32>; 4] {
        [v3[3], v2[0], v2[1], v2[2]]
    }
    #[inline(always)]
    fn sha256swap(v0: [Wrapping<u32>; 4]) -> [Wrapping<u32>; 4] {
        [v0[2], v0[3], v0[0], v0[1]]
    }
    #[inline(always)]
    fn sha256msg1(v0: [Wrapping<u32>; 4], v1: [Wrapping<u32>; 4]) -> [Wrapping<u32>; 4] {
        Self::add(v0, Self::sigma0x4(Self::sha256load(v0, v1)))
    }
    #[inline(always)]
    fn sigma0x4(x: [Wrapping<u32>; 4]) -> [Wrapping<u32>; 4] {
        let t1 = Self::or(Self::shr(x, 7), Self::shl(x, 25));
        let t2 = Self::or(Self::shr(x, 18), Self::shl(x, 14));
        let t3 = Self::shr(x, 3);
        Self::xor(Self::xor(t1, t2), t3)
    }
    #[inline(always)]
    fn sha256msg2(v4: [Wrapping<u32>; 4], v3: [Wrapping<u32>; 4]) -> [Wrapping<u32>; 4] {
        let [x3, x2, x1, x0] = v4;
        let [w15, w14, _, _] = v3;

        let w16 =
            x0 + Wrapping(w14.0.rotate_right(17)) ^ Wrapping(w14.0.rotate_right(19)) ^ (w14 >> 10);
        let w17 =
            x1 + Wrapping(w15.0.rotate_right(17)) ^ Wrapping(w15.0.rotate_right(19)) ^ (w15 >> 10);
        let w18 =
            x2 + Wrapping(w16.0.rotate_right(17)) ^ Wrapping(w16.0.rotate_right(19)) ^ (w16 >> 10);
        let w19 =
            x3 + Wrapping(w17.0.rotate_right(17)) ^ Wrapping(w17.0.rotate_right(19)) ^ (w17 >> 10);

        [w19, w18, w17, w16]
    }
    #[inline(always)]
    fn sha256_digest_round_x2(
        cdgh: [Wrapping<u32>; 4],
        abef: [Wrapping<u32>; 4],
        wk: [Wrapping<u32>; 4],
    ) -> [Wrapping<u32>; 4] {
        macro_rules! big_sigma0 {
            ($a:expr) => {
                Wrapping($a.0.rotate_right(2) ^ $a.0.rotate_right(13) ^ $a.0.rotate_right(22))
            };
        }
        macro_rules! big_sigma1 {
            ($a:expr) => {
                Wrapping($a.0.rotate_right(6) ^ $a.0.rotate_right(11) ^ $a.0.rotate_right(25))
            };
        }
        macro_rules! bool3ary_202 {
            ($a:expr, $b:expr, $c:expr) => {
                $c ^ ($a & ($b ^ $c))
            };
        } // Choose, MD5F, SHA1C
        macro_rules! bool3ary_232 {
            ($a:expr, $b:expr, $c:expr) => {
                ($a & $b) ^ ($a & $c) ^ ($b & $c)
            };
        } // Majority, SHA1M

        let [_, _, wk1, wk0] = wk;
        let [a0, b0, e0, f0] = abef;
        let [c0, d0, g0, h0] = cdgh;

        // a round
        let x0 = big_sigma1!(e0) + bool3ary_202!(e0, f0, g0) + wk0 + h0;
        let y0 = big_sigma0!(a0) + bool3ary_232!(a0, b0, c0);
        let (a1, b1, c1, d1, e1, f1, g1, h1) = (x0 + y0, a0, b0, c0, x0 + d0, e0, f0, g0);

        // a round
        let x1 = big_sigma1!(e1) + bool3ary_202!(e1, f1, g1) + wk1 + h1;
        let y1 = big_sigma0!(a1) + bool3ary_232!(a1, b1, c1);
        let (a2, b2, _, _, e2, f2, _, _) = (x1 + y1, a1, b1, c1, x1 + d1, e1, f1, g1);

        [a2, b2, e2, f2]
    }
    #[inline(always)]
    fn schedule(
        v0: [Wrapping<u32>; 4],
        v1: [Wrapping<u32>; 4],
        v2: [Wrapping<u32>; 4],
        v3: [Wrapping<u32>; 4],
    ) -> [Wrapping<u32>; 4] {
        let t1 = Self::sha256msg1(v0, v1);
        let t2 = Self::sha256load(v2, v3);
        let t3 = Self::add(t1, t2);
        Self::sha256msg2(t3, v3)
    }
    #[inline(always)]
    fn sha256_digest_block_u32(&mut self, block: &[Wrapping<u32>; 16]) {
        let mut abef = [self.state[0], self.state[1], self.state[4], self.state[5]];
        let mut cdgh = [self.state[2], self.state[3], self.state[6], self.state[7]];

        // Rounds 0..64
        let mut w0 = [block[3], block[2], block[1], block[0]];
        let mut w1 = [block[7], block[6], block[5], block[4]];
        let mut w2 = [block[11], block[10], block[9], block[8]];
        let mut w3 = [block[15], block[14], block[13], block[12]];
        let mut w4;

        rounds4!(abef, cdgh, w0, 0);
        rounds4!(abef, cdgh, w1, 1);
        rounds4!(abef, cdgh, w2, 2);
        rounds4!(abef, cdgh, w3, 3);
        schedule_rounds4!(abef, cdgh, w0, w1, w2, w3, w4, 4);
        schedule_rounds4!(abef, cdgh, w1, w2, w3, w4, w0, 5);
        schedule_rounds4!(abef, cdgh, w2, w3, w4, w0, w1, 6);
        schedule_rounds4!(abef, cdgh, w3, w4, w0, w1, w2, 7);
        schedule_rounds4!(abef, cdgh, w4, w0, w1, w2, w3, 8);
        schedule_rounds4!(abef, cdgh, w0, w1, w2, w3, w4, 9);
        schedule_rounds4!(abef, cdgh, w1, w2, w3, w4, w0, 10);
        schedule_rounds4!(abef, cdgh, w2, w3, w4, w0, w1, 11);
        schedule_rounds4!(abef, cdgh, w3, w4, w0, w1, w2, 12);
        schedule_rounds4!(abef, cdgh, w4, w0, w1, w2, w3, 13);
        schedule_rounds4!(abef, cdgh, w0, w1, w2, w3, w4, 14);
        schedule_rounds4!(abef, cdgh, w1, w2, w3, w4, w0, 15);

        let [a, b, e, f] = abef;
        let [c, d, g, h] = cdgh;

        self.state[0] = self.state[0] + a;
        self.state[1] = self.state[1] + b;
        self.state[2] = self.state[2] + c;
        self.state[3] = self.state[3] + d;
        self.state[4] = self.state[4] + e;
        self.state[5] = self.state[5] + f;
        self.state[6] = self.state[6] + g;
        self.state[7] = self.state[7] + h;
    }
}

pub struct SHA512 {
    block_len: usize,
    state: [Wrapping<u64>; 8],
}

macro_rules! rounds4_512 {
    ($ae:ident, $bf:ident, $cg:ident, $dh:ident, $wk0:expr, $wk1:expr) => {{
        let [u, t] = $wk0;
        let [w, v] = $wk1;

        $dh = SHA512::sha512_digest_round($ae, $bf, $cg, $dh, t);
        $cg = SHA512::sha512_digest_round($dh, $ae, $bf, $cg, u);
        $bf = SHA512::sha512_digest_round($cg, $dh, $ae, $bf, v);
        $ae = SHA512::sha512_digest_round($bf, $cg, $dh, $ae, w);
    }};
}
impl SHA512 {
    const K64: [Wrapping<u64>; 80] = [
        Wrapping(0x428a2f98d728ae22),
        Wrapping(0x7137449123ef65cd),
        Wrapping(0xb5c0fbcfec4d3b2f),
        Wrapping(0xe9b5dba58189dbbc),
        Wrapping(0x3956c25bf348b538),
        Wrapping(0x59f111f1b605d019),
        Wrapping(0x923f82a4af194f9b),
        Wrapping(0xab1c5ed5da6d8118),
        Wrapping(0xd807aa98a3030242),
        Wrapping(0x12835b0145706fbe),
        Wrapping(0x243185be4ee4b28c),
        Wrapping(0x550c7dc3d5ffb4e2),
        Wrapping(0x72be5d74f27b896f),
        Wrapping(0x80deb1fe3b1696b1),
        Wrapping(0x9bdc06a725c71235),
        Wrapping(0xc19bf174cf692694),
        Wrapping(0xe49b69c19ef14ad2),
        Wrapping(0xefbe4786384f25e3),
        Wrapping(0x0fc19dc68b8cd5b5),
        Wrapping(0x240ca1cc77ac9c65),
        Wrapping(0x2de92c6f592b0275),
        Wrapping(0x4a7484aa6ea6e483),
        Wrapping(0x5cb0a9dcbd41fbd4),
        Wrapping(0x76f988da831153b5),
        Wrapping(0x983e5152ee66dfab),
        Wrapping(0xa831c66d2db43210),
        Wrapping(0xb00327c898fb213f),
        Wrapping(0xbf597fc7beef0ee4),
        Wrapping(0xc6e00bf33da88fc2),
        Wrapping(0xd5a79147930aa725),
        Wrapping(0x06ca6351e003826f),
        Wrapping(0x142929670a0e6e70),
        Wrapping(0x27b70a8546d22ffc),
        Wrapping(0x2e1b21385c26c926),
        Wrapping(0x4d2c6dfc5ac42aed),
        Wrapping(0x53380d139d95b3df),
        Wrapping(0x650a73548baf63de),
        Wrapping(0x766a0abb3c77b2a8),
        Wrapping(0x81c2c92e47edaee6),
        Wrapping(0x92722c851482353b),
        Wrapping(0xa2bfe8a14cf10364),
        Wrapping(0xa81a664bbc423001),
        Wrapping(0xc24b8b70d0f89791),
        Wrapping(0xc76c51a30654be30),
        Wrapping(0xd192e819d6ef5218),
        Wrapping(0xd69906245565a910),
        Wrapping(0xf40e35855771202a),
        Wrapping(0x106aa07032bbd1b8),
        Wrapping(0x19a4c116b8d2d0c8),
        Wrapping(0x1e376c085141ab53),
        Wrapping(0x2748774cdf8eeb99),
        Wrapping(0x34b0bcb5e19b48a8),
        Wrapping(0x391c0cb3c5c95a63),
        Wrapping(0x4ed8aa4ae3418acb),
        Wrapping(0x5b9cca4f7763e373),
        Wrapping(0x682e6ff3d6b2b8a3),
        Wrapping(0x748f82ee5defb2fc),
        Wrapping(0x78a5636f43172f60),
        Wrapping(0x84c87814a1f0ab72),
        Wrapping(0x8cc702081a6439ec),
        Wrapping(0x90befffa23631e28),
        Wrapping(0xa4506cebde82bde9),
        Wrapping(0xbef9a3f7b2c67915),
        Wrapping(0xc67178f2e372532b),
        Wrapping(0xca273eceea26619c),
        Wrapping(0xd186b8c721c0c207),
        Wrapping(0xeada7dd6cde0eb1e),
        Wrapping(0xf57d4f7fee6ed178),
        Wrapping(0x06f067aa72176fba),
        Wrapping(0x0a637dc5a2c898a6),
        Wrapping(0x113f9804bef90dae),
        Wrapping(0x1b710b35131c471b),
        Wrapping(0x28db77f523047d84),
        Wrapping(0x32caab7b40c72493),
        Wrapping(0x3c9ebe0a15c9bebc),
        Wrapping(0x431d67c49c100d4c),
        Wrapping(0x4cc5d4becb3e42b6),
        Wrapping(0x597f299cfc657e2a),
        Wrapping(0x5fcb6fab3ad6faec),
        Wrapping(0x6c44198c4a475817),
    ];
    const BLOCK_LEN: usize = 128;
    #[inline(always)]
    pub fn new() -> Self {
        let block_len = 0;
        let state = [
            Wrapping(0x6a09e667f3bcc908),
            Wrapping(0xbb67ae8584caa73b),
            Wrapping(0x3c6ef372fe94f82b),
            Wrapping(0xa54ff53a5f1d36f1),
            Wrapping(0x510e527fade682d1),
            Wrapping(0x9b05688c2b3e6c1f),
            Wrapping(0x1f83d9abfb41bd6b),
            Wrapping(0x5be0cd19137e2179),
        ];
        Self { block_len, state }
    }
    #[inline(always)]
    pub fn digest(&mut self, input: &[u8]) -> [u8; 64] {
        let chunks = input.chunks_exact(Self::BLOCK_LEN);
        let remainder = chunks.remainder();
        chunks.for_each(|chunk| {
            self.block_len += 1;
            let mut block = [Wrapping(0u64); 16];
            block
                .iter_mut()
                .zip(chunk.chunks_exact(8))
                .for_each(|(b, c)| {
                    b.0 = u64::from_be_bytes(c.try_into().unwrap());
                });
            self.sha512_digest_block_u64(&block);
        });

        self.finalize(remainder);

        let mut output = [0; 64];
        output
            .chunks_exact_mut(8)
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
        let mut block = [Wrapping(0u64); 16];
        block
            .iter_mut()
            .zip(final_value.chunks_exact(8))
            .for_each(|(b, c)| {
                b.0 = u64::from_be_bytes(c.try_into().unwrap());
            });
        self.sha512_digest_block_u64(&block);
    }
    fn sha512load(v0: [Wrapping<u64>; 2], v1: [Wrapping<u64>; 2]) -> [Wrapping<u64>; 2] {
        [v1[1], v0[0]]
    }

    #[inline(always)]
    pub fn sha512_schedule_x2(
        v0: [Wrapping<u64>; 2],
        v1: [Wrapping<u64>; 2],
        v4to5: [Wrapping<u64>; 2],
        v7: [Wrapping<u64>; 2],
    ) -> [Wrapping<u64>; 2] {
        // sigma 0
        #[inline(always)]
        fn sigma0(x: Wrapping<u64>) -> Wrapping<u64> {
            Wrapping((x.0.rotate_right(1)) ^ (x.0.rotate_right(8)) ^ (x.0 >> 7))
        }

        // sigma 1
        #[inline(always)]
        fn sigma1(x: Wrapping<u64>) -> Wrapping<u64> {
            Wrapping((x.0.rotate_right(19)) ^ (x.0.rotate_left(3)) ^ (x.0 >> 6))
        }

        let [w1, w0] = v0;
        let [_, w2] = v1;
        let [w10, w9] = v4to5;
        let [w15, w14] = v7;

        let w16 = sigma1(w14) + w9 + sigma0(w1) + w0;
        let w17 = sigma1(w15) + w10 + sigma0(w2) + w1;

        [w17, w16]
    }
    #[inline(always)]
    pub fn sha512_digest_round(
        ae: [Wrapping<u64>; 2],
        bf: [Wrapping<u64>; 2],
        cg: [Wrapping<u64>; 2],
        dh: [Wrapping<u64>; 2],
        wk0: Wrapping<u64>,
    ) -> [Wrapping<u64>; 2] {
        macro_rules! big_sigma0 {
            ($a:expr) => {
                Wrapping($a.0.rotate_right(28) ^ $a.0.rotate_right(34) ^ $a.0.rotate_right(39))
            };
        }
        macro_rules! big_sigma1 {
            ($a:expr) => {
                Wrapping($a.0.rotate_right(14) ^ $a.0.rotate_right(18) ^ $a.0.rotate_right(41))
            };
        }
        macro_rules! bool3ary_202 {
            ($a:expr, $b:expr, $c:expr) => {
                $c ^ ($a & ($b ^ $c))
            };
        } // Choose, MD5F, SHA1C
        macro_rules! bool3ary_232 {
            ($a:expr, $b:expr, $c:expr) => {
                ($a & $b) ^ ($a & $c) ^ ($b & $c)
            };
        } // Majority, SHA1M

        let [a0, e0] = ae;
        let [b0, f0] = bf;
        let [c0, g0] = cg;
        let [d0, h0] = dh;

        // a round
        let x0 = big_sigma1!(e0) + bool3ary_202!(e0, f0, g0) + wk0 + h0;
        let y0 = big_sigma0!(a0) + bool3ary_232!(a0, b0, c0);
        let (a1, _, _, _, e1, _, _, _) = (x0 + y0, a0, b0, c0, x0 + d0, e0, f0, g0);

        [a1, e1]
    }

    #[inline(always)]
    fn add_rk(mut w: [Wrapping<u64>; 2], i: usize) -> [Wrapping<u64>; 2] {
        w[1] = w[1] + Self::K64[i * 2];
        w[0] = w[0] + Self::K64[i * 2 + 1];
        w
    }
    #[inline(always)]
    /// Process a block with the SHA-512 algorithm.
    pub fn sha512_digest_block_u64(&mut self, block: &[Wrapping<u64>; 16]) {
        let mut ae = [self.state[0], self.state[4]];
        let mut bf = [self.state[1], self.state[5]];
        let mut cg = [self.state[2], self.state[6]];
        let mut dh = [self.state[3], self.state[7]];

        // Rounds 0..20
        let (mut w1, mut w0) = ([block[3], block[2]], [block[1], block[0]]);
        rounds4_512!(ae, bf, cg, dh, Self::add_rk(w0, 0), Self::add_rk(w1, 1));
        let (mut w3, mut w2) = ([block[7], block[6]], [block[5], block[4]]);
        rounds4_512!(ae, bf, cg, dh, Self::add_rk(w2, 2), Self::add_rk(w3, 3));
        let (mut w5, mut w4) = ([block[11], block[10]], [block[9], block[8]]);
        rounds4_512!(ae, bf, cg, dh, Self::add_rk(w4, 4), Self::add_rk(w5, 5));
        let (mut w7, mut w6) = ([block[15], block[14]], [block[13], block[12]]);
        rounds4_512!(ae, bf, cg, dh, Self::add_rk(w6, 6), Self::add_rk(w7, 7));
        let mut w8 = SHA512::sha512_schedule_x2(w0, w1, SHA512::sha512load(w4, w5), w7);
        let mut w9 = SHA512::sha512_schedule_x2(w1, w2, SHA512::sha512load(w5, w6), w8);
        rounds4_512!(ae, bf, cg, dh, Self::add_rk(w8, 8), Self::add_rk(w9, 9));

        // Rounds 20..40
        w0 = SHA512::sha512_schedule_x2(w2, w3, SHA512::sha512load(w6, w7), w9);
        w1 = SHA512::sha512_schedule_x2(w3, w4, SHA512::sha512load(w7, w8), w0);
        rounds4_512!(ae, bf, cg, dh, Self::add_rk(w0, 10), Self::add_rk(w1, 11));
        w2 = SHA512::sha512_schedule_x2(w4, w5, SHA512::sha512load(w8, w9), w1);
        w3 = SHA512::sha512_schedule_x2(w5, w6, SHA512::sha512load(w9, w0), w2);
        rounds4_512!(ae, bf, cg, dh, Self::add_rk(w2, 12), Self::add_rk(w3, 13));
        w4 = SHA512::sha512_schedule_x2(w6, w7, SHA512::sha512load(w0, w1), w3);
        w5 = SHA512::sha512_schedule_x2(w7, w8, SHA512::sha512load(w1, w2), w4);
        rounds4_512!(ae, bf, cg, dh, Self::add_rk(w4, 14), Self::add_rk(w5, 15));
        w6 = SHA512::sha512_schedule_x2(w8, w9, SHA512::sha512load(w2, w3), w5);
        w7 = SHA512::sha512_schedule_x2(w9, w0, SHA512::sha512load(w3, w4), w6);
        rounds4_512!(ae, bf, cg, dh, Self::add_rk(w6, 16), Self::add_rk(w7, 17));
        w8 = SHA512::sha512_schedule_x2(w0, w1, SHA512::sha512load(w4, w5), w7);
        w9 = SHA512::sha512_schedule_x2(w1, w2, SHA512::sha512load(w5, w6), w8);
        rounds4_512!(ae, bf, cg, dh, Self::add_rk(w8, 18), Self::add_rk(w9, 19));

        // Rounds 40..60
        w0 = SHA512::sha512_schedule_x2(w2, w3, SHA512::sha512load(w6, w7), w9);
        w1 = SHA512::sha512_schedule_x2(w3, w4, SHA512::sha512load(w7, w8), w0);
        rounds4_512!(ae, bf, cg, dh, Self::add_rk(w0, 20), Self::add_rk(w1, 21));
        w2 = SHA512::sha512_schedule_x2(w4, w5, SHA512::sha512load(w8, w9), w1);
        w3 = SHA512::sha512_schedule_x2(w5, w6, SHA512::sha512load(w9, w0), w2);
        rounds4_512!(ae, bf, cg, dh, Self::add_rk(w2, 22), Self::add_rk(w3, 23));
        w4 = SHA512::sha512_schedule_x2(w6, w7, SHA512::sha512load(w0, w1), w3);
        w5 = SHA512::sha512_schedule_x2(w7, w8, SHA512::sha512load(w1, w2), w4);
        rounds4_512!(ae, bf, cg, dh, Self::add_rk(w4, 24), Self::add_rk(w5, 25));
        w6 = SHA512::sha512_schedule_x2(w8, w9, SHA512::sha512load(w2, w3), w5);
        w7 = SHA512::sha512_schedule_x2(w9, w0, SHA512::sha512load(w3, w4), w6);
        rounds4_512!(ae, bf, cg, dh, Self::add_rk(w6, 26), Self::add_rk(w7, 27));
        w8 = SHA512::sha512_schedule_x2(w0, w1, SHA512::sha512load(w4, w5), w7);
        w9 = SHA512::sha512_schedule_x2(w1, w2, SHA512::sha512load(w5, w6), w8);
        rounds4_512!(ae, bf, cg, dh, Self::add_rk(w8, 28), Self::add_rk(w9, 29));

        // Rounds 60..80
        w0 = SHA512::sha512_schedule_x2(w2, w3, SHA512::sha512load(w6, w7), w9);
        w1 = SHA512::sha512_schedule_x2(w3, w4, SHA512::sha512load(w7, w8), w0);
        rounds4_512!(ae, bf, cg, dh, Self::add_rk(w0, 30), Self::add_rk(w1, 31));
        w2 = SHA512::sha512_schedule_x2(w4, w5, SHA512::sha512load(w8, w9), w1);
        w3 = SHA512::sha512_schedule_x2(w5, w6, SHA512::sha512load(w9, w0), w2);
        rounds4_512!(ae, bf, cg, dh, Self::add_rk(w2, 32), Self::add_rk(w3, 33));
        w4 = SHA512::sha512_schedule_x2(w6, w7, SHA512::sha512load(w0, w1), w3);
        w5 = SHA512::sha512_schedule_x2(w7, w8, SHA512::sha512load(w1, w2), w4);
        rounds4_512!(ae, bf, cg, dh, Self::add_rk(w4, 34), Self::add_rk(w5, 35));
        w6 = SHA512::sha512_schedule_x2(w8, w9, SHA512::sha512load(w2, w3), w5);
        w7 = SHA512::sha512_schedule_x2(w9, w0, SHA512::sha512load(w3, w4), w6);
        rounds4_512!(ae, bf, cg, dh, Self::add_rk(w6, 36), Self::add_rk(w7, 37));
        w8 = SHA512::sha512_schedule_x2(w0, w1, SHA512::sha512load(w4, w5), w7);
        w9 = SHA512::sha512_schedule_x2(w1, w2, SHA512::sha512load(w5, w6), w8);
        rounds4_512!(ae, bf, cg, dh, Self::add_rk(w8, 38), Self::add_rk(w9, 39));

        let [a, e] = ae;
        let [b, f] = bf;
        let [c, g] = cg;
        let [d, h] = dh;

        self.state[0] = self.state[0] + a;
        self.state[1] = self.state[1] + b;
        self.state[2] = self.state[2] + c;
        self.state[3] = self.state[3] + d;
        self.state[4] = self.state[4] + e;
        self.state[5] = self.state[5] + f;
        self.state[6] = self.state[6] + g;
        self.state[7] = self.state[7] + h;
    }
}
