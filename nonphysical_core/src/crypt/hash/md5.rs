use core::num::Wrapping;

pub struct MD5 {
    block_len: usize,
    state: [Wrapping<u32>; 4],
}

impl MD5 {
    const RC: [u32; 64] = [
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
        0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
        0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
        0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
        0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
        0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
        0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
        0xeb86d391,
    ];
    const BLOCK_LEN: usize = 64;
    pub fn new() -> Self {
        let block_len = 0;
        let state = [
            Wrapping(0x67452301),
            Wrapping(0xefcdab89),
            Wrapping(0x98badcfe),
            Wrapping(0x10325476),
        ];
        Self { block_len, state }
    }
    #[inline(always)]
    pub fn digest(&mut self, input: &[u8]) -> [u8; 16] {
        let chunks = input.chunks_exact(Self::BLOCK_LEN);
        let remainder = chunks.remainder();
        chunks.for_each(|chunk| {
            self.block_len += 1;
            self.update_block(chunk.try_into().unwrap());
        });

        self.finalize(remainder);

        let mut output = [0; 16];
        output
            .chunks_exact_mut(4)
            .zip(self.state)
            .for_each(|(o, s)| {
                o.iter_mut()
                    .zip(s.0.to_le_bytes().iter())
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
            .to_le_bytes()
            .iter()
            .zip(final_value.iter_mut().skip(Self::BLOCK_LEN - 8))
            .for_each(|(b, i)| *i = *b);

        self.update_block(&final_value);
    }
    #[inline(always)]
    pub fn update_block(&mut self, input: &[u8; Self::BLOCK_LEN]) {
        let mut data = [Default::default(); Self::BLOCK_LEN / 4];
        data.iter_mut()
            .zip(input.chunks_exact(4))
            .for_each(|(data, chunk)| {
                *data = Wrapping(u32::from_le_bytes(chunk.try_into().unwrap()))
            });
        let mut a = self.state[0];
        let mut b = self.state[1];
        let mut c = self.state[2];
        let mut d = self.state[3];
        // Round 1
        a = Self::op(Self::f, a, b, c, d, data[0], Self::RC[0], 7);
        d = Self::op(Self::f, d, a, b, c, data[1], Self::RC[1], 12);
        c = Self::op(Self::f, c, d, a, b, data[2], Self::RC[2], 17);
        b = Self::op(Self::f, b, c, d, a, data[3], Self::RC[3], 22);

        a = Self::op(Self::f, a, b, c, d, data[4], Self::RC[4], 7);
        d = Self::op(Self::f, d, a, b, c, data[5], Self::RC[5], 12);
        c = Self::op(Self::f, c, d, a, b, data[6], Self::RC[6], 17);
        b = Self::op(Self::f, b, c, d, a, data[7], Self::RC[7], 22);

        a = Self::op(Self::f, a, b, c, d, data[8], Self::RC[8], 7);
        d = Self::op(Self::f, d, a, b, c, data[9], Self::RC[9], 12);
        c = Self::op(Self::f, c, d, a, b, data[10], Self::RC[10], 17);
        b = Self::op(Self::f, b, c, d, a, data[11], Self::RC[11], 22);

        a = Self::op(Self::f, a, b, c, d, data[12], Self::RC[12], 7);
        d = Self::op(Self::f, d, a, b, c, data[13], Self::RC[13], 12);
        c = Self::op(Self::f, c, d, a, b, data[14], Self::RC[14], 17);
        b = Self::op(Self::f, b, c, d, a, data[15], Self::RC[15], 22);

        // Round 2
        a = Self::op(Self::g, a, b, c, d, data[1], Self::RC[16], 5);
        d = Self::op(Self::g, d, a, b, c, data[6], Self::RC[17], 9);
        c = Self::op(Self::g, c, d, a, b, data[11], Self::RC[18], 14);
        b = Self::op(Self::g, b, c, d, a, data[0], Self::RC[19], 20);

        a = Self::op(Self::g, a, b, c, d, data[5], Self::RC[20], 5);
        d = Self::op(Self::g, d, a, b, c, data[10], Self::RC[21], 9);
        c = Self::op(Self::g, c, d, a, b, data[15], Self::RC[22], 14);
        b = Self::op(Self::g, b, c, d, a, data[4], Self::RC[23], 20);

        a = Self::op(Self::g, a, b, c, d, data[9], Self::RC[24], 5);
        d = Self::op(Self::g, d, a, b, c, data[14], Self::RC[25], 9);
        c = Self::op(Self::g, c, d, a, b, data[3], Self::RC[26], 14);
        b = Self::op(Self::g, b, c, d, a, data[8], Self::RC[27], 20);

        a = Self::op(Self::g, a, b, c, d, data[13], Self::RC[28], 5);
        d = Self::op(Self::g, d, a, b, c, data[2], Self::RC[29], 9);
        c = Self::op(Self::g, c, d, a, b, data[7], Self::RC[30], 14);
        b = Self::op(Self::g, b, c, d, a, data[12], Self::RC[31], 20);

        // Round 3
        a = Self::op(Self::h, a, b, c, d, data[5], Self::RC[32], 4);
        d = Self::op(Self::h, d, a, b, c, data[8], Self::RC[33], 11);
        c = Self::op(Self::h, c, d, a, b, data[11], Self::RC[34], 16);
        b = Self::op(Self::h, b, c, d, a, data[14], Self::RC[35], 23);

        a = Self::op(Self::h, a, b, c, d, data[1], Self::RC[36], 4);
        d = Self::op(Self::h, d, a, b, c, data[4], Self::RC[37], 11);
        c = Self::op(Self::h, c, d, a, b, data[7], Self::RC[38], 16);
        b = Self::op(Self::h, b, c, d, a, data[10], Self::RC[39], 23);

        a = Self::op(Self::h, a, b, c, d, data[13], Self::RC[40], 4);
        d = Self::op(Self::h, d, a, b, c, data[0], Self::RC[41], 11);
        c = Self::op(Self::h, c, d, a, b, data[3], Self::RC[42], 16);
        b = Self::op(Self::h, b, c, d, a, data[6], Self::RC[43], 23);

        a = Self::op(Self::h, a, b, c, d, data[9], Self::RC[44], 4);
        d = Self::op(Self::h, d, a, b, c, data[12], Self::RC[45], 11);
        c = Self::op(Self::h, c, d, a, b, data[15], Self::RC[46], 16);
        b = Self::op(Self::h, b, c, d, a, data[2], Self::RC[47], 23);

        // Round 4
        a = Self::op(Self::i, a, b, c, d, data[0], Self::RC[48], 6);
        d = Self::op(Self::i, d, a, b, c, data[7], Self::RC[49], 10);
        c = Self::op(Self::i, c, d, a, b, data[14], Self::RC[50], 15);
        b = Self::op(Self::i, b, c, d, a, data[5], Self::RC[51], 21);

        a = Self::op(Self::i, a, b, c, d, data[12], Self::RC[52], 6);
        d = Self::op(Self::i, d, a, b, c, data[3], Self::RC[53], 10);
        c = Self::op(Self::i, c, d, a, b, data[10], Self::RC[54], 15);
        b = Self::op(Self::i, b, c, d, a, data[1], Self::RC[55], 21);

        a = Self::op(Self::i, a, b, c, d, data[8], Self::RC[56], 6);
        d = Self::op(Self::i, d, a, b, c, data[15], Self::RC[57], 10);
        c = Self::op(Self::i, c, d, a, b, data[6], Self::RC[58], 15);
        b = Self::op(Self::i, b, c, d, a, data[13], Self::RC[59], 21);

        a = Self::op(Self::i, a, b, c, d, data[4], Self::RC[60], 6);
        d = Self::op(Self::i, d, a, b, c, data[11], Self::RC[61], 10);
        c = Self::op(Self::i, c, d, a, b, data[2], Self::RC[62], 15);
        b = Self::op(Self::i, b, c, d, a, data[9], Self::RC[63], 21);
        self.state[0] += a;
        self.state[1] += b;
        self.state[2] += c;
        self.state[3] += d;
    }

    fn op<F: Fn(Wrapping<u32>, Wrapping<u32>, Wrapping<u32>) -> Wrapping<u32>>(
        func: F,
        arg1: Wrapping<u32>,
        arg2: Wrapping<u32>,
        arg3: Wrapping<u32>,
        arg4: Wrapping<u32>,
        arg5: Wrapping<u32>,
        arg6: u32,
        arg7: u32,
    ) -> Wrapping<u32> {
        let t = arg1 + func(arg2, arg3, arg4) + arg5 + Wrapping(arg6);
        Wrapping(t.0.rotate_left(arg7)) + arg2
    }

    fn f(x: Wrapping<u32>, y: Wrapping<u32>, z: Wrapping<u32>) -> Wrapping<u32> {
        (x & y) | (!x & z)
    }
    fn g(x: Wrapping<u32>, y: Wrapping<u32>, z: Wrapping<u32>) -> Wrapping<u32> {
        (x & z) | (y & !z)
    }
    fn h(x: Wrapping<u32>, y: Wrapping<u32>, z: Wrapping<u32>) -> Wrapping<u32> {
        x ^ y ^ z
    }
    fn i(x: Wrapping<u32>, y: Wrapping<u32>, z: Wrapping<u32>) -> Wrapping<u32> {
        y ^ (x | !z)
    }
}
