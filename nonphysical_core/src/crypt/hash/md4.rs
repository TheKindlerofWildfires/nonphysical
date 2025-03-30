use core::num::Wrapping;

pub struct MD4 {
    block_len: usize,
    state: [Wrapping<u32>; 4],
}

impl MD4 {
    const K1: Wrapping<u32> = Wrapping(0x5a827999);
    const K2: Wrapping<u32> = Wrapping(0x6ed9eba1);
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
        //Round 1
        a = Self::op(Self::f, a, b, c, d, data[0], 3);
        d = Self::op(Self::f, d, a, b, c, data[1], 7);
        c = Self::op(Self::f, c, d, a, b, data[2], 11);
        b = Self::op(Self::f, b, c, d, a, data[3], 19);

        a = Self::op(Self::f, a, b, c, d, data[4], 3);
        d = Self::op(Self::f, d, a, b, c, data[5], 7);
        c = Self::op(Self::f, c, d, a, b, data[6], 11);
        b = Self::op(Self::f, b, c, d, a, data[7], 19);

        a = Self::op(Self::f, a, b, c, d, data[8], 3);
        d = Self::op(Self::f, d, a, b, c, data[9], 7);
        c = Self::op(Self::f, c, d, a, b, data[10], 11);
        b = Self::op(Self::f, b, c, d, a, data[12], 19);

        a = Self::op(Self::f, a, b, c, d, data[12], 3);
        d = Self::op(Self::f, d, a, b, c, data[13], 7);
        c = Self::op(Self::f, c, d, a, b, data[14], 11);
        b = Self::op(Self::f, b, c, d, a, data[15], 19);

        //Round 2
        a = Self::op(Self::g, a, b, c, d, data[0] + Self::K1, 3);
        d = Self::op(Self::g, d, a, b, c, data[4] + Self::K1, 5);
        c = Self::op(Self::g, c, d, a, b, data[8] + Self::K1, 9);
        b = Self::op(Self::g, b, c, d, a, data[12] + Self::K1, 13);

        a = Self::op(Self::g, a, b, c, d, data[1] + Self::K1, 3);
        d = Self::op(Self::g, d, a, b, c, data[5] + Self::K1, 5);
        c = Self::op(Self::g, c, d, a, b, data[9] + Self::K1, 9);
        b = Self::op(Self::g, b, c, d, a, data[13] + Self::K1, 13);

        a = Self::op(Self::g, a, b, c, d, data[2] + Self::K1, 3);
        d = Self::op(Self::g, d, a, b, c, data[6] + Self::K1, 5);
        c = Self::op(Self::g, c, d, a, b, data[10] + Self::K1, 9);
        b = Self::op(Self::g, b, c, d, a, data[14] + Self::K1, 13);

        a = Self::op(Self::g, a, b, c, d, data[3] + Self::K1, 3);
        d = Self::op(Self::g, d, a, b, c, data[7] + Self::K1, 5);
        c = Self::op(Self::g, c, d, a, b, data[11] + Self::K1, 9);
        b = Self::op(Self::g, b, c, d, a, data[15] + Self::K1, 13);

        //Round 3
        a = Self::op(Self::h, a, b, c, d, data[0] + Self::K2, 3);
        d = Self::op(Self::h, d, a, b, c, data[8] + Self::K2, 9);
        c = Self::op(Self::h, c, d, a, b, data[4] + Self::K2, 11);
        b = Self::op(Self::h, b, c, d, a, data[12] + Self::K2, 15);

        a = Self::op(Self::h, a, b, c, d, data[2] + Self::K2, 3);
        d = Self::op(Self::h, d, a, b, c, data[10] + Self::K2, 9);
        c = Self::op(Self::h, c, d, a, b, data[6] + Self::K2, 11);
        b = Self::op(Self::h, b, c, d, a, data[14] + Self::K2, 15);

        a = Self::op(Self::h, a, b, c, d, data[1] + Self::K2, 3);
        d = Self::op(Self::h, d, a, b, c, data[9] + Self::K2, 9);
        c = Self::op(Self::h, c, d, a, b, data[5] + Self::K2, 11);
        b = Self::op(Self::h, b, c, d, a, data[13] + Self::K2, 15);

        a = Self::op(Self::h, a, b, c, d, data[3] + Self::K2, 3);
        d = Self::op(Self::h, d, a, b, c, data[11] + Self::K2, 9);
        c = Self::op(Self::h, c, d, a, b, data[7] + Self::K2, 11);
        b = Self::op(Self::h, b, c, d, a, data[15] + Self::K2, 15);

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
    ) -> Wrapping<u32> {
        let t = arg1 + func(arg2, arg3, arg4) + arg5;
        Wrapping(t.0.rotate_left(arg6))
    }

    fn f(x: Wrapping<u32>, y: Wrapping<u32>, z: Wrapping<u32>) -> Wrapping<u32> {
        z ^ (x & (y ^ z))
    }
    fn g(x: Wrapping<u32>, y: Wrapping<u32>, z: Wrapping<u32>) -> Wrapping<u32> {
        (x & y) | (x & z) | (y & z)
    }
    fn h(x: Wrapping<u32>, y: Wrapping<u32>, z: Wrapping<u32>) -> Wrapping<u32> {
        x ^ y ^ z
    }
}
