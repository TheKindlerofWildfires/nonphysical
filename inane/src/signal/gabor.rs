
#[cfg(test)]
mod gabor_tests {
    use std::time::SystemTime;

    use baseless::{random::pcg::PermutedCongruentialGenerator, shared::complex::{Complex, ComplexScaler}, signal::gabor::GaborTransform};

    #[test]
    fn gabor_c_time() {
        let mut pcg = PermutedCongruentialGenerator::new(3, 0);
        let mut signal =
            (0..2048*4096).map(|_| ComplexScaler::new(pcg.next_u32() as f32 / u32::MAX as f32, 0.0)).collect::<Vec<_>>();
        

        let window =  (0..2048).map(|_| ComplexScaler::new(pcg.next_u32() as f32 / u32::MAX as f32, 0.0)).collect::<Vec<_>>();       
        let now = SystemTime::now();
        let gb = GaborTransform::new(1,window);
        let _ = gb.gabor(&mut signal);
        let _ = println!("{:?}",now.elapsed());

    }
}