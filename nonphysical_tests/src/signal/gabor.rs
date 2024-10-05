
#[cfg(test)]
mod gabor_tests {
    use std::time::SystemTime;

    use nonphysical_core::{random::pcg::PermutedCongruentialGenerator, shared::complex::{Complex, ComplexScaler}, signal::gabor::{gabor_heap::GaborTransformHeap, GaborTransform}};
    use nonphysical_std::shared::primitive::F32;

    #[test]
    fn gabor_c_time() {
        let mut pcg = PermutedCongruentialGenerator::new(3, 0);
        let mut signal =
            (0..2048*4096).map(|_| ComplexScaler::new(F32(pcg.next_u32() as f32 / u32::MAX as f32), F32(0.0))).collect::<Vec<_>>();
        

        let window =  (0..2048).map(|_| ComplexScaler::new(F32(pcg.next_u32() as f32 / u32::MAX as f32), F32(0.0))).collect::<Vec<_>>();       
        let now = SystemTime::now();
        let gb = GaborTransformHeap::new((1024,window));
        let _ = gb.gabor(&mut signal);
        let _ = println!("{:?}",now.elapsed());

    }
}