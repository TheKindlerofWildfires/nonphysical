#[cfg(test)]
mod wavelet_tests {
    use std::{fs::File, io::Write, time::SystemTime};

    use nonphysical_core::{shared::{complex::{Complex, ComplexScaler}, matrix::Matrix, primitive::Primitive}, signal::{cyclostationary::{cyclostationary_heap::CycloStationaryTransformHeap, CycloStationaryTransform}, wavelet::{wavelet_heap::DaubechiesFirstWaveletHeap, DiscreteWavelet}}};
    use nonphysical_cuda::cuda::runtime::Runtime;
    use nonphysical_std::shared::primitive::F32;
    use nonphysical_core::shared::float::Float;
    use nonphysical_ptx::signal::{cyclostationary::cuda_cyclostationary::CycloStationaryTransformCuda, wavelet::cuda_wavelet::DaubechiesFirstWaveletCuda};
    #[test]
    fn fam_128(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let ncst = 128;
        let mut data = (0..ncst*16).map(|i| ComplexScaler::new(F32::usize(i%ncst), F32::ZERO)).collect::<Vec<_>>();
        let mut data_reference = (0..ncst*16).map(|i| ComplexScaler::new(F32::usize(i%ncst), F32::ZERO)).collect::<Vec<_>>();
        let vec = vec![ComplexScaler::<F32>::ONE;ncst];
        let ref_cst = CycloStationaryTransformHeap::new(vec);
        let now = SystemTime::now();
        let ref_out = ref_cst.fam(&mut data_reference);    
        dbg!(now.elapsed());
        let cst = CycloStationaryTransformCuda::new(ncst);
        let now = SystemTime::now();

        let out = cst.fam(&mut data);
        dbg!(now.elapsed());

        //dbg!(out.rows,out.cols,&out.data);
        let mut buffer = Vec::new();
        let mut file = File::create("cyclo.np").unwrap();
        out.data().for_each(|c|{
            buffer.extend_from_slice(&c.real.to_le_bytes());
            buffer.extend_from_slice(&c.imag.to_le_bytes());

        });
        let _ = file.write_all(&buffer);
    }

    #[test]
    fn fam_256(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let ncst = 256;
        let mut data = (0..ncst*16).map(|i| ComplexScaler::new(F32::usize(i%ncst), F32::ZERO)).collect::<Vec<_>>();
        let mut data_reference = (0..ncst*16).map(|i| ComplexScaler::new(F32::usize(i%ncst), F32::ZERO)).collect::<Vec<_>>();
        let vec = vec![ComplexScaler::<F32>::ONE;ncst];
        let ref_cst = CycloStationaryTransformHeap::new(vec);
        let now = SystemTime::now();
        let ref_out = ref_cst.fam(&mut data_reference);    
        dbg!(now.elapsed());
        dbg!(ref_out.rows, ref_out.cols);
        let cst = CycloStationaryTransformCuda::new(ncst);
        let now = SystemTime::now();

        let out = cst.fam(&mut data);
        dbg!(now.elapsed());

        //dbg!(out.rows,out.cols,&out.data);
        let mut buffer = Vec::new();
        let mut file = File::create("cyclo.np").unwrap();
        out.data().for_each(|c|{
            buffer.extend_from_slice(&c.real.to_le_bytes());
            buffer.extend_from_slice(&c.imag.to_le_bytes());

        });
        let _ = file.write_all(&buffer);
    }
    #[test]
    fn fam_512(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let ncst = 512;
        let mut data = (0..ncst*16).map(|i| ComplexScaler::new(F32::usize(i%ncst), F32::ZERO)).collect::<Vec<_>>();
        let mut data_reference = (0..ncst*16).map(|i| ComplexScaler::new(F32::usize(i%ncst), F32::ZERO)).collect::<Vec<_>>();
        let vec = vec![ComplexScaler::<F32>::ONE;ncst];
        let ref_cst = CycloStationaryTransformHeap::new(vec);
        let now = SystemTime::now();
        let ref_out = ref_cst.fam(&mut data_reference);    
        dbg!(now.elapsed());
        dbg!(ref_out.rows, ref_out.cols);
        let cst = CycloStationaryTransformCuda::new(ncst);
        let now = SystemTime::now();

        let out = cst.fam(&mut data);
        dbg!(now.elapsed());

        //dbg!(out.rows,out.cols,&out.data);
        let mut buffer = Vec::new();
        let mut file = File::create("cyclo.np").unwrap();
        out.data().for_each(|c|{
            buffer.extend_from_slice(&c.real.to_le_bytes());
            buffer.extend_from_slice(&c.imag.to_le_bytes());

        });
        let _ = file.write_all(&buffer);
    }

    #[test]
    fn fam_1024(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let ncst = 1024;
        let mut data = (0..ncst*16).map(|i| ComplexScaler::new(F32::usize(i%ncst), F32::ZERO)).collect::<Vec<_>>();
        let mut data_reference = (0..ncst*16).map(|i| ComplexScaler::new(F32::usize(i%ncst), F32::ZERO)).collect::<Vec<_>>();
        let vec = vec![ComplexScaler::<F32>::ONE;ncst];
        let ref_cst = CycloStationaryTransformHeap::new(vec);
        let now = SystemTime::now();
        let ref_out = ref_cst.fam(&mut data_reference);    
        dbg!(now.elapsed());
        dbg!(ref_out.rows, ref_out.cols);
        let cst = CycloStationaryTransformCuda::new(ncst);
        let now = SystemTime::now();

        let out = cst.fam(&mut data);
        dbg!(now.elapsed());

        //dbg!(out.rows,out.cols,&out.data);
        let mut buffer = Vec::new();
        let mut file = File::create("cyclo.np").unwrap();
        out.data().for_each(|c|{
            buffer.extend_from_slice(&c.real.to_le_bytes());
            buffer.extend_from_slice(&c.imag.to_le_bytes());

        });
        let _ = file.write_all(&buffer);
    }

    #[test]
    fn fam_2048(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let ncst = 2048;
        let mut data = (0..ncst*16).map(|i| ComplexScaler::new(F32::usize(i%ncst), F32::ZERO)).collect::<Vec<_>>();
        let mut data_reference = (0..ncst*16).map(|i| ComplexScaler::new(F32::usize(i%ncst), F32::ZERO)).collect::<Vec<_>>();
        let vec = vec![ComplexScaler::<F32>::ONE;ncst];
        let cst = CycloStationaryTransformCuda::new(ncst);
        let now = SystemTime::now();

        let out = cst.fam(&mut data);
        dbg!(now.elapsed());
    }
}