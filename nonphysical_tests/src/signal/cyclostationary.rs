#[cfg(test)]
mod cyclostationary_heap {
    use std::fs::File;
    use std::io::{Read, Write};
    use std::time::SystemTime;
    use nonphysical_core::shared::matrix::Matrix;
    use nonphysical_core::signal::cyclostationary::cyclostationary_heap::CycloStationaryTransformHeap;
    use nonphysical_core::signal::cyclostationary::CycloStationaryTransform;
    use nonphysical_core::shared::{complex::{Complex, ComplexScaler}, float::Float};
    use nonphysical_std::shared::primitive::F32;


    #[test]
    fn forward_file_static() {
        let mut file = File::open("of.np").unwrap();
        let mut buffer = Vec::new();
        let _ = file.read_to_end(&mut buffer);

        let mut data = buffer.chunks_exact(8).map(|chunk|{
            let r:[u8;4] = chunk[..4].try_into().unwrap();
            let i:[u8;4] = chunk[4..].try_into().unwrap();
            let real = F32(f32::from_le_bytes(r));
            let imag = F32(f32::from_le_bytes(i));
            ComplexScaler::new(real, imag)


        }).collect::<Vec<_>>();
        let _ = dbg!(data.len());

        let csth = CycloStationaryTransformHeap::new( vec![ComplexScaler::<F32>::IDENTITY;256]);
        let now = SystemTime::now();
        let mat = csth.fam(&mut data);
        let _ =dbg!(now.elapsed());
        dbg!(mat.data[0]);
        let mut buffer = Vec::new();
        let mut file = File::create("cyclo.np").unwrap();
        mat.data().for_each(|c|{
            buffer.extend_from_slice(&c.real.to_le_bytes());
            buffer.extend_from_slice(&c.imag.to_le_bytes());

        });
        let _ = file.write_all(&buffer);
        
    }

    #[test]
    fn forward_file_static_large() {
        let mut file = File::open("of.np").unwrap();
        let mut buffer = Vec::new();
        let _ = file.read_to_end(&mut buffer);

        let data = buffer.chunks_exact(8).map(|chunk|{
            let r:[u8;4] = chunk[..4].try_into().unwrap();
            let i:[u8;4] = chunk[4..].try_into().unwrap();
            let real = F32(f32::from_le_bytes(r));
            let imag = F32(f32::from_le_bytes(i));
            ComplexScaler::new(real, imag)


        }).collect::<Vec<_>>();
        let _ = dbg!(data.len());
        let mut aug = Vec::new();
        for _ in 0..10{
            aug.extend_from_slice(&data);
        }
        let csth = CycloStationaryTransformHeap::new( vec![ComplexScaler::<F32>::IDENTITY;8]);
        let now = SystemTime::now();
        let mat = csth.fam(&mut aug);
        dbg!(mat.rows,mat.cols);
        let _ =dbg!(now.elapsed());
        dbg!(mat.data[0]);
        
        let mut buffer = Vec::new();
        let mut file = File::create("cyclo.np").unwrap();
        mat.data().for_each(|c|{
            buffer.extend_from_slice(&c.real.to_le_bytes());
            buffer.extend_from_slice(&c.imag.to_le_bytes());

        });
        let _ = file.write_all(&buffer);
    }
}