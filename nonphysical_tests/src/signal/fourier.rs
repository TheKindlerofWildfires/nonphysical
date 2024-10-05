
#[cfg(test)]
mod fourier_heap {
    use std::time::SystemTime;
    use nonphysical_core::shared::primitive::Primitive;
    use nonphysical_core::signal::fourier::FourierTransform;
    use nonphysical_core::{shared::{complex::{Complex, ComplexScaler}, float::Float}, signal::fourier::fourier_heap::ComplexFourierTransformHeap};
    use nonphysical_std::shared::primitive::F32;

    #[test]
    fn forward_2_static() {
        let mut x = (0..2)
            .map(|i| ComplexScaler::<F32>::new(F32::usize(i), F32::usize(i+1)))
            .collect::<Vec<_>>();
        let fft = ComplexFourierTransformHeap::new(x.len());

        let truth_1 = [
            ComplexScaler::<F32>::new(F32(1.0), F32(3.0)),
            ComplexScaler::<F32>::new(F32(-1.0), F32(-1.)),
        ];
        let truth_2 = [
            ComplexScaler::<F32>::new(F32(0.0), F32(2.0)),
            ComplexScaler::<F32>::new(F32(2.0), F32(4.0)),
        ];
        let truth_3 = [
            ComplexScaler::<F32>::new(F32(2.0), F32(6.0)),
            ComplexScaler::<F32>::new(F32(-2.0), F32(-2.0)),
        ];

        fft.fft(&mut x);
        x.iter().zip(truth_1).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < F32::EPSILON);
        });

        fft.fft(&mut x);

        x.iter().zip(truth_2).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < F32::EPSILON);
        });

        fft.fft(&mut x);

        x.iter().zip(truth_3).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < F32::EPSILON);
        });
    }

    #[test]
    fn forward_4_static() {
        let mut x = (0..4)
            .map(|i| ComplexScaler::<F32>::new(F32(i as f32), F32((i+1) as f32)))
            .collect::<Vec<_>>();
        let fft = ComplexFourierTransformHeap::new(x.len());

        let truth_1 = [
            ComplexScaler::<F32>::new(F32(6.0), F32(10.0)),
            ComplexScaler::<F32>::new(F32(-4.0), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-2.0), F32(-2.0)),
            ComplexScaler::<F32>::new(F32(0.0), F32(-4.0)),
        ];
        let truth_2 = [
            ComplexScaler::<F32>::new(F32(0.0), F32(4.0)),
            ComplexScaler::<F32>::new(F32(12.0), F32(16.0)),
            ComplexScaler::<F32>::new(F32(8.0), F32(12.0)),
            ComplexScaler::<F32>::new(F32(4.0), F32(8.0)),
        ];
        let truth_3 = [
            ComplexScaler::<F32>::new(F32(24.0), F32(40.0)),
            ComplexScaler::<F32>::new(F32(0.0), F32(-16.0)),
            ComplexScaler::<F32>::new(F32(-8.0), F32(-8.0)),
            ComplexScaler::<F32>::new(F32(-16.0), F32(0.0)),
        ];

        fft.fft(&mut x);
        x.iter().zip(truth_1).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < F32::EPSILON);
        });

        fft.fft(&mut x);

        x.iter().zip(truth_2).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < F32::EPSILON);
        });

        fft.fft(&mut x);

        x.iter().zip(truth_3).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < F32::EPSILON);
        });
    }

    #[test]
    fn forward_8_static() {
        let mut x = (0..8)
            .map(|i| ComplexScaler::<F32>::new(F32(i as f32), F32((i+1) as f32)))
            .collect::<Vec<_>>();
        let fft = ComplexFourierTransformHeap::new(x.len());

        let truth_1 = [
            ComplexScaler::<F32>::new(F32(28.0), F32(36.0)),
            ComplexScaler::<F32>::new(F32(-13.65685425), F32(5.65685425)),
            ComplexScaler::<F32>::new(F32(-8.0), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-5.65685425), F32(-2.34314575)),
            ComplexScaler::<F32>::new(F32(-4.0), F32(-4.0)),
            ComplexScaler::<F32>::new(F32(-2.34314575), F32(-5.65685425)),
            ComplexScaler::<F32>::new(F32(0.0), F32(-8.0)),
            ComplexScaler::<F32>::new(F32(5.65685425), F32(-13.65685425)),
        ];
        let truth_2 = [
            ComplexScaler::<F32>::new(F32(0.0), F32(8.0)),
            ComplexScaler::<F32>::new(F32(56.0), F32(64.0)),
            ComplexScaler::<F32>::new(F32(48.0), F32(56.0)),
            ComplexScaler::<F32>::new(F32(40.0), F32(48.0)),
            ComplexScaler::<F32>::new(F32(32.0), F32(40.0)),
            ComplexScaler::<F32>::new(F32(24.0), F32(32.0)),
            ComplexScaler::<F32>::new(F32(16.0), F32(24.0)),
            ComplexScaler::<F32>::new(F32(8.0), F32(16.0)),
        ];

        fft.fft(&mut x);
        x.iter().zip(truth_1).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < F32::EPSILON);
        });

        fft.fft(&mut x);

        x.iter().zip(truth_2).for_each(|(xp, yp)| {
            assert_eq!(*xp, yp);
        });
    }

    #[test]
    fn forward_16_static() {
        let mut x = (0..16)
            .map(|i| ComplexScaler::<F32>::new(F32(i as f32), F32((i+1) as f32)))
            .collect::<Vec<_>>();
        let fft = ComplexFourierTransformHeap::new(x.len());

        let truth_1 = [
            ComplexScaler::<F32>::new(F32(120.0), F32(136.0)),
            ComplexScaler::<F32>::new(F32(-48.21871594), F32(32.21871594)),
            ComplexScaler::<F32>::new(F32(-27.3137085), F32(11.3137085)),
            ComplexScaler::<F32>::new(F32(-19.9728461), F32(3.9728461)),
            ComplexScaler::<F32>::new(F32(-16.0), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-13.3454291), F32(-2.6545709)),
            ComplexScaler::<F32>::new(F32(-11.3137085), F32(-4.6862915)),
            ComplexScaler::<F32>::new(F32(-9.59129894), F32(-6.40870106)),
            ComplexScaler::<F32>::new(F32(-8.0), F32(-8.0)),
            ComplexScaler::<F32>::new(F32(-6.40870106), F32(-9.59129894)),
            ComplexScaler::<F32>::new(F32(-4.6862915), F32(-11.3137085)),
            ComplexScaler::<F32>::new(F32(-2.6545709), F32(-13.3454291)),
            ComplexScaler::<F32>::new(F32(0.0), F32(-16.0)),
            ComplexScaler::<F32>::new(F32(3.9728461), F32(-19.9728461)),
            ComplexScaler::<F32>::new(F32(11.3137085), F32(-27.3137085)),
            ComplexScaler::<F32>::new(F32(32.21871594), F32(-48.21871594)),
        ];

        fft.fft(&mut x);
        x.iter().zip(truth_1).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < F32::EPSILON);
        });
    }

    #[test]
    fn inverse_2_static() {
        let mut x = [
            ComplexScaler::<F32>::new(F32(2.0), F32(6.0)),
            ComplexScaler::<F32>::new(F32(-2.0), F32(-2.0)),
        ];

        let fft = ComplexFourierTransformHeap::new(x.len());

        let truth_1 = [
            ComplexScaler::<F32>::new(F32(0.0), F32(2.0)),
            ComplexScaler::<F32>::new(F32(2.0), F32(4.0)),
        ];
        let truth_2 = [
            ComplexScaler::<F32>::new(F32(1.0), F32(3.0)),
            ComplexScaler::<F32>::new(F32(-1.0), F32(-1.0)),
        ];
        let truth_3 = (0..2)
            .map(|i| ComplexScaler::<F32>::new(F32(i as f32), F32((i+1) as f32)))
            .collect::<Vec<_>>();

        fft.ifft(&mut x);
        x.iter().zip(truth_1).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < F32::EPSILON);
        });

        fft.ifft(&mut x);

        x.iter().zip(truth_2).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < F32::EPSILON);
        });

        fft.ifft(&mut x);

        x.iter().zip(truth_3).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < F32::EPSILON);
        });
    }

    #[test]
    fn inverse_4_static() {
        let mut x = [
            ComplexScaler::<F32>::new(F32(24.0), F32(40.0)),
            ComplexScaler::<F32>::new(F32(0.0), F32(-16.0)),
            ComplexScaler::<F32>::new(F32(-8.0), F32(-8.0)),
            ComplexScaler::<F32>::new(F32(-16.0), F32(0.0)),
        ];

        let fft = ComplexFourierTransformHeap::new(x.len());

        let truth_1 = [
            ComplexScaler::<F32>::new(F32(0.0), F32(4.0)),
            ComplexScaler::<F32>::new(F32(12.0), F32(16.0)),
            ComplexScaler::<F32>::new(F32(8.0), F32(12.0)),
            ComplexScaler::<F32>::new(F32(4.0), F32(8.0)),
        ];
        let truth_2 = [
            ComplexScaler::<F32>::new(F32(6.0), F32(10.0)),
            ComplexScaler::<F32>::new(F32(-4.0), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-2.0), F32(-2.0)),
            ComplexScaler::<F32>::new(F32(0.0), F32(-4.0)),
        ];
        let truth_3 = (0..4)
            .map(|i| ComplexScaler::<F32>::new(F32(i as f32), F32((i+1) as f32)))
            .collect::<Vec<_>>();

        fft.ifft(&mut x);
        x.iter().zip(truth_1).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < F32::EPSILON);
        });

        fft.ifft(&mut x);

        x.iter().zip(truth_2).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < F32::EPSILON);
        });

        fft.ifft(&mut x);

        x.iter().zip(truth_3).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < F32::EPSILON);
        });
    }

    #[test]
    fn inverse_8_static() {
        let mut x = [
            ComplexScaler::<F32>::new(F32(0.0), F32(8.0)),
            ComplexScaler::<F32>::new(F32(56.0), F32(64.0)),
            ComplexScaler::<F32>::new(F32(48.0), F32(56.0)),
            ComplexScaler::<F32>::new(F32(40.0), F32(48.0)),
            ComplexScaler::<F32>::new(F32(32.0), F32(40.0)),
            ComplexScaler::<F32>::new(F32(24.0), F32(32.0)),
            ComplexScaler::<F32>::new(F32(16.0), F32(24.0)),
            ComplexScaler::<F32>::new(F32(8.0), F32(16.0)),
        ];
        let fft = ComplexFourierTransformHeap::new(x.len());

        let truth_1 = [
            ComplexScaler::<F32>::new(F32(28.0), F32(36.0)),
            ComplexScaler::<F32>::new(F32(-13.65685425), F32(5.65685425)),
            ComplexScaler::<F32>::new(F32(-8.0), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-5.65685425), F32(-2.34314575)),
            ComplexScaler::<F32>::new(F32(-4.0), F32(-4.0)),
            ComplexScaler::<F32>::new(F32(-2.34314575), F32(-5.65685425)),
            ComplexScaler::<F32>::new(F32(0.0), F32(-8.0)),
            ComplexScaler::<F32>::new(F32(5.65685425), F32(-13.65685425)),
        ];
        let truth_2 = (0..8)
            .map(|i| ComplexScaler::<F32>::new(F32(i as f32), F32((i+1) as f32)))
            .collect::<Vec<_>>();

        fft.ifft(&mut x);
        x.iter().zip(truth_1).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < F32::EPSILON);
        });

        fft.ifft(&mut x);

        x.iter().zip(truth_2).for_each(|(xp, yp)| {
            assert_eq!(*xp, yp);
        });
    }

    #[test]
    fn inverse_16_static() {
        let mut x = [
            ComplexScaler::<F32>::new(F32(120.0), F32(136.0)),
            ComplexScaler::<F32>::new(F32(-48.21871594), F32(32.21871594)),
            ComplexScaler::<F32>::new(F32(-27.3137085), F32(11.3137085)),
            ComplexScaler::<F32>::new(F32(-19.9728461), F32(3.9728461)),
            ComplexScaler::<F32>::new(F32(-16.0), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-13.3454291), F32(-2.6545709)),
            ComplexScaler::<F32>::new(F32(-11.3137085), F32(-4.6862915)),
            ComplexScaler::<F32>::new(F32(-9.59129894), F32(-6.40870106)),
            ComplexScaler::<F32>::new(F32(-8.0), F32(-8.0)),
            ComplexScaler::<F32>::new(F32(-6.40870106), F32(-9.59129894)),
            ComplexScaler::<F32>::new(F32(-4.6862915), F32(-11.3137085)),
            ComplexScaler::<F32>::new(F32(-2.6545709), F32(-13.3454291)),
            ComplexScaler::<F32>::new(F32(0.0), F32(-16.0)),
            ComplexScaler::<F32>::new(F32(3.9728461), F32(-19.9728461)),
            ComplexScaler::<F32>::new(F32(11.3137085), F32(-27.3137085)),
            ComplexScaler::<F32>::new(F32(32.21871594), F32(-48.21871594)),
        ];
        let fft = ComplexFourierTransformHeap::new(x.len());

        let truth_1 = (0..16)
            .map(|i| ComplexScaler::<F32>::new(F32(i as f32), F32((i+1) as f32)))
            .collect::<Vec<_>>();

        fft.ifft(&mut x);
        x.iter().zip(truth_1).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < F32::EPSILON);
        });
    }
    #[test]
    fn reverse() {
        let mut x = (0..1024)
            .map(|i| ComplexScaler::<F32>::new(F32(i as f32), F32((i+1) as f32)))
            .collect::<Vec<_>>();
        let fft: ComplexFourierTransformHeap<ComplexScaler<F32>> = ComplexFourierTransformHeap::new(x.len());
        fft.fft(&mut x);
        fft.ifft(&mut x);
        (0..1024).zip(x.iter()).for_each(|(i,xp)|{
            assert!((F32(i as f32)-xp.real).l2_norm() < F32::EPSILON);
            assert!((F32((i+1) as f32)-xp.imag).l2_norm() < F32::EPSILON);

        });
    }
    #[test]
    fn time_1024(){
        let forward = SystemTime::now();
        let mut x = (0..1024)
            .map(|i| ComplexScaler::<F32>::new(F32(i as f32), F32((i+1) as f32)))
            .collect::<Vec<_>>();
        let fft: ComplexFourierTransformHeap<ComplexScaler<F32>> = ComplexFourierTransformHeap::new(x.len());
        (0..1024).for_each(|_|{
            fft.fft(&mut x);
        });
        let forward_time = forward.elapsed();
        let backward = SystemTime::now();

        (0..1024).for_each(|_|{
            fft.ifft(&mut x);
        });
        let backward_time = backward.elapsed();

        let _ = println!("Forward {:?}", forward_time);
        let _ = println!("Backward {:?}", backward_time);

    }

    #[test]
    fn time_2048(){
        let forward = SystemTime::now();
        let mut x = (0..2048)
            .map(|i| ComplexScaler::<F32>::new(F32(i as f32), F32((i+1) as f32)))
            .collect::<Vec<_>>();
        let fft: ComplexFourierTransformHeap<ComplexScaler<F32>> = ComplexFourierTransformHeap::new(x.len());
        (0..1024).for_each(|_|{
            fft.fft(&mut x);
        });
        let forward_time = forward.elapsed();
        let backward = SystemTime::now();

        (0..1024).for_each(|_|{
            fft.ifft(&mut x);
        });
        let backward_time = backward.elapsed();

        let _ = println!("Forward {:?}", forward_time);
        let _ = println!("Backward {:?}", backward_time);

    }

    #[test]
    fn time_4096(){
        let forward = SystemTime::now();
        let mut x = (0..4096)
            .map(|i| ComplexScaler::<F32>::new(F32(i as f32), F32((i+1) as f32)))
            .collect::<Vec<_>>();
        let fft: ComplexFourierTransformHeap<ComplexScaler<F32>> = ComplexFourierTransformHeap::new(x.len());
        (0..1024).for_each(|_|{
            fft.fft(&mut x);
        });
        let forward_time = forward.elapsed();
        let backward = SystemTime::now();

        (0..1024).for_each(|_|{
            fft.ifft(&mut x);
        });
        let backward_time = backward.elapsed();

        let _ = println!("Forward {:?}", forward_time);
        let _ = println!("Backward {:?}", backward_time);

    }
}