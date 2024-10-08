

#[cfg(test)]
mod iso_forest_tests {
    use std::time::SystemTime;

    use nonphysical_core::{cluster::iso_forest::IsoForest, random::pcg::PermutedCongruentialGenerator, shared::{float::Float, point::{Point, StaticPoint}, primitive::Primitive}};
    use nonphysical_std::shared::primitive::F32;


    #[test]
    fn create_forest() {
        let data = vec![
            StaticPoint::new([F32(9.308548692822459), F32(2.1673586347139224)]),
            StaticPoint::new([F32(-5.6424039931897765), F32( -1.9620561766472002)]),
            StaticPoint::new([F32(-9.821995596375428), F32( -3.1921112766174997)]),
            StaticPoint::new([F32(-4.992109362834896), F32( -2.0745015313494455)]),
            StaticPoint::new([F32(10.107315875917662), F32( 2.4489015959094216)]),
            StaticPoint::new([F32(-7.962477597931141), F32( -5.494741864480315)]),
            StaticPoint::new([F32(10.047917462523671), F32( 5.1631966716389766)]),
            StaticPoint::new([F32(-5.243921934674187), F32( -2.963359100733349)]),
            StaticPoint::new([F32(-9.940544426622527), F32( -3.2655473073528816)]),
            StaticPoint::new([F32(8.30445373000034), F32( 2.129694332932624)]),
            StaticPoint::new([F32(-9.196460281784482), F32( -3.987773678358418)]),
            StaticPoint::new([F32(-10.513583123594056), F32( -2.5364233580562887)]),
            StaticPoint::new([F32(9.072668506714033), F32( 3.405664632524281)]),
            StaticPoint::new([F32(-7.031861004012987), F32( -2.2616818331210844)]),
            StaticPoint::new([F32(9.627963795272553), F32( 4.502533177849574)]),
            StaticPoint::new([F32(-10.442760023564471), F32( -5.0830680881481065)]),
            StaticPoint::new([F32(8.292151321984209), F32( 3.8776876670218834)]),
            StaticPoint::new([F32(-6.51560033683665), F32( -3.8185628318207585)]),
            StaticPoint::new([F32(-10.887633624071544), F32(-4.416570704487158)]),
            StaticPoint::new([F32(-9.465804800021168), F32( -2.2222090878656884)]),
            StaticPoint::new([F32(100.0), F32(100.0)]),
            StaticPoint::new([F32(-100.0), F32(100.0)]),
            StaticPoint::new([F32(100.0), F32(-100.0)]),
            StaticPoint::new([F32(-100.0), F32(-100.0)]),
        ];

        let iso_forest: IsoForest<StaticPoint<F32, 2>> =
            IsoForest::new(&data, 20, 10, 0, 1, F32(0.0), F32(0.1));

        assert!(iso_forest.trees.len() == 21);
        assert!((iso_forest.average_path - F32(3.7488806)).l2_norm() < F32::EPSILON);
    }

    #[test]
    fn score_forest() {
        let data = vec![
            StaticPoint::new([F32(9.308548692822459), F32(2.1673586347139224)]),
            StaticPoint::new([F32(-5.6424039931897765), F32( -1.9620561766472002)]),
            StaticPoint::new([F32(-9.821995596375428), F32( -3.1921112766174997)]),
            StaticPoint::new([F32(-4.992109362834896), F32( -2.0745015313494455)]),
            StaticPoint::new([F32(10.107315875917662), F32( 2.4489015959094216)]),
            StaticPoint::new([F32(-7.962477597931141), F32( -5.494741864480315)]),
            StaticPoint::new([F32(10.047917462523671), F32( 5.1631966716389766)]),
            StaticPoint::new([F32(-5.243921934674187), F32( -2.963359100733349)]),
            StaticPoint::new([F32(-9.940544426622527), F32( -3.2655473073528816)]),
            StaticPoint::new([F32(8.30445373000034), F32( 2.129694332932624)]),
            StaticPoint::new([F32(-9.196460281784482), F32( -3.987773678358418)]),
            StaticPoint::new([F32(-10.513583123594056), F32( -2.5364233580562887)]),
            StaticPoint::new([F32(9.072668506714033), F32( 3.405664632524281)]),
            StaticPoint::new([F32(-7.031861004012987), F32( -2.2616818331210844)]),
            StaticPoint::new([F32(9.627963795272553), F32( 4.502533177849574)]),
            StaticPoint::new([F32(-10.442760023564471), F32( -5.0830680881481065)]),
            StaticPoint::new([F32(8.292151321984209), F32( 3.8776876670218834)]),
            StaticPoint::new([F32(-6.51560033683665), F32( -3.8185628318207585)]),
            StaticPoint::new([F32(-10.887633624071544), F32(-4.416570704487158)]),
            StaticPoint::new([F32(-9.465804800021168), F32( -2.2222090878656884)]),
            StaticPoint::new([F32(100.0), F32(100.0)]),
            StaticPoint::new([F32(-100.0), F32(100.0)]),
            StaticPoint::new([F32(100.0), F32(-100.0)]),
            StaticPoint::new([F32(-100.0), F32(-100.0)]),
        ];
        let iso_forest: IsoForest<StaticPoint<F32, 2>> =
            IsoForest::new(&data, 100, 10, 1, 1, F32(0.0), F32(0.1));
        let scores = iso_forest.score(&data);
        let known_scores = vec![
            0.5267391, 0.48126653, 0.48582554, 0.48349816, 0.53105694, 0.48471618, 0.53105694,
            0.48126653, 0.48702633, 0.52544886, 0.4834478, 0.49014044, 0.5267391, 0.48193628,
            0.5267391, 0.49014044, 0.52544886, 0.48021436, 0.49307784, 0.48450702, 0.6727004,
            0.6713952, 0.6727004, 0.6713952,
        ];
        scores
            .into_iter()
            .zip(known_scores.into_iter())
            .for_each(|(s, ks)| assert!((s - F32(ks)).l2_norm() < F32::EPSILON));
    }

    #[test]
    fn score_novel() {
        let data = vec![
            StaticPoint::new([F32(9.308548692822459), F32(2.1673586347139224)]),
            StaticPoint::new([F32(-5.6424039931897765), F32( -1.9620561766472002)]),
            StaticPoint::new([F32(-9.821995596375428), F32( -3.1921112766174997)]),
            StaticPoint::new([F32(-4.992109362834896), F32( -2.0745015313494455)]),
            StaticPoint::new([F32(10.107315875917662), F32( 2.4489015959094216)]),
            StaticPoint::new([F32(-7.962477597931141), F32( -5.494741864480315)]),
            StaticPoint::new([F32(10.047917462523671), F32( 5.1631966716389766)]),
            StaticPoint::new([F32(-5.243921934674187), F32( -2.963359100733349)]),
            StaticPoint::new([F32(-9.940544426622527), F32( -3.2655473073528816)]),
            StaticPoint::new([F32(8.30445373000034), F32( 2.129694332932624)]),
            StaticPoint::new([F32(-9.196460281784482), F32( -3.987773678358418)]),
            StaticPoint::new([F32(-10.513583123594056), F32( -2.5364233580562887)]),
            StaticPoint::new([F32(9.072668506714033), F32( 3.405664632524281)]),
            StaticPoint::new([F32(-7.031861004012987), F32( -2.2616818331210844)]),
            StaticPoint::new([F32(9.627963795272553), F32( 4.502533177849574)]),
            StaticPoint::new([F32(-10.442760023564471), F32( -5.0830680881481065)]),
            StaticPoint::new([F32(8.292151321984209), F32( 3.8776876670218834)]),
            StaticPoint::new([F32(-6.51560033683665), F32( -3.8185628318207585)]),
            StaticPoint::new([F32(-10.887633624071544), F32(-4.416570704487158)]),
            StaticPoint::new([F32(-9.465804800021168), F32( -2.2222090878656884)]),
        ];

        let noise_data = vec![
            StaticPoint::new([F32(100.0), F32(100.0)]),
            StaticPoint::new([F32(-100.0), F32(100.0)]),
            StaticPoint::new([F32(100.0), F32(-100.0)]),
            StaticPoint::new([F32(-100.0), F32(-100.0)]),
        ];
        let iso_forest: IsoForest<StaticPoint<F32, 2>> =
            IsoForest::new(&data, 60, 20, 1, 1, F32(0.0), F32(0.1));
        let scores = iso_forest.score(&noise_data);
        let known_scores = vec![0.6145915, 0.56156874, 0.6145915, 0.56156874];
        scores.into_iter().zip(known_scores.into_iter()).for_each(|(s, ks)| {
            assert!((s - F32(ks)).l2_norm() < F32::EPSILON);
        });

        let combined_data = data
            .iter()
            .chain(noise_data.iter())
            .cloned()
            .collect::<Vec<_>>();

        let scores = iso_forest.score(&combined_data);
        let known_scores = vec![
            0.5114004, 0.47956806, 0.45588937, 0.50679964, 0.51736003, 0.45492068, 0.51736003,
            0.49616563, 0.45588937, 0.5175009, 0.454131, 0.470897, 0.512141, 0.4601812, 0.5136253,
            0.470897, 0.5175009, 0.4714252, 0.47527155, 0.45588937, 0.6145915, 0.56156874,
            0.6145915, 0.56156874,
        ];
        scores.into_iter().zip(known_scores.into_iter()).for_each(|(s, ks)| {
            assert!((s - F32(ks)).l2_norm() < F32::EPSILON);
        });
    }

    #[test]
    fn iso_forest_time_static1() {
        let mut rng = PermutedCongruentialGenerator::new(0, 1);
        let x: Vec<F32> = rng.interval(2048);
        let data = x.into_iter().map(|x| StaticPoint::new([x])).collect::<Vec<_>>();
        let now = SystemTime::now();
        let iso_forest: IsoForest<StaticPoint<F32, 1>> =
            IsoForest::new(&data, 128, 128, 0, 1, F32(0.0), F32(0.1));
        let _ = iso_forest.score(&data);
        let _ = println!("{:?}",now.elapsed()); 
    }

    #[test]
    fn iso_forest_time_static2() {
        let mut rng = PermutedCongruentialGenerator::new(0, 1);
        let x: Vec<F32> = rng.interval(2048);
        let y = rng.interval(2048);
        let data = x
            .into_iter()
            .zip(y.into_iter())
            .map(|(x, y)| StaticPoint::new([x, y]))
            .collect::<Vec<_>>();
        let now = SystemTime::now();
        let iso_forest: IsoForest<StaticPoint<F32, 2>> =
            IsoForest::new(&data, 128, 128, 1, 1, F32(0.0), F32(0.1));
        let _ = iso_forest.score(&data);
        let _ = println!("{:?}",now.elapsed()); 
    }

    #[test]
    fn iso_forest_time_single() {
        let mut rng = PermutedCongruentialGenerator::new(0, 1);
        let x: Vec<F32> = rng.interval(2048);
        let data = x;
        let now = SystemTime::now();
        let iso_forest: IsoForest<F32> = IsoForest::new(&data, 128, 128, 0, 1, F32(0.0), F32(0.1));
        let _ = iso_forest.score(&data);
        let _ = println!("{:?}",now.elapsed()); 
    }

    #[test]
    fn iso_forest_time() {
        let mut pcg = PermutedCongruentialGenerator::new(3, 0);
        let data = (0..4096 * 2)
            .map(|_| F32(pcg.next_u32() as f32 / u32::MAX as f32))
            .collect::<Vec<_>>();
        let points = data.chunks_exact(2).map(|chunk| StaticPoint::new([chunk[0],chunk[1]])).collect::<Vec<_>>();
        let now = SystemTime::now();
        let iso = IsoForest::new(&points,128,128,1,1,F32(0.4),F32(0.5));
        let _ = iso.cluster(&points);
        let _ = println!("{:?}", now.elapsed());

        let mut pcg = PermutedCongruentialGenerator::new(3, 0);
        let data = (0..4096 * 2)
            .map(|_| F32(pcg.next_u32() as f32 / u32::MAX as f32))
            .collect::<Vec<_>>();
        let points = data.chunks_exact(3).map(|chunk| StaticPoint::new([chunk[0],chunk[1],chunk[2]])).collect::<Vec<_>>();
        let now = SystemTime::now();
        let iso = IsoForest::new(&points,256,256,1,1,F32(0.4),F32(0.5));
        let _ = iso.cluster(&points);
        let _ = println!("{:?}", now.elapsed());
    }
}
