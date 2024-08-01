
#[cfg(test)]
mod hdbscan_tests {
    use crate::{random::pcg::PermutedCongruentialGenerator, shared::point::StaticPoint};
    use std::time::SystemTime;

    use super::*;

    #[test]
    fn hdbscan_simple() {
        let data = vec![
            StaticPoint::new([9.308548692822459, 2.1673586347139224]),
            StaticPoint::new([-5.6424039931897765, -1.9620561766472002]),
            StaticPoint::new([-9.821995596375428, -3.1921112766174997]),
            StaticPoint::new([-4.992109362834896, -2.0745015313494455]),
            StaticPoint::new([10.107315875917662, 2.4489015959094216]),
            StaticPoint::new([-7.962477597931141, -5.494741864480315]),
            StaticPoint::new([10.047917462523671, 5.1631966716389766]),
            StaticPoint::new([-5.243921934674187, -2.963359100733349]),
            StaticPoint::new([-9.940544426622527, -3.2655473073528816]),
            StaticPoint::new([8.30445373000034, 2.129694332932624]),
            StaticPoint::new([-9.196460281784482, -3.987773678358418]),
            StaticPoint::new([-10.513583123594056, -2.5364233580562887]),
            StaticPoint::new([9.072668506714033, 3.405664632524281]),
            StaticPoint::new([-7.031861004012987, -2.2616818331210844]),
            StaticPoint::new([9.627963795272553, 4.502533177849574]),
            StaticPoint::new([-10.442760023564471, -5.0830680881481065]),
            StaticPoint::new([8.292151321984209, 3.8776876670218834]),
            StaticPoint::new([-6.51560033683665, -3.8185628318207585]),
            StaticPoint::new([-10.887633624071544, -4.416570704487158]),
            StaticPoint::new([-9.465804800021168, -2.2222090878656884]),
        ];
        let hdbscan = Hdbscan::new(3, 20, false, 5);
        let mask = hdbscan.cluster(&data);
        let known_mask = if mask[0] == Core(0) {
            vec![
                Core(0),
                Core(1),
                Core(1),
                Core(1),
                Core(0),
                Core(1),
                Core(0),
                Core(1),
                Core(1),
                Core(0),
                Core(1),
                Core(1),
                Core(0),
                Core(1),
                Core(0),
                Core(1),
                Core(0),
                Core(1),
                Core(1),
                Core(1),
            ]
        } else {
            vec![
                Core(1),
                Core(0),
                Core(0),
                Core(0),
                Core(1),
                Core(0),
                Core(1),
                Core(0),
                Core(0),
                Core(1),
                Core(0),
                Core(0),
                Core(1),
                Core(0),
                Core(1),
                Core(0),
                Core(1),
                Core(0),
                Core(0),
                Core(0),
            ]
        };
        mask.into_iter().zip(known_mask.into_iter()).for_each(|(m, k)| {
            assert!(m == k);
        });
    }

    #[test]
    fn hdbscan_single_on() {
        let data = vec![
            StaticPoint::new([9.308548692822459, 2.1673586347139224]),
            StaticPoint::new([10.107315875917662, 2.4489015959094216]),
            StaticPoint::new([10.047917462523671, 5.1631966716389766]),
            StaticPoint::new([8.30445373000034, 2.129694332932624]),
            StaticPoint::new([9.072668506714033, 3.405664632524281]),
            StaticPoint::new([9.627963795272553, 4.502533177849574]),
            StaticPoint::new([8.292151321984209, 3.8776876670218834]),
            StaticPoint::new([10.308548692822459, 2.1673586347139224]),
            StaticPoint::new([11.107315875917662, 2.4489015959094216]),
            StaticPoint::new([11.047917462523671, 5.1631966716389766]),
            StaticPoint::new([9.30445373000034, 2.129694332932624]),
            StaticPoint::new([10.072668506714033, 4.405664632524281]),
            StaticPoint::new([10.627963795272553, 5.502533177849574]),
            StaticPoint::new([9.292151321984209, 4.8776876670218834]),
            StaticPoint::new([9.308548692822459, 3.1673586347139224]),
            StaticPoint::new([10.107315875917662, 3.4489015959094216]),
            StaticPoint::new([10.047917462523671, 4.1631966716389766]),
            StaticPoint::new([8.30445373000034, 3.129694332932624]),
            StaticPoint::new([9.072668506714033, 4.405664632524281]),
            StaticPoint::new([9.627963795272553, 5.502533177849574]),
            StaticPoint::new([8.292151321984209, 4.8776876670218834]),
            StaticPoint::new([10.308548692822459, 3.1673586347139224]),
            StaticPoint::new([11.107315875917662, 3.4489015959094216]),
            StaticPoint::new([11.047917462523671, 4.1631966716389766]),
            StaticPoint::new([9.30445373000034, 3.129694332932624]),
            StaticPoint::new([10.072668506714033, 4.405664632524281]),
            StaticPoint::new([10.627963795272553, 5.502533177849574]),
            StaticPoint::new([9.292151321984209, 4.8776876670218834]),
        ];

        let hdbscan = Hdbscan::new(3, 20, true, 5);
        let mask = hdbscan.cluster(&data);
        mask.into_iter().for_each(|m| {
            assert!(m == Core(0));
        });
    }
    #[test]
    fn hdbscan_single_off() {
        let data = vec![
            StaticPoint::new([9.308548692822459, 2.1673586347139224]),
            StaticPoint::new([10.107315875917662, 2.4489015959094216]),
            StaticPoint::new([10.047917462523671, 5.1631966716389766]),
            StaticPoint::new([8.30445373000034, 2.129694332932624]),
            StaticPoint::new([9.072668506714033, 3.405664632524281]),
            StaticPoint::new([9.627963795272553, 4.502533177849574]),
            StaticPoint::new([8.292151321984209, 3.8776876670218834]),
            StaticPoint::new([10.308548692822459, 2.1673586347139224]),
            StaticPoint::new([11.107315875917662, 2.4489015959094216]),
            StaticPoint::new([11.047917462523671, 5.1631966716389766]),
            StaticPoint::new([9.30445373000034, 2.129694332932624]),
            StaticPoint::new([10.072668506714033, 4.405664632524281]),
            StaticPoint::new([10.627963795272553, 5.502533177849574]),
            StaticPoint::new([9.292151321984209, 4.8776876670218834]),
            StaticPoint::new([9.308548692822459, 3.1673586347139224]),
            StaticPoint::new([10.107315875917662, 3.4489015959094216]),
            StaticPoint::new([10.047917462523671, 4.1631966716389766]),
            StaticPoint::new([8.30445373000034, 3.129694332932624]),
            StaticPoint::new([9.072668506714033, 4.405664632524281]),
            StaticPoint::new([9.627963795272553, 5.502533177849574]),
            StaticPoint::new([8.292151321984209, 4.8776876670218834]),
            StaticPoint::new([10.308548692822459, 3.1673586347139224]),
            StaticPoint::new([11.107315875917662, 3.4489015959094216]),
            StaticPoint::new([11.047917462523671, 4.1631966716389766]),
            StaticPoint::new([9.30445373000034, 3.129694332932624]),
            StaticPoint::new([10.072668506714033, 4.405664632524281]),
            StaticPoint::new([10.627963795272553, 5.502533177849574]),
            StaticPoint::new([9.292151321984209, 4.8776876670218834]),
        ];
        let hdbscan = Hdbscan::new(3, 20, false, 5);
        let mask = hdbscan.cluster(&data);
        dbg!(&mask);
        let known_mask = vec![
            Noise,
            Noise,
            Core(0),
            Noise,
            Noise,
            Core(0),
            Noise,
            Noise,
            Noise,
            Noise,
            Noise,
            Core(0),
            Noise,
            Core(0),
            Noise,
            Core(0),
            Core(0),
            Noise,
            Core(0),
            Noise,
            Noise,
            Noise,
            Noise,
            Noise,
            Noise,
            Core(0),
            Noise,
            Core(0),
        ];

        mask.into_iter().zip(known_mask.into_iter()).for_each(|(m, k)| {
            assert!(m == k);
        });
    }

    #[test]
    fn hdbscan_time() {
        let mut data = Vec::new();
        (0..20000).for_each(|i| data.push(StaticPoint::new([f32::usize(i), f32::usize(i + 1)])));
        let now = SystemTime::now();
        let hdbscan = Hdbscan::new(2, 1000, false, 32);
        let _ = hdbscan.cluster(&data);
        let _ = dbg!(now.elapsed()); //about twice as fast as python.. but could it be better?

        let mut data = Vec::new();
        (0..20000).for_each(|i| data.push(i as f32));
        let now = SystemTime::now();
        let hdbscan = Hdbscan::new(2, 1000, false, 32);
        let _ = hdbscan.cluster(&data);
        let _ = dbg!(now.elapsed()); //faster, but not by much
    }

    #[test]
    fn hdbscan_r_time() {
        let mut pcg = PermutedCongruentialGenerator::new(3, 0);
        let data = (0..2048 * 2)
            .map(|_| pcg.next_u32() as f32 / u32::MAX as f32)
            .collect::<Vec<_>>();
        let points = data.chunks_exact(2).map(|chunk| StaticPoint::new([chunk[0],chunk[1]])).collect::<Vec<_>>();
        let now = SystemTime::now();
        let hdb = Hdbscan::new(16, 2048, true, 16);
        let _ = hdb.cluster(&points);
        let _ = println!("{:?}", now.elapsed());

        let mut pcg = PermutedCongruentialGenerator::new(3, 0);
        let data = (0..4096 * 3)
            .map(|_| pcg.next_u32() as f32 / u32::MAX as f32)
            .collect::<Vec<_>>();
        let points = data.chunks_exact(3).map(|chunk| StaticPoint::new([chunk[0],chunk[1],chunk[2]])).collect::<Vec<_>>();
        let now = SystemTime::now();
        let hdb = Hdbscan::new(16, 2048, true, 16);
        let _ = hdb.cluster(&points);
        let _ = println!("{:?}", now.elapsed());
    }
}
