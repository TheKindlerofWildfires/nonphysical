
#[cfg(test)]
mod dbscan_tests{
    use baseless::{cluster::{dbscan::Dbscan, Classification::Core}, shared::point::{Point, StaticPoint}};
    #[test]
    fn dbscan_simple() {
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
        let dbscan = Dbscan::new(9.0,5);
        let mask = dbscan.cluster(&data);
        let known_mask = vec![Core(0), Core(1), Core(1), Core(1), Core(0), Core(1), Core(0), Core(1), Core(1), Core(0), Core(1), Core(1), Core(0), Core(1), Core(0), Core(1), Core(0), Core(1), Core(1), Core(1)];
        mask.into_iter().zip(known_mask.into_iter()).for_each(|(m,k)|{
            assert!(m==k);
        });
        

    }
}