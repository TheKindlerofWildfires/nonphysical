#[cfg(test)]
mod sscl_tests {
    use std::time::SystemTime;

    use nonphysical_core::{cluster::sscl::SelfSelectiveCompetitiveLearning, shared::{point::{Point, StaticPoint}, primitive::Primitive}};
    use nonphysical_std::shared::primitive::F32;

    #[test]
    fn sscl_basic(){
        let mut sscl: SelfSelectiveCompetitiveLearning<StaticPoint<F32,2>> = SelfSelectiveCompetitiveLearning::new(0,5);
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
        let now = SystemTime::now();
        sscl.cluster(&data);
        dbg!(now.elapsed());
    }


    #[test]
    fn sscl_basic_2(){
        let mut sscl: SelfSelectiveCompetitiveLearning<StaticPoint<F32,2>> = SelfSelectiveCompetitiveLearning::new(0,5);
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
            StaticPoint::new([F32(-100.442760023564471), F32( -5.0830680881481065)]),
            StaticPoint::new([F32(80.292151321984209), F32( 3.8776876670218834)]),
            StaticPoint::new([F32(-89.51560033683665), F32( -3.8185628318207585)]),
            StaticPoint::new([F32(-100.887633624071544), F32(-4.416570704487158)]),
            StaticPoint::new([F32(-90.465804800021168), F32( -2.2222090878656884)]),
        ];
        let now = SystemTime::now();
        sscl.cluster(&data);
        dbg!(now.elapsed());
    }
    #[test]
    fn sscl_large(){
        let mut sscl: SelfSelectiveCompetitiveLearning<StaticPoint<F32,2>> = SelfSelectiveCompetitiveLearning::new(0,5);
        let base_data = vec![
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
        let data:Vec<_> = (0..100).flat_map(|i|{
            base_data.iter().map(|bd| {
                let mut cp = *bd;
                cp.scale(F32::usize(i));
                cp}).collect::<Vec<_>>()
        }).collect(); 
        dbg!(data.len());
        let now = SystemTime::now();
        sscl.cluster(&data);
        dbg!(now.elapsed());
    }

    #[test]
    fn sscl_speed(){
        let mut sscl: SelfSelectiveCompetitiveLearning<StaticPoint<F32,2>> = SelfSelectiveCompetitiveLearning::new(0,50);
        let base_data = vec![
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
        let data:Vec<_> = (0..10000).flat_map(|i|{
            base_data.iter().map(|bd| {
                let mut cp = *bd;
                cp.scale(F32::usize(i));
                cp}).collect::<Vec<_>>()
        }).collect(); 
        dbg!(data.len());
        let now = SystemTime::now();
        sscl.cluster(&data);
        dbg!(now.elapsed());
    }

    #[test]
    fn sscl_speed_2(){
        let mut sscl: SelfSelectiveCompetitiveLearning<F32> = SelfSelectiveCompetitiveLearning::new(0,5);
        let base_data = vec![
            F32(9.308548692822459), F32(2.1673586347139224),
            F32(-5.6424039931897765), F32( -1.9620561766472002),
            F32(-9.821995596375428), F32( -3.1921112766174997),
            F32(-4.992109362834896), F32( -2.0745015313494455),
            F32(10.107315875917662), F32( 2.4489015959094216),
            F32(-7.962477597931141), F32( -5.494741864480315),
            F32(10.047917462523671), F32( 5.1631966716389766),
            F32(-5.243921934674187), F32( -2.963359100733349),
            F32(-9.940544426622527), F32( -3.2655473073528816),
            F32(8.30445373000034), F32( 2.129694332932624),
            F32(-9.196460281784482), F32( -3.987773678358418),
            F32(-10.513583123594056), F32( -2.5364233580562887),
            F32(9.072668506714033), F32( 3.405664632524281),
            F32(-7.031861004012987), F32( -2.2616818331210844),
            F32(9.627963795272553), F32( 4.502533177849574),
            F32(-10.442760023564471), F32( -5.0830680881481065),
            F32(8.292151321984209), F32( 3.8776876670218834),
            F32(-6.51560033683665), F32( -3.8185628318207585),
            F32(-10.887633624071544), F32(-4.416570704487158),
            F32(-9.465804800021168), F32( -2.2222090878656884),
        ];
        let data:Vec<_> = (0..100000).flat_map(|i|{
            base_data.iter().map(|bd| {
                let mut cp = *bd;
                cp.scale(F32::usize(i));
                cp}).collect::<Vec<_>>()
        }).collect(); 
        dbg!(data.len());
        let now = SystemTime::now();
        sscl.cluster(&data);
        dbg!(now.elapsed());
    }
}
