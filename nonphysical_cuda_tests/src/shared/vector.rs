#[cfg(test)]
mod vector_tests {
    use nonphysical_core::shared::{primitive::Primitive, vector::{float_vector::FloatVector, Vector}};
    use nonphysical_cuda::cuda::runtime::Runtime;
    use nonphysical_ptx::shared::vector::cuda_vector::CudaVector;
    use nonphysical_std::shared::primitive::F32;
    use nonphysical_core::shared::float::Float;
    #[test]
    fn sum(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::sum(data.iter());
        let known = FloatVector::sum(data.iter());
        //assert!((out-known).l2_norm()<F32::EPSILON);
    }
    #[test]
    fn product(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(1..32).map(|i| F32::usize(1)).collect::<Vec<_>>();
        let out = CudaVector::product(data.iter());
        let known = FloatVector::product(data.iter());
        let z:F32 = FloatVector::product([].iter().skip(1).step_by(2) );
        //assert!((out-known).l2_norm()<F32::EPSILON);
    }
    #[test]
    fn greater(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::greater(data.iter());
        let known = FloatVector::greater(data.iter());
        assert!((out-known).l2_norm()<F32::EPSILON);
    }
    #[test]
    fn lesser(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::lesser(data.iter());
        let known = FloatVector::lesser(data.iter());
        assert!((out-known).l2_norm()<F32::EPSILON);
    }
    #[test]
    fn mean(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::mean(data.iter());
        let known = FloatVector::mean(data.iter());
        assert!((out-known).l2_norm()<F32::EPSILON);
    }
    #[test]
    fn variance(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let (out_m, out_v) = CudaVector::variance(data.iter());
        let (known_m,known_v) = FloatVector::variance(data.iter());
        assert!((out_m-known_m).l2_norm()<F32::EPSILON);
        assert!((out_v-known_v).l2_norm()<F32::EPSILON);

    }
    #[test]
    fn deviation(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let (out_m, out_d) = CudaVector::deviation(data.iter());
        let (known_m,known_d) = FloatVector::deviation(data.iter());
        assert!((out_m-known_m).l2_norm()<F32::EPSILON);
        assert!((out_d-known_d).l2_norm()<F32::EPSILON);
    }
    #[test]
    fn add(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::add(data.iter(),F32::usize(2));
        let known = FloatVector::add(data.iter(),F32::usize(2));
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn sub(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::sub(data.iter(),F32::usize(2));
        let known = FloatVector::sub(data.iter(),F32::usize(2));
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn mul(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::mul(data.iter(),F32::usize(2));
        let known = FloatVector::mul(data.iter(),F32::usize(2));
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn div(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::div(data.iter(),F32::usize(2));
        let known = FloatVector::div(data.iter(),F32::usize(2));
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn scale(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::scale(data.iter(),F32::usize(2));
        let known = FloatVector::scale(data.iter(),F32::usize(2));
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn descale(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::descale(data.iter(),F32::usize(2));
        let known = FloatVector::descale(data.iter(),F32::usize(2));
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn fma(){
        return;
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::fma(data.iter(),F32::usize(2), F32::usize(3));
        let known = FloatVector::fma(data.iter(),F32::usize(2), F32::usize(3));
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn powf(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::powf(data.iter(),F32::usize(2));
        let known = FloatVector::powf(data.iter(),F32::usize(2));
        out.zip(known).for_each(|(o,k)|{
            assert!(((o-k)/(o+k+F32::ONE)).l2_norm()<F32::ONE);
        });
    }
    #[test]
    fn ln(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(1..1024*1024+1).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::ln(data.iter());
        let known = FloatVector::ln(data.iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn log2(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(1..1024*1024+1).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::log2(data.iter());
        let known = FloatVector::log2(data.iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn exp(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::exp(data.iter());
        let known = FloatVector::exp(data.iter());
        out.zip(known).for_each(|(o,k)|{
            assert!(((o-k)/(o+k+F32::ONE)).l2_norm()<F32::ONE);
        });
    }
    #[test]
    fn exp2(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();
        let out = CudaVector::exp2(data.iter());
        let known = FloatVector::exp2(data.iter());
        out.zip(known).for_each(|(o,k)|{
            assert!(((o-k)/(o+k+F32::ONE)).l2_norm()<F32::ONE);
        });
    }
    #[test]
    fn recip(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(1..1024*1024+1).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();
        let out = CudaVector::recip(data.iter());
        let known = FloatVector::recip(data.iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::ONE);
        });
    }
    #[test]
    fn sin(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::sin(data.iter());
        let known = FloatVector::sin(data.iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::ONE);
        });
    }
    #[test]
    fn cos(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::cos(data.iter());
        let known = FloatVector::cos(data.iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::ONE);
        });
    }
    #[test]
    fn tan(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::tan(data.iter());
        let known = FloatVector::tan(data.iter());
        out.zip(known).for_each(|(o,k)|{
            assert!(((o-k)/(o+k+F32::ONE)).l2_norm()<F32::ONE);
        });
    }
    #[test]
    fn asin(){
        return;//No device support
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::asin(data.iter());
        let known = FloatVector::asin(data.iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn acos(){
        return;//No device support
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::acos(data.iter());
        let known = FloatVector::acos(data.iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn atan(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::atan(data.iter());
        let known = FloatVector::atan(data.iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn sinh(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();
        let out = CudaVector::sinh(data.iter());
        let known = FloatVector::sinh(data.iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn cosh(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();
        let out = CudaVector::cosh(data.iter());
        let known = FloatVector::cosh(data.iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn tanh(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();
        let out = CudaVector::tanh(data.iter());
        let known = FloatVector::tanh(data.iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn asinh(){
        return;//No device support
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::asinh(data.iter());
        let known = FloatVector::asinh(data.iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn acosh(){
        return;//No device support
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::acosh(data.iter());
        let known = FloatVector::acosh(data.iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn atanh(){
        return;//No device support
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::atanh(data.iter());
        let known = FloatVector::atanh(data.iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn l1_norm(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::l1_norm(data.iter());
        let known = FloatVector::l1_norm(data.iter());
        out.zip(known).for_each(|(o,k)|{
            assert!(((o-k)/(o+k+F32::ONE)).l2_norm()<F32::ONE);
        });
    }
    #[test]
    fn l2_norm(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::l2_norm(data.iter());
        let known = FloatVector::l2_norm(data.iter());
        out.zip(known).for_each(|(o,k)|{
            assert!(((o-k)/(o+k+F32::ONE)).l2_norm()<F32::ONE);
        });
    }
    #[test]
    fn add_ref(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data_t =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let mut data_k =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        CudaVector::add_ref(data_t.iter_mut(), F32::usize(2));
        FloatVector::add_ref(data_k.iter_mut(), F32::usize(2));
        data_t.iter().zip(data_k.iter()).for_each(|(o,k)|{
            assert!((*o-*k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn sub_ref(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data_t =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let mut data_k =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        CudaVector::sub_ref(data_t.iter_mut(), F32::usize(2));
        FloatVector::sub_ref(data_k.iter_mut(), F32::usize(2));
        data_t.iter().zip(data_k.iter()).for_each(|(o,k)|{
            assert!((*o-*k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn mul_ref(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data_t =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let mut data_k =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        CudaVector::mul_ref(data_t.iter_mut(), F32::usize(2));
        FloatVector::mul_ref(data_k.iter_mut(), F32::usize(2));
        data_t.iter().zip(data_k.iter()).for_each(|(o,k)|{
            assert!((*o-*k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn div_ref(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data_t =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let mut data_k =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        CudaVector::div_ref(data_t.iter_mut(), F32::usize(2));
        FloatVector::div_ref(data_k.iter_mut(), F32::usize(2));
        data_t.iter().zip(data_k.iter()).for_each(|(o,k)|{
            assert!((*o-*k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn scale_ref(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data_t =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let mut data_k =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        CudaVector::scale_ref(data_t.iter_mut(), F32::usize(2));
        FloatVector::scale_ref(data_k.iter_mut(), F32::usize(2));
        data_t.iter().zip(data_k.iter()).for_each(|(o,k)|{
            assert!((*o-*k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn descale_ref(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data_t =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let mut data_k =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        CudaVector::descale_ref(data_t.iter_mut(), F32::usize(2));
        FloatVector::descale_ref(data_k.iter_mut(), F32::usize(2));
        data_t.iter().zip(data_k.iter()).for_each(|(o,k)|{
            assert!((*o-*k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn fma_ref(){
        return;
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data_t =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let mut data_k =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        CudaVector::fma_ref(data_t.iter_mut(), F32::usize(2),F32::usize(3));
        FloatVector::fma_ref(data_k.iter_mut(), F32::usize(2),F32::usize(3));
        data_t.iter().zip(data_k.iter()).for_each(|(o,k)|{
            assert!((*o-*k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn powf_ref(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data_t =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let mut data_k =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        CudaVector::powf_ref(data_t.iter_mut(), F32::usize(2));
        FloatVector::powf_ref(data_k.iter_mut(), F32::usize(2));
        data_t.iter().zip(data_k.iter()).for_each(|(o,k)|{
            assert!(((*o-*k)/(*o+*k+F32::ONE)).l2_norm()<F32::ONE);
        });
    }

    #[test]
    fn ln_ref(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data_t =(1..1024*1024+1).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let mut data_k =(1..1024*1024+1).map(|i| F32::usize(i)).collect::<Vec<_>>();
        CudaVector::ln_ref(data_t.iter_mut());
        FloatVector::ln_ref(data_k.iter_mut());
        data_t.iter().zip(data_k.iter()).for_each(|(o,k)|{
            assert!((*o-*k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn log2_ref(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data_t =(1..1024*1024+1).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let mut data_k =(1..1024*1024+1).map(|i| F32::usize(i)).collect::<Vec<_>>();
        CudaVector::log2_ref(data_t.iter_mut());
        FloatVector::log2_ref(data_k.iter_mut());
        data_t.iter().zip(data_k.iter()).for_each(|(o,k)|{
            assert!((*o-*k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn exp_ref(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data_t =(0..1024*1024).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();
        let mut data_k =(0..1024*1024).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();
        CudaVector::exp_ref(data_t.iter_mut());
        FloatVector::exp_ref(data_k.iter_mut());
        data_t.iter().zip(data_k.iter()).for_each(|(o,k)|{
            assert!(((*o-*k)/(*o+*k+F32::ONE)).l2_norm()<F32::ONE);
        });
    }
    #[test]
    fn exp2_ref(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data_t =(0..1024*1024).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();
        let mut data_k =(0..1024*1024).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();
        CudaVector::exp2_ref(data_t.iter_mut());
        FloatVector::exp2_ref(data_k.iter_mut());
        data_t.iter().zip(data_k.iter()).for_each(|(o,k)|{
            assert!((*o-*k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn recip_ref(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data_t =(1..1024*1024+1).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();
        let mut data_k =(1..1024*1024+1).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();
        CudaVector::recip_ref(data_t.iter_mut());
        FloatVector::recip_ref(data_k.iter_mut());
        data_t.iter().zip(data_k.iter()).for_each(|(o,k)|{
            assert!(((*o-*k)/(*o+*k+F32::ONE)).l2_norm()<F32::ONE);
        });
    }
    #[test]
    fn sin_ref(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data_t =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let mut data_k =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        CudaVector::sin_ref(data_t.iter_mut());
        FloatVector::sin_ref(data_k.iter_mut());
        data_t.iter().zip(data_k.iter()).for_each(|(o,k)|{
            assert!(((*o-*k)/((*o+*k).l2_norm()+F32::ONE)).l2_norm()<F32::ONE);
        });
    }
    #[test]
    fn cos_ref(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data_t =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let mut data_k =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        CudaVector::cos_ref(data_t.iter_mut());
        FloatVector::cos_ref(data_k.iter_mut());
        data_t.iter().zip(data_k.iter()).for_each(|(o,k)|{
            assert!(((*o-*k)/((*o+*k).l2_norm()+F32::ONE)).l2_norm()<F32::ONE);
        });
    }
    #[test]
    fn tan_ref(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data_t =(0..1024*1024).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();
        let mut data_k =(0..1024*1024).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();
        CudaVector::tan_ref(data_t.iter_mut());
        FloatVector::tan_ref(data_k.iter_mut());
        data_t.iter().zip(data_k.iter()).for_each(|(o,k)|{
            assert!(((*o-*k)/((*o+*k).l2_norm()+F32::ONE)).l2_norm()<F32::ONE);
        });
    }
    #[test]
    fn asin_ref(){
        return;//no device support
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data_t =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let mut data_k =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        CudaVector::asin_ref(data_t.iter_mut());
        FloatVector::asin_ref(data_k.iter_mut());
        data_t.iter().zip(data_k.iter()).for_each(|(o,k)|{
            assert!((*o-*k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn acos_ref(){
        return;//no device support
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data_t =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let mut data_k =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        CudaVector::acos_ref(data_t.iter_mut());
        FloatVector::acos_ref(data_k.iter_mut());
        data_t.iter().zip(data_k.iter()).for_each(|(o,k)|{
            assert!((*o-*k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn atan_ref(){
        return;//no device support
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data_t =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let mut data_k =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        CudaVector::atan_ref(data_t.iter_mut());
        FloatVector::atan_ref(data_k.iter_mut());
        data_t.iter().zip(data_k.iter()).for_each(|(o,k)|{
            assert!((*o-*k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn sinh_ref(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data_t =(0..1024*1024).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();
        let mut data_k =(0..1024*1024).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();
        CudaVector::sinh_ref(data_t.iter_mut());
        FloatVector::sinh_ref(data_k.iter_mut());
        data_t.iter().zip(data_k.iter()).for_each(|(o,k)|{
            assert!(((*o-*k)/((*o+*k).l2_norm()+F32::ONE)).l2_norm()<F32::ONE);
        });
    }
    #[test]
    fn cosh_ref(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data_t =(0..1024*1024).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();
        let mut data_k =(0..1024*1024).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();
        CudaVector::cosh_ref(data_t.iter_mut());
        FloatVector::cosh_ref(data_k.iter_mut());
        data_t.iter().zip(data_k.iter()).for_each(|(o,k)|{
            assert!(((*o-*k)/((*o+*k).l2_norm()+F32::ONE)).l2_norm()<F32::ONE);
        });
    }
    #[test]
    fn tanh_ref(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data_t =(0..1024*1024).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();
        let mut data_k =(0..1024*1024).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();
        CudaVector::tanh_ref(data_t.iter_mut());
        FloatVector::tanh_ref(data_k.iter_mut());
        data_t.iter().zip(data_k.iter()).for_each(|(o,k)|{
            assert!(((*o-*k)/((*o+*k).l2_norm()+F32::ONE)).l2_norm()<F32::ONE);
        });
    }
    #[test]
    fn asinh_ref(){
        return;//no device support
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data_t =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let mut data_k =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        CudaVector::asinh_ref(data_t.iter_mut());
        FloatVector::asinh_ref(data_k.iter_mut());
        data_t.iter().zip(data_k.iter()).for_each(|(o,k)|{
            assert!((*o-*k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn acosh_ref(){
        return;//no device support
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data_t =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let mut data_k =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        CudaVector::acosh_ref(data_t.iter_mut());
        FloatVector::acosh_ref(data_k.iter_mut());
        data_t.iter().zip(data_k.iter()).for_each(|(o,k)|{
            assert!((*o-*k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn atanh_ref(){
        return;//no device support
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data_t =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let mut data_k =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        CudaVector::atanh_ref(data_t.iter_mut());
        FloatVector::atanh_ref(data_k.iter_mut());
        data_t.iter().zip(data_k.iter()).for_each(|(o,k)|{
            assert!((*o-*k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn add_vec(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let vec = (0..1024*1024).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();
        let out = CudaVector::add_vec(data.iter(),vec.iter());
        let known = FloatVector::add_vec(data.iter(),vec.iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn sub_vec(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let vec = (0..1024*1024).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();
        let out = CudaVector::sub_vec(data.iter(),vec.iter());
        let known = FloatVector::sub_vec(data.iter(),vec.iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn mul_vec(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let vec = (0..1024*1024).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();
        let out = CudaVector::mul_vec(data.iter(),vec.iter());
        let known = FloatVector::mul_vec(data.iter(),vec.iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn div_vec(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(1..1024*1024+1).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let vec = (1..1024*1024+1).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();
        let out = CudaVector::div_vec(data.iter(),vec.iter());
        let known = FloatVector::div_vec(data.iter(),vec.iter());
        out.zip(known).for_each(|(o,k)|{
            assert!(((o-k)/((o+k).l2_norm()+F32::ONE)).l2_norm()<F32::ONE);
        });
    }
    #[test]
    fn fma_vec(){
        return;
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let mul = (0..1024*1024).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();
        let add = (0..1024*1024).map(|i| F32::usize(i).cos()).collect::<Vec<_>>();

        let out = CudaVector::fma_vec(data.iter(),mul.iter(), add.iter());
        let known = FloatVector::fma_vec(data.iter(),mul.iter(), add.iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn powf_vec(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let vec = (1..1024*1024+1).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();
        let out = CudaVector::div_vec(data.iter(),vec.iter());
        let known = FloatVector::div_vec(data.iter(),vec.iter());
        out.zip(known).for_each(|(o,k)|{
            assert!(((o-k)/((o+k).l2_norm()+F32::ONE)).l2_norm()<F32::ONE);
        });
    }
    #[test]
    fn greater_vec(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let vec = (0..1024*1024).map(|i| F32::usize(i).sin()*F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::greater_vec(data.iter(),vec.iter());
        let known = FloatVector::greater_vec(data.iter(),vec.iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn lesser_vec(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let vec = (0..1024*1024).map(|i| F32::usize(i).sin()*F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::lesser_vec(data.iter(),vec.iter());
        let known = FloatVector::lesser_vec(data.iter(),vec.iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn add_vec_ref(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let mut known =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let vec = (0..1024*1024).map(|i| F32::usize(i).sin()*F32::usize(i)).collect::<Vec<_>>();
        CudaVector::add_vec_ref(data.iter_mut(),vec.iter());
        FloatVector::add_vec_ref(known.iter_mut(),vec.iter());
        data.iter().zip(known.iter()).for_each(|(o,k)|{
            assert!((*o-*k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn sub_vec_ref(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let mut known =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let vec = (0..1024*1024).map(|i| F32::usize(i).sin()*F32::usize(i)).collect::<Vec<_>>();
        CudaVector::add_vec_ref(data.iter_mut(),vec.iter());
        FloatVector::add_vec_ref(known.iter_mut(),vec.iter());
        data.iter().zip(known.iter()).for_each(|(o,k)|{
            assert!((*o-*k).l2_norm()<F32::EPSILON);
        });    }
    #[test]
    fn mul_vec_ref(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let mut known =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let vec = (0..1024*1024).map(|i| F32::usize(i).sin()*F32::usize(i)).collect::<Vec<_>>();
        CudaVector::add_vec_ref(data.iter_mut(),vec.iter());
        FloatVector::add_vec_ref(known.iter_mut(),vec.iter());
        data.iter().zip(known.iter()).for_each(|(o,k)|{
            assert!((*o-*k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn div_vec_ref(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let mut known =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let vec = (0..1024*1024).map(|i| F32::usize(i).sin()*F32::usize(i)).collect::<Vec<_>>();
        CudaVector::add_vec_ref(data.iter_mut(),vec.iter());
        FloatVector::add_vec_ref(known.iter_mut(),vec.iter());
        data.iter().zip(known.iter()).for_each(|(o,k)|{
            assert!((*o-*k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn fma_vec_ref(){
        return;//
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let mut known =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let vec = (0..1024*1024).map(|i| F32::usize(i).sin()*F32::usize(i)).collect::<Vec<_>>();
        CudaVector::add_vec_ref(data.iter_mut(),vec.iter());
        FloatVector::add_vec_ref(known.iter_mut(),vec.iter());
        data.iter().zip(known.iter()).for_each(|(o,k)|{
            assert!((*o-*k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn powf_vec_ref(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let mut known =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let vec = (0..1024*1024).map(|i| F32::usize(i).sin()*F32::usize(i)).collect::<Vec<_>>();
        CudaVector::add_vec_ref(data.iter_mut(),vec.iter());
        FloatVector::add_vec_ref(known.iter_mut(),vec.iter());
        data.iter().zip(known.iter()).for_each(|(o,k)|{
            assert!((*o-*k).l2_norm()<F32::EPSILON);
        });
    }

    #[test]
    fn greater_vec_ref(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let mut known =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let vec = (0..1024*1024).map(|i| F32::usize(i).sin()*F32::usize(i)).collect::<Vec<_>>();
        CudaVector::add_vec_ref(data.iter_mut(),vec.iter());
        FloatVector::add_vec_ref(known.iter_mut(),vec.iter());
        data.iter().zip(known.iter()).for_each(|(o,k)|{
            assert!((*o-*k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn lesser_vec_ref(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let mut data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let mut known =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let vec = (0..1024*1024).map(|i| F32::usize(i).sin()*F32::usize(i)).collect::<Vec<_>>();
        CudaVector::add_vec_ref(data.iter_mut(),vec.iter());
        FloatVector::add_vec_ref(known.iter_mut(),vec.iter());
        data.iter().zip(known.iter()).for_each(|(o,k)|{
            assert!((*o-*k).l2_norm()<F32::EPSILON);
        }); 
    }
    #[test]
    fn dot(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let vec = (0..1024*1024).map(|i| F32::usize(i).sin()*F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::dot(data.iter(),vec.iter());
        let known = FloatVector::dot(data.iter(),vec.iter());
        assert!((out-known).l2_norm()<F32::EPSILON);
    }
    #[test]
    fn quote(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let vec = (0..1024*1024).map(|i| F32::usize(i).sin()*F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::quote(data.iter(),vec.iter());
        let known = FloatVector::quote(data.iter(),vec.iter());
        assert!((out-known).l2_norm()<F32::EPSILON);
    }
    #[test]
    fn add_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();

        let out = CudaVector::add_direct(data.into_iter(),F32::usize(2));
        let known = FloatVector::add_direct(reference.into_iter(),F32::usize(2));
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn sub_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();

        let out = CudaVector::sub_direct(data.into_iter(),F32::usize(2));
        let known = FloatVector::sub_direct(reference.into_iter(),F32::usize(2));
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn mul_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();

        let out = CudaVector::mul_direct(data.into_iter(),F32::usize(2));
        let known = FloatVector::mul_direct(reference.into_iter(),F32::usize(2));
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn div_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();

        let out = CudaVector::div_direct(data.into_iter(),F32::usize(2));
        let known = FloatVector::div_direct(reference.into_iter(),F32::usize(2));
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn scale_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();

        let out = CudaVector::scale_direct(data.into_iter(),F32::usize(2));
        let known = FloatVector::scale_direct(reference.into_iter(),F32::usize(2));
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn descale_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();

        let out = CudaVector::descale_direct(data.into_iter(),F32::usize(2));
        let known = FloatVector::descale_direct(reference.into_iter(),F32::usize(2));
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn fma_direct(){
        return;
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();

        let out = CudaVector::fma_direct(data.into_iter(),F32::usize(2), F32::usize(3));
        let known = FloatVector::fma_direct(reference.into_iter(),F32::usize(2), F32::usize(3));
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn powf_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();

        let out = CudaVector::powf_direct(data.into_iter(),F32::usize(2));
        let known = FloatVector::powf_direct(reference.into_iter(),F32::usize(2));
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn ln_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();

        let out = CudaVector::ln_direct(data.into_iter());
        let known = FloatVector::ln_direct(reference.into_iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn log2_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();

        let out = CudaVector::log2_direct(data.into_iter());
        let known = FloatVector::log2_direct(reference.into_iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn exp_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();

        let out = CudaVector::exp_direct(data.into_iter());
        let known = FloatVector::exp_direct(reference.into_iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn exp2_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();

        let out = CudaVector::exp2_direct(data.into_iter());
        let known = FloatVector::exp2_direct(reference.into_iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn recip_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();

        let out = CudaVector::recip_direct(data.into_iter());
        let known = FloatVector::recip_direct(reference.into_iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn sin_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();

        let out = CudaVector::sin_direct(data.into_iter());
        let known = FloatVector::sin_direct(reference.into_iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn cos_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();

        let out = CudaVector::cos_direct(data.into_iter());
        let known = FloatVector::cos_direct(reference.into_iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn tan_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();

        let out = CudaVector::tan_direct(data.into_iter());
        let known = FloatVector::tan_direct(reference.into_iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn asin_direct(){
        return;

        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();

        let out = CudaVector::asin_direct(data.into_iter());
        let known = FloatVector::asin_direct(reference.into_iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn acos_direct(){
        return;
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();

        let out = CudaVector::acos_direct(data.into_iter());
        let known = FloatVector::acos_direct(reference.into_iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn atan_direct(){
        return;

        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();

        let out = CudaVector::atan_direct(data.into_iter());
        let known = FloatVector::atan_direct(reference.into_iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn sinh_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();

        let out = CudaVector::sinh_direct(data.into_iter());
        let known = FloatVector::sinh_direct(reference.into_iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn cosh_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();

        let out = CudaVector::cosh_direct(data.into_iter());
        let known = FloatVector::cosh_direct(reference.into_iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn tanh_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();

        let out = CudaVector::tanh_direct(data.into_iter());
        let known = FloatVector::tanh_direct(reference.into_iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn asinh_direct(){
        return;

        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();

        let out = CudaVector::asinh_direct(data.into_iter());
        let known = FloatVector::asinh_direct(reference.into_iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn acosh_direct(){
        return;

        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();

        let out = CudaVector::acosh_direct(data.into_iter());
        let known = FloatVector::acosh_direct(reference.into_iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn atanh_direct(){
        return;

        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();

        let out = CudaVector::atanh_direct(data.into_iter());
        let known = FloatVector::atanh_direct(reference.into_iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn l1_norm_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();

        let out = CudaVector::l1_norm_direct(data.into_iter());
        let known = FloatVector::l1_norm_direct(reference.into_iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn l2_norm_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();

        let out = CudaVector::l2_norm_direct(data.into_iter());
        let known = FloatVector::l2_norm_direct(reference.into_iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn add_vec_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let vec = (0..1024*1024).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();
        let ref_vec = (0..1024*1024).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();

        let out = CudaVector::add_vec_direct(data.into_iter(),vec.into_iter());
        let known = FloatVector::add_vec_direct(reference.into_iter(),ref_vec.into_iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn sub_vec_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let vec = (0..1024*1024).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();
        let ref_vec = (0..1024*1024).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();

        let out = CudaVector::add_vec_direct(data.into_iter(),vec.into_iter());
        let known = FloatVector::add_vec_direct(reference.into_iter(),ref_vec.into_iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn mul_vec_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let vec = (0..1024*1024).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();
        let ref_vec = (0..1024*1024).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();

        let out = CudaVector::add_vec_direct(data.into_iter(),vec.into_iter());
        let known = FloatVector::add_vec_direct(reference.into_iter(),ref_vec.into_iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn div_vec_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let vec = (0..1024*1024).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();
        let ref_vec = (0..1024*1024).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();

        let out = CudaVector::add_vec_direct(data.into_iter(),vec.into_iter());
        let known = FloatVector::add_vec_direct(reference.into_iter(),ref_vec.into_iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn fma_vec_direct(){
        return;
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let vec_mul = (0..1024*1024).map(|i| F32::usize(i).cos()).collect::<Vec<_>>();
        let ref_vec_mul = (0..1024*1024).map(|i| F32::usize(i).cos()).collect::<Vec<_>>();

        let vec_add = (0..1024*1024).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();
        let ref_vec_add = (0..1024*1024).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();

        let out = CudaVector::fma_vec_direct(data.into_iter(),vec_mul.into_iter(),vec_add.into_iter());
        let known = FloatVector::fma_vec_direct(reference.into_iter(),ref_vec_mul.into_iter(),ref_vec_add.into_iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn powf_vec_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let vec = (0..1024*1024).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();
        let ref_vec = (0..1024*1024).map(|i| F32::usize(i).sin()).collect::<Vec<_>>();

        let out = CudaVector::powf_vec_direct(data.into_iter(),vec.into_iter());
        let known = FloatVector::powf_vec_direct(reference.into_iter(),ref_vec.into_iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn greater_vec_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let vec = (0..1024*1024).map(|i| F32::usize(i).sin()*F32::usize(i)).collect::<Vec<_>>();
        let ref_vec = (0..1024*1024).map(|i| F32::usize(i).sin()*F32::usize(i)).collect::<Vec<_>>();

        let out = CudaVector::greater_vec_direct(data.into_iter(),vec.into_iter());
        let known = FloatVector::greater_vec_direct(reference.into_iter(),ref_vec.into_iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn lesser_vec_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let vec = (0..1024*1024).map(|i| F32::usize(i).sin()*F32::usize(i)).collect::<Vec<_>>();
        let ref_vec = (0..1024*1024).map(|i| F32::usize(i).sin()*F32::usize(i)).collect::<Vec<_>>();

        let out = CudaVector::lesser_vec_direct(data.into_iter(),vec.into_iter());
        let known = FloatVector::lesser_vec_direct(reference.into_iter(),ref_vec.into_iter());
        out.zip(known).for_each(|(o,k)|{
            assert!((o-k).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn sum_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::sum_direct(data.into_iter());
        let known = FloatVector::sum_direct(reference.into_iter());
        assert!((out-known).l2_norm()<F32::EPSILON);
    }
    #[test]
    fn product_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(1..1024*1024+1).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(1..1024*1024+1).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::product_direct(data.into_iter());
        let known = FloatVector::product_direct(reference.into_iter());
        assert!((out-known).l2_norm()<F32::EPSILON);
    }
    #[test]
    fn greater_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::greater_direct(data.into_iter());
        let known = FloatVector::greater_direct(reference.into_iter());
        assert!((out-known).l2_norm()<F32::EPSILON);
    }
    #[test]
    fn lesser_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::lesser_direct(data.into_iter());
        let known = FloatVector::lesser_direct(reference.into_iter());
        assert!((out-known).l2_norm()<F32::EPSILON);
    }
    #[test]
    fn mean_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::mean_direct(data.into_iter());
        let known = FloatVector::mean_direct(reference.into_iter());
        assert!((out-known).l2_norm()<F32::EPSILON);
    }
    #[test]
    fn variance_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let (out_m, out_v) = CudaVector::deviation_direct(data.into_iter());
        let (known_m, known_v) = FloatVector::deviation_direct(reference.into_iter());
        assert!((out_m-known_m).l2_norm()<F32::EPSILON);
        assert!((out_v-known_v).l2_norm()<F32::EPSILON);
    }
    #[test]
    fn deviation_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let (out_m, out_d) = CudaVector::deviation_direct(data.into_iter());
        let (known_m, known_d) = FloatVector::deviation_direct(reference.into_iter());
        assert!((out_m-known_m).l2_norm()<F32::EPSILON);
        assert!((out_d-known_d).l2_norm()<F32::EPSILON);
    }

    #[test]
    fn dot_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let vec = (0..1024*1024).map(|i| F32::usize(i).sin()*F32::usize(i)).collect::<Vec<_>>();
        let ref_vec = (0..1024*1024).map(|i| F32::usize(i).sin()*F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::dot_direct(data.into_iter(),vec.into_iter());
        let known = FloatVector::dot_direct(reference.into_iter(),ref_vec.into_iter());
        assert!((out-known).l2_norm()<F32::EPSILON);
    }
    #[test]
    fn quote_direct(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let reference =(0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let vec = (0..1024*1024).map(|i| F32::usize(i).sin()*F32::usize(i)).collect::<Vec<_>>();
        let ref_vec = (0..1024*1024).map(|i| F32::usize(i).sin()*F32::usize(i)).collect::<Vec<_>>();
        let out = CudaVector::quote_direct(data.into_iter(),vec.into_iter());
        let known = FloatVector::quote_direct(reference.into_iter(),ref_vec.into_iter());
        assert!((out-known).l2_norm()<F32::EPSILON);
    }
 }