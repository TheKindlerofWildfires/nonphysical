#[cfg(test)]
mod vector_tests {
    use nonphysical_core::shared::primitive::Primitive;
    use nonphysical_cuda::cuda::runtime::Runtime;
    use nonphysical_ptx::shared::vector::vector_driver::*;
    use nonphysical_std::shared::primitive::F32;
    use nonphysical_core::shared::float::Float;
    use vector_driver::CudaVectorHost;
    use std::sync::Arc;
    #[test]
    fn sum_host() {
        let runtime = Arc::new(Runtime::new(0, "../nonphysical_ptx.ptx"));
        //Allocate a host buffer
        let numbers = (0..1024 * 1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let res = CudaVectorHost::sum(runtime, &numbers);
        assert!(res.0 == 549755780000.0);
    }

    #[test]
    fn l1_min_host() {
        let runtime = Arc::new(Runtime::new(0, "../nonphysical_ptx.ptx"));
        //Allocate a host buffer
        let numbers = (0..5).map(|i| F32::isize(i-2)).collect::<Vec<_>>();
        let res = CudaVectorHost::l1_min(runtime, &numbers);
        dbg!(res);
        //assert!(res.0 == 0.0);
    }

    #[test]
    fn l1_sum_host() {
        let runtime = Arc::new(Runtime::new(0, "../nonphysical_ptx.ptx"));
        //Allocate a host buffer
        let numbers = (0..1024 * 1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let res = CudaVectorHost::l1_sum(runtime, &numbers);
        assert!(res.0 == 549755780000.0);
    }

    #[test]
    fn l2_sum_host() {
        let runtime = Arc::new(Runtime::new(0, "../nonphysical_ptx.ptx"));
        //Allocate a host buffer
        let numbers = (0..1024 * 1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let res = CudaVectorHost::l2_sum(runtime, &numbers);
        dbg!(res);
        assert!(res.0 == 3.843066e17);
    }

    #[test]
    fn add_host() {
        let runtime = Arc::new(Runtime::new(0, "../nonphysical_ptx.ptx"));
        //Allocate a host buffer
        let mut numbers = (0..1024 * 1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        CudaVectorHost::add(runtime, &mut numbers, F32::ONE);
        (0..1024 * 1024).for_each(|i| {
            assert!(i as f32 + 1.0 == numbers[i].0);
        });
    }

    #[test]
    fn sub_host() {
        let runtime = Arc::new(Runtime::new(0, "../nonphysical_ptx.ptx"));
        //Allocate a host buffer
        let mut numbers = (0..1024 * 1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        CudaVectorHost::sub(runtime, &mut numbers, F32::ONE);
        (0..1024 * 1024).for_each(|i| {
            assert!(i as f32 - 1.0 == numbers[i].0);
        });
    }

    #[test]
    fn mul_host() {
        let runtime = Arc::new(Runtime::new(0, "../nonphysical_ptx.ptx"));
        //Allocate a host buffer
        let mut numbers = (0..1024 * 1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        CudaVectorHost::mul(runtime, &mut numbers, F32(2.0));
        (0..1024 * 1024).for_each(|i| {
            assert!(i as f32 * 2.0 == numbers[i].0);
        });
    }

    #[test]
    fn div_host() {
        let runtime = Arc::new(Runtime::new(0, "../nonphysical_ptx.ptx"));
        //Allocate a host buffer
        let mut numbers = (0..1024 * 1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        CudaVectorHost::div(runtime, &mut numbers, F32(2.0));
        (0..1024 * 1024).for_each(|i| {
            assert!(i as f32 / 2.0 == numbers[i].0);
        });
    }
    
    #[test]
    fn add_vec_host() {
        let runtime = Arc::new(Runtime::new(0, "../nonphysical_ptx.ptx"));
        //Allocate a host buffer
        let mut numbers = (0..1024 * 1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let others = (0..1024 * 1024).map(|i| F32::usize(i)+F32::PI).collect::<Vec<_>>();

        CudaVectorHost::add_vec(runtime, &mut numbers, &others);
        (0..1024 * 1024).for_each(|i| {
            assert!((F32::usize(i)+F32::usize(i)+ F32::PI-numbers[i]).l2_norm()<F32::ONE); //Precision on big numbers is hard
        });
    }

    #[test]
    fn sub_vec_host() {
        let runtime = Arc::new(Runtime::new(0, "../nonphysical_ptx.ptx"));
        //Allocate a host buffer
        let mut numbers = (0..1024 * 1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let others = (0..1024 *1024).map(|i| F32::usize(i)+F32::PI).collect::<Vec<_>>();

        CudaVectorHost::sub_vec(runtime, &mut numbers, &others);
        (0..1024 * 1024).for_each(|i| {
            assert!((F32::usize(i)-(F32::usize(i)+ F32::PI)-numbers[i]).l2_norm()<F32::ONE); //Precision on big numbers is hard
        });
    }

    #[test]
    fn mul_vec_host() {
        let runtime = Arc::new(Runtime::new(0, "../nonphysical_ptx.ptx"));
        //Allocate a host buffer
        let mut numbers = (0..1024 * 1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let others = (0..1024 * 1024).map(|i| F32::usize(i)+F32::PI).collect::<Vec<_>>();

        CudaVectorHost::mul_vec(runtime, &mut numbers, &others);
        (0..1024 * 1024).for_each(|i| {
            assert!((F32::usize(i)*(F32::usize(i)+ F32::PI)-numbers[i]).l2_norm()<F32::ONE); //Precision on big numbers is hard
        });
    }


    #[test]
    fn div_vec_host() {
        let runtime = Arc::new(Runtime::new(0, "../nonphysical_ptx.ptx"));
        //Allocate a host buffer
        let mut numbers = (0..1024 * 1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let others = (0..1024 * 1024).map(|i| F32::usize(i)+F32::PI).collect::<Vec<_>>();

        CudaVectorHost::div_vec(runtime, &mut numbers, &others);
        (0..1024 * 1024).for_each(|i| {
            assert!((F32::usize(i)/(F32::usize(i)+ F32::PI)-numbers[i]).l2_norm()<F32::ONE); //Precision on big numbers is hard
        });
    }


}
#[cfg(test)]
mod real_vector_tests {
    use nonphysical_core::shared::{primitive::Primitive, vector::{float_vector::FloatVector, Vector}};
    use nonphysical_cuda::cuda::runtime::Runtime;
    use nonphysical_ptx::shared::vector::vector_driver::*;
    use nonphysical_std::shared::primitive::F32;
    use nonphysical_core::shared::float::Float;
    use real_vector_driver::CudaRealVectorHost;
    use std::sync::Arc;
    #[test]
    fn mean_host() {
        let runtime = Arc::new(Runtime::new(0, "../nonphysical_ptx.ptx"));
        //Allocate a host buffer
        let numbers = (0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let right = FloatVector::mean(numbers.iter());
        let res = CudaRealVectorHost::mean(runtime, &numbers);
        assert!((right-res).l2_norm()<F32::ONE); //floating points are not great
    }

    #[test]
    fn variance_host() {
        let runtime = Arc::new(Runtime::new(0, "../nonphysical_ptx.ptx"));
        //Allocate a host buffer
        let numbers = (0..1024*1).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let (right_mean, right_var) = FloatVector::variance(numbers.iter());
        let (mean,var) = CudaRealVectorHost::variance(runtime.clone(), &numbers);
        assert!((right_mean-mean).l2_norm()<F32::ONE); //floating points are not great
        assert!((right_var-var).l2_norm()<F32::ONE); //floating points are not great


        let numbers = (0..1024*1024).map(|i| F32::usize(i)).collect::<Vec<_>>();
        let (right_mean, right_var) = FloatVector::variance(numbers.iter());
        let (mean,var) = CudaRealVectorHost::variance(runtime, &numbers);
        assert!((right_mean-mean)/(right_mean)<F32::EPSILON); //floating points are not great
        dbg!((right_var-var)/(right_var));
        assert!(((right_var-var)/(right_var)).l2_norm()<F32::EPSILON); //floating points are not great
    }
}