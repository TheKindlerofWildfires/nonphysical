use crate::shared::primitive::F32;
#[no_mangle]
pub extern "ptx-kernel" fn sscl_cluster(args: &mut FourierArguments<ComplexScaler<F32>>) {
    //Step are as follows

    /*
        Transfer data to device
        use a min/max kernel to vote on min/max point (each thread handles many points to make the atomics not ugly)
        use a mean/var kernel to get variance (each thread handles many points to ...)
            may need to be mean in groups, var in groups
        surface mean/var to CPU, calculate EPS
        init a cluster CPU side
        Pass cluster in, all data gets worked for the inner loop, each thread handles a few points to minimize atomic locks
            Do the 'easy' computations early, then hit the locks
            Repeat this up to count, not sure if surfacing break criteria here is needed
        Surface the cluster data to CPU, do splitting work
        use mean/var kernel to re-estimate variance/eps
        repeat to convergence by passing clusters in
            clusters should live in sm for locks, actual min/max will be SM for locks, mean/var will live in SM for locks
            it would be nice to put the active data point itself in sm
     */

    /*
        Dev order:
            create min/max kernels and mean/var kernels for point vectors 
            write out the cuda shell
            do the work
    
     */

}