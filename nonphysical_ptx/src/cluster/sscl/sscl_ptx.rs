use crate::shared::primitive::F32;
#[no_mangle]

/***
 * 
 * KERNEL FLAWED: SHARED MEMORY ASSUMPTIONS ARE BAD
 * FIX BY ALLOCATING CONST SHARED MEM AND DOING ALL WORK THERE
 * THIS MEANS DRIVER CAN HAVE CONST CLUSTER COUNT TOO
 * 
 */
pub extern "ptx-kernel" fn sscl_cluster(args: None) {
    //Step 1: Find resp points like a vector 
    let thread_idx = unsafe { _thread_idx_x() } as usize;
    let (start, mut stop) = vector_index();
    if start > args.data.len() {
        return;
    }
    //Step 2: Dedicate shared memory for a_adjust, c_adjust, r_adjust, p_adjust
    //also for NA-NR-NC
    let mut block_acc = CuShared::<F32, 7>::new(); //These can be points FYI, but shared memory for point is not well done esp around atomics (that will need fixing when point vectors are impl)
    
    stop = min(stop, args.data.len());
    let data = &args.data;

    if thread_idx == 0 {
        //actual global copy needed here
        //block_acc.store(0, F32::ZERO);
    }
    args.data[start..stop].zip(marks[start..stop]).for_each(|(x,m)|{
        let (_, win_cluster_idx) = clusters.iter().enumerate().fold(
            (P::Primitive::MAX, 0),
            |acc, (i, cluster)| {
                let distance = x.l2_distance(&cluster.p);
                if distance < acc.0 {
                    (distance, i)
                } else {
                    acc
                }
            },
        );
        *m=win_cluster_idx;
        let wc = clusters[win_cluster_idx];
        let pa = win_cluster.p.l2_distance(&win_cluster.a);
        let px = win_cluster.p.l2_distance(&x);
        let pr = win_cluster.p.l2_distance(&win_cluster.r);

        let theta_a = P::Primitive::usize(if pa >= px { 1 } else { 0 });
        let theta_r = P::Primitive::usize(if px >= pr { 1 } else { 0 });

        let delta = pa / (px + pa);
        let rho = px / (px + pr);
        let beta = pr / (px + pr);
        //do the atomic inc, sync, and pull 
        block_acc.reduce_add(0,delta*theta_a);
        block_acc.reduce_add(1,P::Primitive::ONE);
        block_acc.reduce_add(2,P::Primitive::ONE);

        unsafe{_syncthreads()};

        let mut a_adjust = x - wc.a;
        a_adjust.scale(delta / wc.na * theta_a);
        let mut c_adjust = x - wc.c;
        c_adjust.scale(wc.nc.recip());
        let mut r_adjust = x - wc.r;
        r_adjust.scale(rho / wc.nr * theta_r);
        let mut p_adjust = x - wc.p;
        p_adjust.scale(delta * beta);
        //add all of the adjustres up in shared memory
        block_acc.reduce_add(3,a_adjust);
        block_acc.reduce_add(4,c_adjust);
        block_acc.reduce_add(5,r_adjust);
        block_acc.reduce_add(7,p_adjust);
    });
    if thread_idx==0{

    }

}