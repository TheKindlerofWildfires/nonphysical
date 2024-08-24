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
       write acc the cuda shell
       do the work

*/

pub struct CudaSSCLHost<P: Point> {
    phantom_data: PhantomData<P>,
}
pub struct CudaVectorDevice<P: Point> {
    phantom_data: PhantomData<P>,
}

impl<P: Point> CudaSSCLHost<P> {
    /*
    pub fn map_single_alloc<'a>(host_data: &'a mut [P], other: P)->VectorArgumentsMap<'a, P>{
        let mut global_data = CuGlobalSliceRef::alloc(&host_data);
        let mut global_other = CuGlobalSlice::alloc(&[other]);
        global_data.store(host_data);
        global_other.store(&[other]);
        VectorArgumentsMap {
            data: global_data,
            map: global_other,
        }
    }*/

    //This version is a little careless with device vs host functions, because the clusters need to be reallocated 
    //ideally all memory ops are done in host function and all actual math are done in device functions
    //return type is unclear, standard is Class<> but Cluster<> might make as much sense 
    //Maybe other algs should return similar 
    pub fn cluster(runtime: Arc<Runtime>, host_data: &[P]) {
        let mut acc = [P::ZERO]; //Like MIN/MAX or MEAN/VAR
        let mut marks = vec![0;host_data.len()];
        let mut global_data = CuGlobalSlice::alloc(&host_data);
        let mut global_marks = CuGlobalSliceRef::alloc(&marks);
        let mut global_acc = CuGlobalSliceRef::alloc(&acc);
        global_data.store(host_data);
        let mut rng = PermutedCongruentialGenerator::new(state, state + 1);

        //Step 1: Find the maximum / minimum of the host data
        let mut global_memory = VectorArgumentsReduce{
            data: global_data,
            acc: global_acc,
        };
        global_memory.acc.store(&[P::MIN]);
        CudaPointVectorDevice::max(runtime.clone(), &mut global_memory);
        runtime.sync();
        global_memory.acc.load(&mut acc);
        let max_p = acc[0];

        global_memory.acc.store(&[P::MAX]);
        CudaPointVectorDevice::min(runtime.clone(), &mut global_memory);
        runtime.sync();
        global_memory.acc.load(&mut acc);
        let min_p = acc[0];


        //Step 2: Create a new cluster
        let p_init = P::random_uniform(&min_p, &max_p, &mut rng);
        let cluster_init =
            SelfSelectiveCompetitiveLearning::new_cluster(&mut rng, p_init, (min_p, max_p));
        let clusters = vec![cluster_init];


        //Step 3: Iterate
        loop {
            //must realloc clusters... An amortized solution would be helpful heres
            let mut global_clusters = CuGlobalSliceRef::alloc(&global_clusters);

            let global_cluster_memory = SSCLArguments{
                data: global_data,
                marks: global_marks,
                clusters: global_clusters
            };
            
            CudaSSCLDevice::iterate(runtime.clone(), &mut global_cluster_memory);
            runtime.sync();
            global_cluster_memory.clusters.load(&mut clusters);
            //load the clusters back here
            let (max_pc, max_pc_idx) =
                clusters
                    .iter()
                    .enumerate()
                    .fold((P::Primitive::MIN, 0), |acc, (i, cluster)| {
                        let pc = cluster.p.l1_distance(&cluster.c);
                        if pc > acc.0 {
                            (pc, i)
                        } else {
                            acc
                        }
                    });
            //update EPS
            let global_memory = VectorArgumentsMapReduce{
                data: global_data,
                map: global_marks,
                acc: global_acc,
            };
            CudaPointVectorDevice::marked_variance(runtime.clone(),&global_memory);
            global_memory.acc.load(&mut acc);
            let var = acc[0];
            let mut eps = var
            .data()
            .fold(P::Primitive::ZERO, |acc, v| acc + *v)
            .sqrt();
            //Split if needed
            if max_pc > eps {
                let split_cluster = &clusters[max_pc_idx];
                let new_p = split_cluster.r;
                let new_cluster = SelfSelectiveCompetitiveLearning::new_cluster(&mut rng, new_p, (min_p, max_p));
                clusters.push(new_cluster);
                //reset the clusters
                let max_distance = min_p.l1_distance(&max_p);
                clusters.iter_mut().for_each(|cluster| {
                    cluster.r = cluster.p;
                    while cluster.p.l1_distance(&cluster.a) < max_distance / Primitive::usize(2) {
                        cluster.a = P::random_uniform(&min_p, &max_p, &mut self.rng);
                    }
                    cluster.c = cluster.a;
                    cluster.na = P::Primitive::ZERO;
                    cluster.nc = P::Primitive::ZERO;
                    cluster.nr = P::Primitive::ZERO;
                });



            }else{
                break;
            }
        }
    }
}

impl <P:Point> CudaSSCLDevice<P>{
    pub fn iterate(runtime: Arc<Runtime>, arguments: SSCLArguments){
        let kernel = format!("sscl_iterate_{}", P::type_id());
        Self::vector_launch(runtime, args, args.data.ptr.len(), kernel); //there is no vector launch but I like the idea
    }
}
