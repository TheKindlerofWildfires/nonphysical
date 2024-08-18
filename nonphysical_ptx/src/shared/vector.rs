pub mod vector_ptx;
pub mod vector_driver;

const CYCLE_COMPARE: usize = 1; //this is saying 'one op per thread' which is bad with atomics


pub struct VectorArgumentsReduce<'a, F: Float> {
    pub data: CuGlobalSlice<'a, F>,
    pub acc: CuGlobalSliceRef<'a, F>,
}

pub struct VectorArgumentsMap<'a, F: Float> {
    pub data: CuGlobalSliceRef<'a, F>,
    pub map: CuGlobalSlice<'a, F>,
}

pub struct VectorArgumentsMapReduce<'a, F: Float> {
    pub data: CuGlobalSliceRef<'a, F>,
    pub map: CuGlobalSlice<'a, F>,
    pub acc: CuGlobalSliceRef<'a, F>,
}