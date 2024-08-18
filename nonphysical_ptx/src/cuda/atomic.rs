pub trait Atomic<T>{
     fn atomic_add(&mut self, index: usize, value: T) -> T;

     fn atomic_sub(&mut self, index: usize, value: T) -> T;

     fn atomic_exch(&mut self, index: usize, value: T) -> T;

     fn atomic_max(&mut self, index: usize, value: T) -> T;

     fn atomic_min(&mut self, index: usize, value: T) -> T;

     fn atomic_inc(&mut self, index: usize, value: T) -> T;

     fn atomic_dec(&mut self, index: usize, value: T) -> T;

     fn atomic_cas(&mut self, index: usize,  compare: T, value: T) -> T;

     fn atomic_and(&mut self, index: usize, value: T) -> T;

     fn atomic_or(&mut self, index: usize, value: T) -> T;

     fn atomic_xor(&mut self, index: usize, value: T) -> T;

}

pub trait Reduce<T>{
     fn reduce_add(&mut self, index: usize, value: T);

     fn reduce_max(&mut self, index: usize, value: T);

     fn reduce_min(&mut self, index: usize, value: T);

     fn reduce_inc(&mut self, index: usize, value: T);

     fn reduce_dec(&mut self, index: usize, value: T);

     fn reduce_and(&mut self, index: usize, value: T);

     fn reduce_or(&mut self, index: usize, value: T);

     fn reduce_xor(&mut self, index: usize, value: T);
}



