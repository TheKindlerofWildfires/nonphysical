/*
    To use shared memory it's at the start of the file (or function) like
        .shared .align $data_type_align .b8 $name[align]  
    Then it's loaded like
        ld.shared.$data_type $register [name+offset]
    Then it's set like
        st.shared.$data_type $register [name+offset]


    The interesting part here is that I really do think it needs to happen at the start of the function, and I have no way to enforce that 

*/

use core::arch::asm;
use nonphysical_core::shared::float::Float;



#[derive(Clone,Copy)]
pub struct CuShared {
    ptr: u32
}


/* 
pub struct CuShared<'a, T> {
    ptr: &'a mut [T],
}*/

/* 
impl<'a, T> Deref for CuShared<'a, T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.ptr
    }
}

impl<'a, T> DerefMut for CuShared<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.ptr
    }
}*/
//A very hacky constant shared memory impl 
impl CuShared {
    pub fn new<const count: usize, const align: usize>() -> Self{
        let mut ptr;
        unsafe {
            asm!(
                ".shared .align {a} .b8 nonphysical[{b}];",
                "    mov.u32 {pt}, nonphysical;",
                a = const align,
                b = const align*count,
                pt = out(reg32) ptr,
            );
            
        }
        //let ptr = unsafe { core::slice::from_raw_parts_mut(shared_ptr as *mut T, align*count) };
        Self { ptr }
    }
    
    //this function might as well be called 'shoot yourself in the foot'
    pub fn load_f32(&self,index:usize) -> F32{
        let offset = index*4;
        let index = self.ptr+offset as u32;
        let mut out = F32(0.0);
        unsafe {
            asm!(
                "ld.shared.f32 {o}, [{idx}];",
                idx = in(reg32) index,
                o = out(reg32) out.0,
            );
        }
        out
    }

    //this function might as well be called 'shoot yourself in the foot v2'
    pub fn store_f32(&self,index:usize,data:F32){
        let offset = index*4;
        let index = self.ptr+offset as u32;
        unsafe {
            asm!(
                "st.shared.f32 [{idx}], {d};",
                idx = in(reg32) index,
                d = in(reg32) data.0,
            );
        }
    }

    pub unsafe fn atomic_max_f32(&self, index: usize, data: F32) {
        let offset = index*4;
        let index = self.ptr+offset as u32;
        unsafe {
            asm!(
                "red.shared.max.s32 [{idx}], {d};",
                idx = in(reg32) index,
                d = in(reg32) data.0
            ); 
        }
    }

    pub unsafe fn atomic_min_f32(&self, index: usize, data: F32) {
        let offset = index*4;
        let index = self.ptr+offset as u32;
        unsafe {
            asm!(
                "red.shared.min.s32 [{idx}], {d};",
                idx = in(reg32) index,
                d = in(reg32) data.0
            ); 
        }
    }

}

use crate::{cuda::cu_slice::CuSliceRef, shared::primitive::F32};

/* 
#[no_mangle]
pub extern "ptx-kernel" fn test(x: &mut CuSliceRef<f32>){
    use core::arch::nvptx::_thread_idx_x;

    /*
    let mut shared = CuShared::new::<8192,4>();
    let idx = unsafe { _thread_idx_x() } as usize;
    let shared_ptr = unsafe { ptr_gen_to_shared_p0i8_p0i8(shared.ptr as *mut i8) };
    unsafe { *shared_ptr = 0 };*/
    let mut new_shared: CuShared<F32> = CuShared::new::<8192,4>();
    x[0] = 2.0;
    new_shared[0]=F32(5.0);
    x[1] = new_shared[0].0;
    x[2] = 3.0;
    //new_shared[0]=5.0;
    /* 
    for i in 1..9{
        shared.store_f32(i*idx, F32(x[0]));
    }

    for i in 1..9{
        x[0]=shared.load_f32(i*idx).0+shared.load_f32(idx*2).0;
    }*/
    

}

extern {
    #[link_name = "llvm.fma.f32"]
    pub fn fma_f32(a: f32, b: f32, c: f32) -> f32;

    #[link_name = "llvm.nvvm.ptr.gen.to.shared.p0i8.p0i8"]
    pub fn ptr_gen_to_shared_p0i8_p0i8(a: *mut i8) -> *mut i8;

    #[link_name = "llvm.nvvm.ptr.shared.to.gen.p0i8.p0i8"]
    pub fn ptr_shared_to_gen_p0i8_p0i8(a: *mut i8) -> *mut i8;

    #[link_name = "llvm.nvvm.isspacep.shared"]
    pub fn isspacep_shared(a: *mut i8) -> bool;
}*/