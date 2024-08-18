use crate::shared::primitive::F32;
use nonphysical_core::shared::float::Float;
use core::arch::asm;
use crate::cuda::atomic::{Atomic,Reduce};
use crate::cuda::global::device::CuGlobalSliceRef;

impl<'a> Atomic<F32> for CuGlobalSliceRef<'a,F32>{
    fn atomic_add(&mut self, index: usize, value: F32) -> F32 {
       assert!(index<self.ptr.len());
       let index = self.ptr.as_ptr() as u32 +(index*size_of::<F32>()) as u32;
       let mut out = F32::ZERO;
       unsafe {
           asm!(
               "atom.global.add.f32 {o},[{idx}], {v};",
               idx = in(reg32) index,
               v = in(reg32) value.0,
               o = out(reg32) out.0,
           ); 
       }
       out
   }

    fn atomic_sub(&mut self, index: usize, value: F32) -> F32 {
       assert!(index<self.ptr.len());
       let index = self.ptr.as_ptr() as u32 +(index*size_of::<F32>()) as u32;
       let mut out = F32::ZERO;
       unsafe {
           asm!(
               "atom.global.sub.f32 {o},[{idx}], {v};",
               idx = in(reg32) index,
               v = in(reg32) value.0,
               o = out(reg32) out.0,
           ); 
       }
       out
   }

    fn atomic_exch(&mut self, index: usize, value: F32) -> F32 {
       assert!(index<self.ptr.len());
       let index = self.ptr.as_ptr() as u32 +(index*size_of::<F32>()) as u32;
       let mut out = F32::ZERO;
       unsafe {
           asm!(
               "atom.global.exch.f32 {o},[{idx}], {v};",
               idx = in(reg32) index,
               v = in(reg32) value.0,
               o = out(reg32) out.0,
           ); 
       }
       out
   }

    fn atomic_max(&mut self, index: usize, value: F32) -> F32 {
       assert!(index<self.ptr.len());
       let index = self.ptr.as_ptr() as u32 +(index*size_of::<F32>()) as u32;
       let mut out = F32::ZERO;
       unsafe {
           asm!(
               "atom.global.max.f32 {o},[{idx}], {v};",
               idx = in(reg32) index,
               v = in(reg32) value.0,
               o = out(reg32) out.0,
           ); 
       }
       out
   }

    fn atomic_min(&mut self, index: usize, value: F32) -> F32 {
       assert!(index<self.ptr.len());
       let index = self.ptr.as_ptr() as u32 +(index*size_of::<F32>()) as u32;
       let mut out = F32::ZERO;
       unsafe {
           asm!(
               "atom.global.min.f32 {o},[{idx}], {v};",
               idx = in(reg32) index,
               v = in(reg32) value.0,
               o = out(reg32) out.0,
           ); 
       }
       out
   }

    fn atomic_inc(&mut self, index: usize, value: F32) -> F32 {
       assert!(index<self.ptr.len());
       let index = self.ptr.as_ptr() as u32 +(index*size_of::<F32>()) as u32;
       let mut out = F32::ZERO;
       unsafe {
           asm!(
               "atom.global.inc.f32 {o},[{idx}], {v};",
               idx = in(reg32) index,
               v = in(reg32) value.0,
               o = out(reg32) out.0,
           ); 
       }
       out
   }

    fn atomic_dec(&mut self, index: usize, value: F32) -> F32 {
       assert!(index<self.ptr.len());
       let index = self.ptr.as_ptr() as u32 +(index*size_of::<F32>()) as u32;
       let mut out = F32::ZERO;
       unsafe {
           asm!(
               "atom.global.dec.f32 {o},[{idx}], {v};",
               idx = in(reg32) index,
               v = in(reg32) value.0,
               o = out(reg32) out.0,
           ); 
       }
       out
   }

    fn atomic_cas(&mut self, index: usize, compare: F32, value: F32) -> F32 {
       assert!(index<self.ptr.len());
       let index = self.ptr.as_ptr() as u32 +(index*size_of::<F32>()) as u32;
       let mut out = F32::ZERO;
       unsafe {
           asm!(
               "atom.global.cas.f32 {o},[{idx}], {v}, {c};",
               idx = in(reg32) index,
               v = in(reg32) value.0,
               c = in(reg32) compare.0,
               o = out(reg32) out.0,
           ); 
       }
       out
   }

    fn atomic_and(&mut self, index: usize, value: F32) -> F32 {
       //this operation seems poorly defined for floats
       assert!(index<self.ptr.len());
       let index = self.ptr.as_ptr() as u32 +(index*size_of::<F32>()) as u32;
       let mut out = F32::ZERO;
       unsafe {
           asm!(
               "atom.global.and.f32 {o},[{idx}], {v};",
               idx = in(reg32) index,
               v = in(reg32) value.0,
               o = out(reg32) out.0,
           ); 
       }
       out
   }

    fn atomic_or(&mut self, index: usize, value: F32) -> F32 {
       //this operation seems poorly defined for floats
       assert!(index<self.ptr.len());
       let index = self.ptr.as_ptr() as u32 +(index*size_of::<F32>()) as u32;
       let mut out = F32::ZERO;
       unsafe {
           asm!(
               "atom.global.or.f32 {o},[{idx}], {v};",
               idx = in(reg32) index,
               v = in(reg32) value.0,
               o = out(reg32) out.0,
           ); 
       }
       out
   }

    fn atomic_xor(&mut self, index: usize, value: F32) -> F32 {
       //this operation seems poorly defined for floats
       assert!(index<self.ptr.len());
       let index = self.ptr.as_ptr() as u32 +(index*size_of::<F32>()) as u32;
       let mut out = F32::ZERO;
       unsafe {
           asm!(
               "atom.global.xor.f32 {o},[{idx}], {v};",
               idx = in(reg32) index,
               v = in(reg32) value.0,
               o = out(reg32) out.0,
           ); 
       }
       out
   }
}

impl<'a>  Reduce<F32> for CuGlobalSliceRef<'a,F32>{
   fn reduce_add(&mut self, index: usize, value: F32) {
       assert!(index<self.ptr.len());
       let index = self.ptr.as_ptr() as u32 +(index*size_of::<F32>()) as u32;
       unsafe {
           asm!(
               "red.global.add.f32 [{idx}], {v};",
               idx = in(reg32) index,
               v = in(reg32) value.0,
           ); 
       }
   }

   fn reduce_max(&mut self, index: usize, value: F32) {
       assert!(index<self.ptr.len());
       let index = self.ptr.as_ptr() as u32 +(index*size_of::<F32>()) as u32;
       unsafe {
           asm!(
               "red.global.max.f32 [{idx}], {v};",
               idx = in(reg32) index,
               v = in(reg32) value.0,
           ); 
       }
   }

   fn reduce_min(&mut self, index: usize, value: F32) {
       assert!(index<self.ptr.len());
       let index = self.ptr.as_ptr() as u32 +(index*size_of::<F32>()) as u32;
       unsafe {
           asm!(
               "red.global.min.f32 [{idx}], {v};",
               idx = in(reg32) index,
               v = in(reg32) value.0,
           ); 
       }
   }

   fn reduce_inc(&mut self, index: usize, value: F32) {
       assert!(index<self.ptr.len());
       let index = self.ptr.as_ptr() as u32 +(index*size_of::<F32>()) as u32;
       unsafe {
           asm!(
               "red.global.inc.f32 [{idx}], {v};",
               idx = in(reg32) index,
               v = in(reg32) value.0,
           ); 
       }
   }

   fn reduce_dec(&mut self, index: usize, value: F32) {
       assert!(index<self.ptr.len());
       let index = self.ptr.as_ptr() as u32 +(index*size_of::<F32>()) as u32;
       unsafe {
           asm!(
               "red.global.dec.f32 [{idx}], {v};",
               idx = in(reg32) index,
               v = in(reg32) value.0,
           ); 
       }
   }

   fn reduce_and(&mut self, index: usize, value: F32) {
       //this operation seems poorly defined for floats
       assert!(index<self.ptr.len());
       let index = self.ptr.as_ptr() as u32 +(index*size_of::<F32>()) as u32;
       unsafe {
           asm!(
               "red.global.and.f32 [{idx}], {v};",
               idx = in(reg32) index,
               v = in(reg32) value.0,
           ); 
       }
   }

   fn reduce_or(&mut self, index: usize, value: F32) {
       //this operation seems poorly defined for floats
       assert!(index<self.ptr.len());
       let index = self.ptr.as_ptr() as u32 +(index*size_of::<F32>()) as u32;
       unsafe {
           asm!(
               "red.global.or.f32 [{idx}], {v};",
               idx = in(reg32) index,
               v = in(reg32) value.0,
           ); 
       }
   }

   fn reduce_xor(&mut self, index: usize, value: F32) {
       //this operation seems poorly defined for floats
       assert!(index<self.ptr.len());
       let index = self.ptr.as_ptr() as u32 +(index*size_of::<F32>()) as u32;
       unsafe {
           asm!(
               "red.global.xor.f32 [{idx}], {v};",
               idx = in(reg32) index,
               v = in(reg32) value.0,
           ); 
       }
   }
}



impl<'a>  Atomic<u32> for CuGlobalSliceRef<'a,u32>{
   fn atomic_add(&mut self, index: usize, value: u32) -> u32 {
      assert!(index<self.ptr.len());
      let index = self.ptr.as_ptr() as u32 +(index*size_of::<u32>()) as u32;
      let mut out: u32;
      unsafe {
          asm!(
              "atom.global.add.u32 {o},[{idx}], {v};",
              idx = in(reg32) index,
              v = in(reg32) value,
              o = out(reg32) out,
          ); 
      }
      out
  }

   fn atomic_sub(&mut self, index: usize, value: u32) -> u32 {
      assert!(index<self.ptr.len());
      let index = self.ptr.as_ptr() as u32 +(index*size_of::<u32>()) as u32;
      let mut out: u32;
      unsafe {
          asm!(
              "atom.global.sub.u32 {o},[{idx}], {v};",
              idx = in(reg32) index,
              v = in(reg32) value,
              o = out(reg32) out,
          ); 
      }
      out
  }

   fn atomic_exch(&mut self, index: usize, value: u32) -> u32 {
      assert!(index<self.ptr.len());
      let index = self.ptr.as_ptr() as u32 +(index*size_of::<u32>()) as u32;
      let mut out: u32;
      unsafe {
          asm!(
              "atom.global.exch.u32 {o},[{idx}], {v};",
              idx = in(reg32) index,
              v = in(reg32) value,
              o = out(reg32) out,
          ); 
      }
      out
  }

   fn atomic_max(&mut self, index: usize, value: u32) -> u32 {
      assert!(index<self.ptr.len());
      let index = self.ptr.as_ptr() as u32 +(index*size_of::<u32>()) as u32;
      let mut out: u32;
      unsafe {
          asm!(
              "atom.global.max.u32 {o},[{idx}], {v};",
              idx = in(reg32) index,
              v = in(reg32) value,
              o = out(reg32) out,
          ); 
      }
      out
  }

   fn atomic_min(&mut self, index: usize, value: u32) -> u32 {
      assert!(index<self.ptr.len());
      let index = self.ptr.as_ptr() as u32 +(index*size_of::<u32>()) as u32;
      let mut out: u32;
      unsafe {
          asm!(
              "atom.global.min.u32 {o},[{idx}], {v};",
              idx = in(reg32) index,
              v = in(reg32) value,
              o = out(reg32) out,
          ); 
      }
      out
  }

   fn atomic_inc(&mut self, index: usize, value: u32) -> u32 {
      assert!(index<self.ptr.len());
      let index = self.ptr.as_ptr() as u32 +(index*size_of::<u32>()) as u32;
      let mut out: u32;
      unsafe {
          asm!(
              "atom.global.inc.u32 {o},[{idx}], {v};",
              idx = in(reg32) index,
              v = in(reg32) value,
              o = out(reg32) out,
          ); 
      }
      out
  }

   fn atomic_dec(&mut self, index: usize, value: u32) -> u32 {
      assert!(index<self.ptr.len());
      let index = self.ptr.as_ptr() as u32 +(index*size_of::<u32>()) as u32;
      let mut out: u32;
      unsafe {
          asm!(
              "atom.global.dec.u32 {o},[{idx}], {v};",
              idx = in(reg32) index,
              v = in(reg32) value,
              o = out(reg32) out,
          ); 
      }
      out
  }

   fn atomic_cas(&mut self, index: usize, compare: u32, value: u32) -> u32 {
      assert!(index<self.ptr.len());
      let index = self.ptr.as_ptr() as u32 +(index*size_of::<u32>()) as u32;
      let mut out: u32;
      unsafe {
          asm!(
              "atom.global.cas.u32 {o},[{idx}], {v}, {c};",
              idx = in(reg32) index,
              v = in(reg32) value,
              c = in(reg32) compare,
              o = out(reg32) out,
          ); 
      }
      out
  }

   fn atomic_and(&mut self, index: usize, value: u32) -> u32 {
      //this operation seems poorly defined for floats
      assert!(index<self.ptr.len());
      let index = self.ptr.as_ptr() as u32 +(index*size_of::<u32>()) as u32;
      let mut out: u32;
      unsafe {
          asm!(
              "atom.global.and.u32 {o},[{idx}], {v};",
              idx = in(reg32) index,
              v = in(reg32) value,
              o = out(reg32) out,
          ); 
      }
      out
  }

   fn atomic_or(&mut self, index: usize, value: u32) -> u32 {
      //this operation seems poorly defined for floats
      assert!(index<self.ptr.len());
      let index = self.ptr.as_ptr() as u32 +(index*size_of::<u32>()) as u32;
      let mut out: u32;
      unsafe {
          asm!(
              "atom.global.or.u32 {o},[{idx}], {v};",
              idx = in(reg32) index,
              v = in(reg32) value,
              o = out(reg32) out,
          ); 
      }
      out
  }

   fn atomic_xor(&mut self, index: usize, value: u32) -> u32 {
      //this operation seems poorly defined for floats
      assert!(index<self.ptr.len());
      let index = self.ptr.as_ptr() as u32 +(index*size_of::<u32>()) as u32;
      let mut out: u32;
      unsafe {
          asm!(
              "atom.global.xor.u32 {o},[{idx}], {v};",
              idx = in(reg32) index,
              v = in(reg32) value,
              o = out(reg32) out,
          ); 
      }
      out
  }
}

impl<'a> Reduce<u32> for CuGlobalSliceRef<'a, u32>{
  fn reduce_add(&mut self, index: usize, value: u32) {
      assert!(index<self.ptr.len());
      let index = self.ptr.as_ptr() as u32 +(index*size_of::<u32>()) as u32;
      unsafe {
          asm!(
              "red.global.add.u32 [{idx}], {v};",
              idx = in(reg32) index,
              v = in(reg32) value,
          ); 
      }
  }

  fn reduce_max(&mut self, index: usize, value: u32) {
      assert!(index<self.ptr.len());
      let index = self.ptr.as_ptr() as u32 +(index*size_of::<u32>()) as u32;
      unsafe {
          asm!(
              "red.global.max.u32 [{idx}], {v};",
              idx = in(reg32) index,
              v = in(reg32) value,
          ); 
      }
  }

  fn reduce_min(&mut self, index: usize, value: u32) {
      assert!(index<self.ptr.len());
      let index = self.ptr.as_ptr() as u32 +(index*size_of::<u32>()) as u32;
      unsafe {
          asm!(
              "red.global.min.u32 [{idx}], {v};",
              idx = in(reg32) index,
              v = in(reg32) value,
          ); 
      }
  }

  fn reduce_inc(&mut self, index: usize, value: u32) {
      assert!(index<self.ptr.len());
      let index = self.ptr.as_ptr() as u32 +(index*size_of::<u32>()) as u32;
      unsafe {
          asm!(
              "red.global.inc.u32 [{idx}], {v};",
              idx = in(reg32) index,
              v = in(reg32) value,
          ); 
      }
  }

  fn reduce_dec(&mut self, index: usize, value: u32) {
      assert!(index<self.ptr.len());
      let index = self.ptr.as_ptr() as u32 +(index*size_of::<u32>()) as u32;
      unsafe {
          asm!(
              "red.global.dec.u32 [{idx}], {v};",
              idx = in(reg32) index,
              v = in(reg32) value,
          ); 
      }
  }

  fn reduce_and(&mut self, index: usize, value: u32) {
      //this operation seems poorly defined for floats
      assert!(index<self.ptr.len());
      let index = self.ptr.as_ptr() as u32 +(index*size_of::<u32>()) as u32;
      unsafe {
          asm!(
              "red.global.and.u32 [{idx}], {v};",
              idx = in(reg32) index,
              v = in(reg32) value,
          ); 
      }
  }

  fn reduce_or(&mut self, index: usize, value: u32) {
      //this operation seems poorly defined for floats
      assert!(index<self.ptr.len());
      let index = self.ptr.as_ptr() as u32 +(index*size_of::<u32>()) as u32;
      unsafe {
          asm!(
              "red.global.or.u32 [{idx}], {v};",
              idx = in(reg32) index,
              v = in(reg32) value,
          ); 
      }
  }

  fn reduce_xor(&mut self, index: usize, value: u32) {
      //this operation seems poorly defined for floats
      assert!(index<self.ptr.len());
      let index = self.ptr.as_ptr() as u32 +(index*size_of::<u32>()) as u32;
      unsafe {
          asm!(
              "red.global.xor.u32 [{idx}], {v};",
              idx = in(reg32) index,
              v = in(reg32) value,
          ); 
      }
  }
}