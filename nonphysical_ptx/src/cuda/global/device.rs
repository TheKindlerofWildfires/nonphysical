use core::ops::{Deref, DerefMut};

pub struct CuGlobalSlice<'a, T: Copy> {
    pub ptr: &'a [T],
}

pub struct CuGlobalSliceRef<'a, T: Copy> {
    pub ptr: &'a mut [T],
}

impl<'a, T: Copy> Deref for CuGlobalSlice<'a, T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.ptr
    }
}

impl<'a, T: Copy> Deref for CuGlobalSliceRef<'a, T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.ptr
    }
}

impl<'a, T: Copy> DerefMut for CuGlobalSliceRef<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.ptr
    }
}

impl<'a, T: Copy> CuGlobalSlice<'a, T> {
    pub fn get(&self) -> &[T] {
        self.ptr
    }
}

impl<'a, T: Copy> CuGlobalSliceRef<'a, T> {
    pub fn get(&mut self) -> &mut [T] {
        self.ptr
    }
}

pub struct CuGlobalBox<'a, T: Copy> {
    pub ptr: &'a T,
}

pub struct CuGlobalBoxRef<'a, T: Copy> {
    pub ptr: &'a mut T,
}

impl<'a, T: Copy> Deref for CuGlobalBox<'a, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        self.ptr
    }
}

impl<'a, T: Copy> Deref for CuGlobalBoxRef<'a, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        self.ptr
    }
}

impl<'a, T: Copy> DerefMut for CuGlobalBoxRef<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.ptr
    }
}

impl<'a, T: Copy> CuGlobalBox<'a, T> {
    pub fn get(&self) -> &T {
        self.ptr
    }
}

impl<'a, T: Copy> CuGlobalBoxRef<'a, T> {
    pub fn get(&mut self) -> &mut T {
        self.ptr
    }
}
