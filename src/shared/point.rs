use super::float::Float;

#[derive(Clone,Debug)]
pub struct Point<T:Float, const N: usize>{
    pub data: [T;N]
}

impl<T:Float, const N: usize> Point<T,N>{
    pub fn new(data: [T;N])-> Self{
        Self{data}
    }
    pub fn from_vec(vec: Vec<T>) -> Self{
        debug_assert!(vec.len() == N);
        let mut data = [T::ZERO;N];
        data.iter_mut().zip(vec.iter()).for_each(|(d,v)| *d = * v);
        Self { data }
    }

    pub const ZERO: Self = Self { data: [T::ZERO;N] };
    pub const MAX: Self = Self { data: [T::MAX;N] };
    pub const MIN: Self = Self { data: [T::MIN;N] };

    pub fn square_distance(&self, other: &Self) -> T{
        self.data.iter().zip(other.data.iter()).fold(T::ZERO,|acc,(p1,p2)|{
            acc+ (*p1-*p2).square_norm()
        })
    }

    pub fn distance(&self, other: &Self)-> T{
        self.data.iter().zip(other.data.iter()).fold(T::ZERO,|acc,(p1,p2)|{
            acc+ (*p1-*p2).norm()
        })
    }
}