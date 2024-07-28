use crate::{linear::svd::{RealSingularValueDecomposition, SingularValueDecomposition}, shared::{float::Float, matrix::Matrix, real::Real, vector::{RealVector, Vector}}};

pub trait PrincipleComponentAnalysis<F: Float> {
    fn pca(data: &mut Matrix<F>, components: usize) -> Matrix<F>;
    fn normalize(data: &mut Matrix<F>);
}

pub struct RealPrincipleComponentAnalysis {}
pub struct ComplexPrincipleComponentAnalysis {}

impl<R: Real<Primitive = R>> PrincipleComponentAnalysis<R> for RealPrincipleComponentAnalysis {
    fn pca(data: &mut Matrix<R>, components: usize) -> Matrix<R> {
        dbg!(&data);
        Self::normalize(data);
        dbg!(&data);

        let (mut u,s,mut v) = RealSingularValueDecomposition::jacobi_svd_full(data);
        let signs = v.data_rows().map(|row|{
            dbg!(&row);
            let (_,_,sign) = row.iter().enumerate().fold((R::ZERO,0,R::ONE), |acc, (i,rp)|{
                if rp.l1_norm()>acc.0{
                    (rp.l1_norm(),i,rp.sign())
                }else{
                    acc
                }
            });
            sign
        }).collect::<Vec<_>>();
        u.data_rows_ref().zip(signs.iter()).for_each(|(row,sign)|{
            <Vec<&'_ R> as Vector<R>>::mul(row.iter_mut(), *sign);
        });
        v.data_rows_ref().zip(signs.iter()).for_each(|(row,sign)|{
            <Vec<&'_ R> as Vector<R>>::mul(row.iter_mut(), *sign);
        });
        dbg!(&u,&s,&v);
        todo!()
    }

    fn normalize(data: &mut Matrix<R>) {
        data.data_rows_ref().for_each(|row| {
            let mean = <Vec<&'_ R> as RealVector<R>>::mean(row.iter());
            row.iter_mut()
                .for_each(|c| *c = *c - mean);
        });
    }
}
//The best version of this probably uses SVD
#[cfg(test)]
mod pca_tests {
    use std::time::SystemTime;
    use super::*;

    #[test]
    fn pca_3x3() {
        let mut m = Matrix::new(3, (0..9).map(|i| i as f32).collect()).transposed();
        let transformed = RealPrincipleComponentAnalysis::pca(&mut m, 2);
        let known_values = vec![
            -5.19615242,
           0.0,
            5.19615242,
        ];
        transformed
            .data()
            .zip(known_values.iter())
            .for_each(|(c, k)| {
                assert!((*c - *k).l2_norm() < f32::EPSILON);
            });
    }
}
