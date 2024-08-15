use alloc::vec::Vec;

use crate::{
    linear::svd::{RealSingularValueDecomposition, SingularValueDecomposition},
    shared::{
        float::Float,
        matrix::{heap::MatrixHeap, Matrix},
        real::Real,
        vector::{RealVector, Vector},
    },
};

pub trait PrincipleComponentAnalysis<F: Float> {
    type Matrix: Matrix<F>;
    fn pca(data: &mut Self::Matrix, components: usize) -> Self::Matrix;
    fn normalize(data: &mut Self::Matrix);
}

pub struct RealPrincipleComponentAnalysis {}
pub struct ComplexPrincipleComponentAnalysis {}

impl<R: Real<Primitive = R>> PrincipleComponentAnalysis<R> for RealPrincipleComponentAnalysis {
    type Matrix = MatrixHeap<R>;
    fn pca(data: &mut Self::Matrix, components: usize) -> Self::Matrix {
        Self::normalize(data);
        dbg!(&data);

        let (mut u, s, v) = RealSingularValueDecomposition::jacobi_svd_full(data);
        dbg!(&u,&s,&v);

        let signs = v
            .data_rows()
            .map(|row| {
                let (_, sign) = row
                    .iter()
                    .fold((R::ZERO, R::ONE), |acc,  rp| {
                        if rp.l1_norm() > acc.0 {
                            (rp.l1_norm(), rp.sign())
                        } else {
                            acc
                        }
                    });
                sign
            })
            .collect::<Vec<_>>();
        u.data_rows_ref().zip(signs).for_each(|(row, sign)| {
            Vector::mul(row.iter_mut(), sign);
        });

        /*
        v.data_rows_ref().zip(signs.into_iter()).for_each(|(row,sign)|{
            <Vec<&'_ R> as Vector<R>>::mul(row.iter_mut(), *sign);
        });*/

        let mut ret = Self::Matrix::new((
            components,
            u.data()
                .take(u.rows * components)
                .copied()
                .collect::<Vec<_>>()),
        );
        ret.data_rows_ref().zip(s).for_each(|(row, sp)| {
            Vector::mul(row.iter_mut(), sp);
        });
        ret
    }

    fn normalize(data: &mut Self::Matrix) {
        
        (0..data.cols).for_each(|i|{
            let mean = RealVector::mean(data.data_col(i));
            Vector::sub(data.data_col_ref(i), mean);
        });
        /* 
        data.data_rows_ref().for_each(|row| {
            let mean = RealVector::mean(row.iter());
            Vector::sub(row.iter_mut(), mean);
        });*/
    }
}