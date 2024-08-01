use alloc::vec::Vec;

use crate::{
    linear::svd::{RealSingularValueDecomposition, SingularValueDecomposition},
    shared::{
        float::Float,
        matrix::Matrix,
        real::Real,
        vector::{RealVector, Vector},
    },
};

pub trait PrincipleComponentAnalysis<F: Float> {
    fn pca(data: &mut Matrix<F>, components: usize) -> Matrix<F>;
    fn normalize(data: &mut Matrix<F>);
}

pub struct RealPrincipleComponentAnalysis {}
pub struct ComplexPrincipleComponentAnalysis {}

impl<R: Real<Primitive = R>> PrincipleComponentAnalysis<R> for RealPrincipleComponentAnalysis {
    fn pca(data: &mut Matrix<R>, components: usize) -> Matrix<R> {
        Self::normalize(data);

        let (mut u, s, v) = RealSingularValueDecomposition::jacobi_svd_full(data);

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
            <Vec<&'_ R> as Vector<R>>::mul(row.iter_mut(), sign);
        });

        /*
        v.data_rows_ref().zip(signs.into_iter()).for_each(|(row,sign)|{
            <Vec<&'_ R> as Vector<R>>::mul(row.iter_mut(), *sign);
        });*/

        let mut ret = Matrix::<R>::new(
            components,
            u.data()
                .take(u.rows * components)
                .copied()
                .collect::<Vec<_>>(),
        );
        ret.data_rows_ref().zip(s).for_each(|(row, sp)| {
            <Vec<&'_ R> as Vector<R>>::mul(row.iter_mut(), sp);
        });
        ret
    }

    fn normalize(data: &mut Matrix<R>) {
        data.data_rows_ref().for_each(|row| {
            let mean = <Vec<&'_ R> as RealVector<R>>::mean_ref(row.iter_mut());
            row.iter_mut().for_each(|c| *c -= mean);
        });
    }
}