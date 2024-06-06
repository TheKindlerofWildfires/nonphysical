pub struct SingularValueDecomposition{
    columns: usize,
}

impl SingularValueDecomposition{
    pub fn new(columns: usize){
        
    }
    pub fn svd(&self, matrix: &[Complex64]){
        let rows = matrix.len()/self.columns;
        let dim = self.columns.min(rows);

    }
}