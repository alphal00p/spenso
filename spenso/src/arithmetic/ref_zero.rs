use crate::{contraction::RefZero, data::DenseTensor, structure::TensorStructure};

impl<T, S> RefZero for DenseTensor<T, S>
where
    T: RefZero + Clone,
    S: TensorStructure + Clone,
{
    fn ref_zero(&self) -> Self {
        let zero = self.data[0].ref_zero();
        DenseTensor {
            structure: self.structure.clone(),
            data: vec![zero; self.data.len()],
        }
    }
}
