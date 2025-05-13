use crate::{
    algebra::algebraic_traits::{IsZero, RefZero},
    algebra::upgrading_arithmetic::{FallibleAddAssign, FallibleSubAssign, TrySmallestUpgrade},
    iterators::IteratableTensor,
    structure::{StructureContract, TensorStructure},
    tensors::data::{DataIterator, DenseTensor, SetTensorData, SparseTensor},
};

use std::iter::Iterator;

use super::{ContractableWith, ContractionError, ExteriorProduct, ExteriorProductInterleaved};

impl<T, U, I> ExteriorProduct<SparseTensor<T, I>> for SparseTensor<U, I>
where
    U: ContractableWith<
        T,
        Out: FallibleAddAssign<U::Out> + FallibleSubAssign<U::Out> + Clone + RefZero + IsZero,
    >,
    I: TensorStructure + Clone + StructureContract,
{
    type LCM = SparseTensor<U::Out, I>;

    fn exterior_product(
        &self,
        other: &SparseTensor<T, I>,
        final_structure: I,
    ) -> Result<Self::LCM, ContractionError> {
        let mut out = SparseTensor::empty(final_structure);
        let stride = other.size()?;

        for (i, u) in self.flat_iter() {
            for (j, t) in other.flat_iter() {
                let _ = out.set_flat(i * stride + j, u.mul_fallible(t).unwrap());
            }
        }

        Ok(out)
    }
}

impl<T, U, I> ExteriorProduct<DenseTensor<T, I>> for DenseTensor<U, I>
where
    U: ContractableWith<
        T,
        Out: FallibleAddAssign<U::Out> + FallibleSubAssign<U::Out> + Clone + RefZero + IsZero,
    >,
    I: TensorStructure + Clone + StructureContract,
{
    type LCM = DenseTensor<U::Out, I>;

    fn exterior_product(
        &self,
        other: &DenseTensor<T, I>,
        final_structure: I,
    ) -> Result<Self::LCM, ContractionError> {
        let zero = self.data[0].try_upgrade().unwrap().into_owned().ref_zero();
        let mut out = DenseTensor {
            data: vec![zero.clone(); final_structure.size()?],
            structure: final_structure,
        };

        let stride = other.size()?;

        for (i, u) in self.flat_iter() {
            for (j, t) in other.flat_iter() {
                let _ = out.set_flat(i * stride + j, u.mul_fallible(t).unwrap());
            }
        }

        Ok(out)
    }
}

impl<T, U, I> ExteriorProduct<SparseTensor<T, I>> for DenseTensor<U, I>
where
    U: ContractableWith<
        T,
        Out: FallibleAddAssign<U::Out> + FallibleSubAssign<U::Out> + Clone + RefZero + IsZero,
    >,
    I: TensorStructure + Clone + StructureContract,
{
    type LCM = DenseTensor<U::Out, I>;

    fn exterior_product(
        &self,
        other: &SparseTensor<T, I>,
        final_structure: I,
    ) -> Result<Self::LCM, ContractionError> {
        let zero = self.data[0].try_upgrade().unwrap().into_owned().ref_zero();
        let mut out = DenseTensor {
            data: vec![zero.clone(); final_structure.size()?],
            structure: final_structure,
        };
        let stride = other.size()?;

        for (i, u) in self.flat_iter() {
            for (j, t) in other.flat_iter() {
                let _ = out.set_flat(i * stride + j, u.mul_fallible(t).unwrap());
            }
        }

        Ok(out)
    }
}

impl<T, U, I> ExteriorProduct<DenseTensor<T, I>> for SparseTensor<U, I>
where
    U: ContractableWith<
        T,
        Out: FallibleAddAssign<U::Out> + FallibleSubAssign<U::Out> + Clone + RefZero + IsZero,
    >,
    T: TrySmallestUpgrade<U, LCM = U::Out>,
    I: TensorStructure + Clone + StructureContract,
{
    type LCM = DenseTensor<U::Out, I>;

    fn exterior_product(
        &self,
        other: &DenseTensor<T, I>,
        final_structure: I,
    ) -> Result<Self::LCM, ContractionError> {
        let zero = if let Some((_, s)) = self.flat_iter().next() {
            s.try_upgrade().unwrap().as_ref().ref_zero()
        } else if let Some((_, o)) = other.iter_flat().next() {
            o.try_upgrade().unwrap().as_ref().ref_zero()
        } else {
            return Err(ContractionError::EmptySparse);
        };
        let mut out = DenseTensor {
            data: vec![zero.clone(); final_structure.size()?],
            structure: final_structure,
        };

        let stride = other.size()?;

        for (i, u) in self.flat_iter() {
            for (j, t) in other.flat_iter() {
                let _ = out.set_flat(i * stride + j, u.mul_fallible(t).unwrap());
            }
        }

        Ok(out)
    }
}

impl<T, U, I> ExteriorProductInterleaved<SparseTensor<T, I>> for SparseTensor<U, I>
where
    U: ContractableWith<
        T,
        Out: FallibleAddAssign<U::Out> + FallibleSubAssign<U::Out> + Clone + RefZero + IsZero,
    >,
    I: TensorStructure + Clone + StructureContract,
{
    type LCM = SparseTensor<U::Out, I>;

    fn exterior_product_interleaved(
        &self,
        other: &SparseTensor<T, I>,
        result_structure: <Self::LCM as crate::structure::HasStructure>::Structure,
        partition: bitvec::prelude::BitVec,
    ) -> Result<Self::LCM, ContractionError> {
        todo!()
    }
}

impl<T, U, I> ExteriorProductInterleaved<DenseTensor<T, I>> for DenseTensor<U, I>
where
    U: ContractableWith<
        T,
        Out: FallibleAddAssign<U::Out> + FallibleSubAssign<U::Out> + Clone + RefZero + IsZero,
    >,
    I: TensorStructure + Clone + StructureContract,
{
    type LCM = DenseTensor<U::Out, I>;

    fn exterior_product_interleaved(
        &self,
        other: &DenseTensor<T, I>,
        result_structure: <Self::LCM as crate::structure::HasStructure>::Structure,
        partition: bitvec::prelude::BitVec,
    ) -> Result<Self::LCM, ContractionError> {
        todo!()
    }
}

impl<T, U, I> ExteriorProductInterleaved<SparseTensor<T, I>> for DenseTensor<U, I>
where
    U: ContractableWith<
        T,
        Out: FallibleAddAssign<U::Out> + FallibleSubAssign<U::Out> + Clone + RefZero + IsZero,
    >,
    I: TensorStructure + Clone + StructureContract,
{
    type LCM = DenseTensor<U::Out, I>;

    fn exterior_product_interleaved(
        &self,
        other: &SparseTensor<T, I>,
        result_structure: <Self::LCM as crate::structure::HasStructure>::Structure,
        partition: bitvec::prelude::BitVec,
    ) -> Result<Self::LCM, ContractionError> {
        todo!()
    }
}

impl<T, U, I> ExteriorProductInterleaved<DenseTensor<T, I>> for SparseTensor<U, I>
where
    U: ContractableWith<
        T,
        Out: FallibleAddAssign<U::Out> + FallibleSubAssign<U::Out> + Clone + RefZero + IsZero,
    >,
    T: TrySmallestUpgrade<U, LCM = U::Out>,
    I: TensorStructure + Clone + StructureContract,
{
    type LCM = DenseTensor<U::Out, I>;

    fn exterior_product_interleaved(
        &self,
        other: &DenseTensor<T, I>,
        result_structure: <Self::LCM as crate::structure::HasStructure>::Structure,
        partition: bitvec::prelude::BitVec,
    ) -> Result<Self::LCM, ContractionError> {
        todo!()
    }
}
