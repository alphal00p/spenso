use log::trace;
use std::collections::HashMap;
// use num::Zero;

use crate::{
    algebra::{
        algebraic_traits::{IsZero, RefZero},
        upgrading_arithmetic::{FallibleAddAssign, FallibleSubAssign, TrySmallestUpgrade},
    },
    iterators::{
        CoreFlatFiberIterator, Fiber, FiberData, IteratableTensor, IteratesAlongFibers,
        ResetableIterator, ShiftableIterator,
    },
    structure::{
        concrete_index::ExpandedIndex, slot::IsAbstractSlot, HasStructure, StructureContract,
        TensorStructure,
    },
    tensors::data::{DataIterator, DenseTensor, SparseTensor},
};

use std::iter::Iterator;

use super::{ContractableWith, ContractionError, SingleContract, SingleContractInterleaved};

impl<T, U, I> SingleContract<DenseTensor<T, I>> for DenseTensor<U, I>
where
    U: ContractableWith<
        T,
        Out: FallibleAddAssign<U::Out> + FallibleSubAssign<U::Out> + Clone + RefZero + IsZero,
    >,
    I: TensorStructure + Clone + StructureContract,
{
    type LCM = DenseTensor<U::Out, I>;

    fn single_contract(
        &self,
        other: &DenseTensor<T, I>,
        final_structure: I,
        i: usize,
        j: usize,
    ) -> Result<Self::LCM, ContractionError> {
        // trace!("single contract dense dense");
        let zero = self.data[0].try_upgrade().unwrap().into_owned().ref_zero();
        let mut result_data = vec![zero.clone(); final_structure.size()?];
        let mut result_index = 0;

        let mut self_iter = self.fiber_class(i.into()).iter();
        let mut other_iter = other.fiber_class(j.into()).iter();

        let fiber_representation = self.reps()[i];

        for mut fiber_a in self_iter.by_ref() {
            for fiber_b in other_iter.by_ref() {
                for (k, ((a, _), (b, _))) in (fiber_a.by_ref()).zip(fiber_b).enumerate() {
                    if fiber_representation.is_neg(k) {
                        result_data[result_index]
                            .sub_assign_fallible(&(a.mul_fallible(b).unwrap()));
                    } else {
                        result_data[result_index].add_assign_fallible(&a.mul_fallible(b).unwrap());
                    }
                }
                result_index += 1;
                fiber_a.reset();
            }
            other_iter.reset();
        }
        let result = DenseTensor {
            data: result_data,
            structure: final_structure,
        };

        Ok(result)
    }
}

impl<T, U, I> SingleContract<SparseTensor<T, I>> for DenseTensor<U, I>
where
    U: ContractableWith<
        T,
        Out: FallibleAddAssign<U::Out> + FallibleSubAssign<U::Out> + Clone + RefZero + IsZero,
    >,
    I: TensorStructure + Clone + StructureContract,
{
    type LCM = DenseTensor<U::Out, I>;

    fn single_contract(
        &self,
        other: &SparseTensor<T, I>,
        final_structure: I,
        i: usize,
        j: usize,
    ) -> Result<Self::LCM, ContractionError> {
        trace!("single contract dense sparse");
        let zero = self.data[0].try_upgrade().unwrap().into_owned().ref_zero();
        let mut result_data = vec![zero.clone(); final_structure.size()?];
        let mut result_index = 0;

        let mut self_iter = self.fiber_class(i.into()).iter();
        let mut other_iter = other.fiber_class(j.into()).iter();

        let fiber_representation = self.reps()[i];

        for mut fiber_a in self_iter.by_ref() {
            for mut fiber_b in other_iter.by_ref() {
                for (k, (b, skip, _)) in fiber_b.by_ref().enumerate() {
                    if let Some((a, _)) = fiber_a.by_ref().nth(skip) {
                        if fiber_representation.is_neg(k + skip) {
                            result_data[result_index]
                                .sub_assign_fallible(&a.mul_fallible(b).unwrap());
                        } else {
                            result_data[result_index]
                                .add_assign_fallible(&a.mul_fallible(b).unwrap());
                        }
                    }
                }
                result_index += 1;
                fiber_a.reset();
            }
            other_iter.reset();
        }

        let result = DenseTensor {
            data: result_data,
            structure: final_structure,
        };

        Ok(result)
    }
}

impl<T, U, I> SingleContract<DenseTensor<T, I>> for SparseTensor<U, I>
where
    U: ContractableWith<
        T,
        Out: FallibleAddAssign<U::Out> + FallibleSubAssign<U::Out> + Clone + RefZero + IsZero,
    >,
    T: TrySmallestUpgrade<U, LCM = U::Out>,
    I: TensorStructure + Clone + StructureContract,
{
    type LCM = DenseTensor<U::Out, I>;

    fn single_contract(
        &self,
        other: &DenseTensor<T, I>,
        final_structure: I,
        i: usize,
        j: usize,
    ) -> Result<Self::LCM, ContractionError> {
        trace!("single contract sparse dense");
        let zero = if let Some((_, s)) = self.flat_iter().next() {
            s.try_upgrade().unwrap().as_ref().ref_zero()
        } else if let Some((_, o)) = other.iter_flat().next() {
            o.try_upgrade().unwrap().as_ref().ref_zero()
        } else {
            return Err(ContractionError::EmptySparse);
        };

        let mut result_data = vec![zero.clone(); final_structure.size()?];
        let mut result_index = 0;

        let mut self_iter = self.fiber_class(i.into()).iter();
        let mut other_iter = other.fiber_class(j.into()).iter();

        let fiber_representation = self.reps()[i];

        for mut fiber_a in self_iter.by_ref() {
            for mut fiber_b in other_iter.by_ref() {
                for (k, (a, skip, _)) in fiber_a.by_ref().enumerate() {
                    if let Some((b, _)) = fiber_b.by_ref().nth(skip) {
                        if fiber_representation.is_neg(k + skip) {
                            result_data[result_index]
                                .sub_assign_fallible(&a.mul_fallible(b).unwrap());
                        } else {
                            result_data[result_index]
                                .add_assign_fallible(&a.mul_fallible(b).unwrap());
                        }
                    }
                }
                result_index += 1;
                fiber_a.reset();
            }
            other_iter.reset();
        }

        let result = DenseTensor {
            data: result_data,
            structure: final_structure,
        };

        Ok(result)
    }
}

impl<T, U, I> SingleContract<SparseTensor<T, I>> for SparseTensor<U, I>
where
    U: ContractableWith<
        T,
        Out: FallibleAddAssign<U::Out> + FallibleSubAssign<U::Out> + Clone + RefZero + IsZero,
    >,
    I: TensorStructure + Clone + StructureContract,
{
    type LCM = SparseTensor<U::Out, I>;
    #[allow(clippy::comparison_chain)]
    fn single_contract(
        &self,
        other: &SparseTensor<T, I>,
        final_structure: I,
        i: usize,
        j: usize,
    ) -> Result<Self::LCM, ContractionError> {
        trace!("single contract sparse sparse");

        let mut result_data = HashMap::default();
        if let Some((_, s)) = self.flat_iter().next() {
            let zero = s.try_upgrade().unwrap().as_ref().ref_zero();
            let mut result_index = 0;

            let self_iter = self.fiber_class(i.into()).iter();
            let mut other_iter = other.fiber_class(j.into()).iter();

            let metric = self.external_structure()[i].rep().negative()?;

            for mut fiber_a in self_iter {
                for mut fiber_b in other_iter.by_ref() {
                    let mut items = fiber_a
                        .next()
                        .map(|(a, skip, _)| (a, skip))
                        .zip(fiber_b.next().map(|(b, skip, _)| (b, skip)));

                    let mut value = zero.clone();
                    let mut nonzero = false;

                    while let Some(((a, skip_a), (b, skip_b))) = items {
                        if skip_a > skip_b {
                            let b = fiber_b
                                .by_ref()
                                .next()
                                .map(|(b, skip, _)| (b, skip + skip_b + 1));
                            items = Some((a, skip_a)).zip(b);
                        } else if skip_b > skip_a {
                            let a = fiber_a
                                .by_ref()
                                .next()
                                .map(|(a, skip, _)| (a, skip + skip_a + 1));
                            items = a.zip(Some((b, skip_b)));
                        } else {
                            if metric[skip_a] {
                                value.sub_assign_fallible(&a.mul_fallible(b).unwrap());
                            } else {
                                value.add_assign_fallible(&a.mul_fallible(b).unwrap());
                            }
                            let b = fiber_b
                                .by_ref()
                                .next()
                                .map(|(b, skip, _)| (b, skip + skip_b + 1));
                            let a = fiber_a
                                .by_ref()
                                .next()
                                .map(|(a, skip, _)| (a, skip + skip_a + 1));
                            items = a.zip(b);
                            nonzero = true;
                        }
                    }
                    if nonzero && !value.is_zero() {
                        result_data.insert(result_index.into(), value);
                    }
                    result_index += 1;
                    fiber_a.reset();
                }
                other_iter.reset();
            }
        }
        let result = SparseTensor {
            elements: result_data,
            structure: final_structure,
        };

        Ok(result)
    }
}

impl<T, U, I> SingleContractInterleaved<DenseTensor<T, I>> for DenseTensor<U, I>
where
    U: ContractableWith<
        T,
        Out: FallibleAddAssign<U::Out> + FallibleSubAssign<U::Out> + Clone + RefZero + IsZero,
    >,
    I: TensorStructure + Clone + StructureContract,
{
    type LCM = DenseTensor<U::Out, I>;

    fn single_contract_interleaved(
        &self,
        other: &DenseTensor<T, I>,
        resulting_structure: <Self::LCM as crate::structure::HasStructure>::Structure,
        resulting_partition: bitvec::prelude::BitVec,
        i: usize,
        j: usize,
    ) -> Result<Self::LCM, ContractionError> {
        let zero = self.data[0].try_upgrade().unwrap().into_owned().ref_zero();
        let mut result_data = vec![zero.clone(); resulting_structure.size()?];

        let self_fiber_class = Fiber::from(&resulting_partition, &resulting_structure);
        let (mut self_fiber_class_iter, mut other_fiber_class_iter) =
            CoreFlatFiberIterator::new_paired_conjugates(&self_fiber_class);

        let mut iter_self = self.fiber(FiberData::from(i)).iter();
        let mut iter_other = other.fiber(FiberData::from(j)).iter();

        let fiber_representation = self.reps()[i];
        let fiber_dim: usize = fiber_representation.dim.try_into()?;

        for mut fiber_class_a_id in self_fiber_class_iter.by_ref() {
            for fiber_class_b_id in other_fiber_class_iter.by_ref() {
                let result_index = fiber_class_a_id + fiber_class_b_id;
                let ((_, expa), (_, expb)): ((Vec<_>, ExpandedIndex), (Vec<_>, ExpandedIndex)) =
                    resulting_structure
                        .expanded_index(result_index)?
                        .into_iter()
                        .enumerate()
                        .partition(|(i, a)| resulting_partition[*i]);

                let shift_a = self.structure().flat_index(expa).unwrap();
                let shift_b = other.structure().flat_index(expb).unwrap();

                iter_self.shift(shift_a.into());
                iter_other.shift(shift_b.into());

                for (k, ((a, _), (b, _))) in
                    (iter_self.by_ref()).zip(iter_other.by_ref()).enumerate()
                {
                    if fiber_representation.is_neg(k) {
                        result_data[usize::from(result_index)]
                            .sub_assign_fallible(&(a.mul_fallible(b).unwrap()));
                    } else {
                        result_data[usize::from(result_index)]
                            .add_assign_fallible(&a.mul_fallible(b).unwrap());
                    }
                }
            }
            other_fiber_class_iter.reset();
        }
        let result = DenseTensor {
            data: result_data,
            structure: resulting_structure,
        };

        Ok(result)
        // todo!()
    }
}

impl<T, U, I> SingleContractInterleaved<SparseTensor<T, I>> for DenseTensor<U, I>
where
    U: ContractableWith<
        T,
        Out: FallibleAddAssign<U::Out> + FallibleSubAssign<U::Out> + Clone + RefZero + IsZero,
    >,
    I: TensorStructure + Clone + StructureContract,
{
    type LCM = DenseTensor<U::Out, I>;

    fn single_contract_interleaved(
        &self,
        other: &SparseTensor<T, I>,
        resulting_structure: <Self::LCM as crate::structure::HasStructure>::Structure,
        resulting_partition: bitvec::prelude::BitVec,
        i: usize,
        j: usize,
    ) -> Result<Self::LCM, ContractionError> {
        todo!()
    }
}

impl<T, U, I> SingleContractInterleaved<DenseTensor<T, I>> for SparseTensor<U, I>
where
    U: ContractableWith<
        T,
        Out: FallibleAddAssign<U::Out> + FallibleSubAssign<U::Out> + Clone + RefZero + IsZero,
    >,
    T: TrySmallestUpgrade<U, LCM = U::Out>,
    I: TensorStructure + Clone + StructureContract,
{
    type LCM = DenseTensor<U::Out, I>;

    fn single_contract_interleaved(
        &self,
        other: &DenseTensor<T, I>,
        resulting_structure: <Self::LCM as crate::structure::HasStructure>::Structure,
        resulting_partition: bitvec::prelude::BitVec,
        i: usize,
        j: usize,
    ) -> Result<Self::LCM, ContractionError> {
        todo!()
    }
}

impl<T, U, I> SingleContractInterleaved<SparseTensor<T, I>> for SparseTensor<U, I>
where
    U: ContractableWith<
        T,
        Out: FallibleAddAssign<U::Out> + FallibleSubAssign<U::Out> + Clone + RefZero + IsZero,
    >,
    I: TensorStructure + Clone + StructureContract,
{
    type LCM = SparseTensor<U::Out, I>;
    #[allow(clippy::comparison_chain)]

    fn single_contract_interleaved(
        &self,
        other: &SparseTensor<T, I>,
        resulting_structure: <Self::LCM as crate::structure::HasStructure>::Structure,
        resulting_partition: bitvec::prelude::BitVec,
        i: usize,
        j: usize,
    ) -> Result<Self::LCM, ContractionError> {
        todo!()
    }
}
