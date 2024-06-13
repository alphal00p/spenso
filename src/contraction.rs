use ahash::AHashMap;

use crate::TrySmallestUpgrade;

use super::{
    DataIterator, DataTensor, DenseTensor, FallibleAddAssign, FallibleMul, FallibleSubAssign,
    HasStructure, NumTensor, Representation, SetTensorData, SparseTensor, StructureContract,
};

use std::iter::Iterator;

// pub trait LeastCommonStorage<Other: HasTensorData + SetTensorData>:
//     HasTensorData + SetTensorData
// {
//     type LCMData;
//     type OutStorage: SetTensorData<SetData = Self::LCMData>;
//     fn least_common_storage(&self, other: &Other) -> Self::OutStorage;

//     fn empty(structure: Self::Structure) -> Self::OutStorage;
// }

// impl<T, U, I, LCMData> LeastCommonStorage<DenseTensor<T, I>> for DenseTensor<U, I>
// where
//     T: ContractableWith<U, Out = LCMData> + Clone,
//     U: ContractableWith<T, Out = LCMData> + Clone,
//     LCMData: Default + Clone,
//     I: HasStructure + StructureContract + Clone,
// {
//     type OutStorage = DenseTensor<LCMData, I>;
//     type LCMData = LCMData;

//     fn empty(structure: Self::Structure) -> Self::OutStorage {
//         DenseTensor::default(structure)
//     }

//     fn least_common_storage(&self, other: &DenseTensor<T, I>) -> Self::OutStorage {
//         let mut final_structure = self.structure().clone();
//         final_structure.merge(other.structure());
//         DenseTensor::default(final_structure)
//     }
// }

// impl<T, U, I, LCMData> LeastCommonStorage<DenseTensor<T, I>> for SparseTensor<U, I>
// where
//     T: ContractableWith<U, Out = LCMData> + Clone,
//     U: ContractableWith<T, Out = LCMData> + Clone,
//     LCMData: Default + Clone,
//     I: HasStructure + StructureContract + Clone,
// {
//     type OutStorage = DenseTensor<LCMData, I>;
//     type LCMData = LCMData;

//     fn empty(structure: Self::Structure) -> Self::OutStorage {
//         DenseTensor::default(structure)
//     }

//     fn least_common_storage(&self, other: &DenseTensor<T, I>) -> Self::OutStorage {
//         let mut final_structure = self.structure().clone();
//         final_structure.merge(other.structure());
//         DenseTensor::default(final_structure)
//     }
// }

// impl<T, U, I, LCMData> LeastCommonStorage<SparseTensor<T, I>> for DenseTensor<U, I>
// where
//     T: ContractableWith<U, Out = LCMData> + Clone,
//     U: ContractableWith<T, Out = LCMData> + Clone,
//     LCMData: Default + Clone,
//     I: HasStructure + StructureContract + Clone,
// {
//     type OutStorage = DenseTensor<LCMData, I>;
//     type LCMData = LCMData;

//     fn empty(structure: Self::Structure) -> Self::OutStorage {
//         DenseTensor::default(structure)
//     }

//     fn least_common_storage(&self, other: &SparseTensor<T, I>) -> Self::OutStorage {
//         let mut final_structure = self.structure().clone();
//         final_structure.merge(other.structure());
//         DenseTensor::default(final_structure)
//     }
// }

// impl<T, U, I, LCMData> LeastCommonStorage<SparseTensor<T, I>> for SparseTensor<U, I>
// where
//     T: ContractableWith<U, Out = LCMData> + Clone,
//     U: ContractableWith<T, Out = LCMData> + Clone,
//     LCMData: Default + Clone,
//     I: HasStructure + StructureContract + Clone,
// {
//     type OutStorage = SparseTensor<LCMData, I>;
//     type LCMData = LCMData;

//     fn empty(structure: Self::Structure) -> Self::OutStorage {
//         SparseTensor::empty(structure)
//     }

//     fn least_common_storage(&self, other: &SparseTensor<T, I>) -> Self::OutStorage {
//         let mut final_structure = self.structure().clone();
//         final_structure.merge(other.structure());
//         SparseTensor::empty(final_structure)
//     }
// }

pub trait ExteriorProduct<T> {
    type LCM;
    fn exterior_product(&self, other: &T) -> Self::LCM;
}

impl<T, U, I, O> ExteriorProduct<DenseTensor<T, I>> for DenseTensor<U, I>
where
    U: ContractableWith<T, Out = O>,
    T: ContractableWith<U, Out = O>,
    O: FallibleAddAssign<O> + FallibleSubAssign<O> + Clone + RefZero + IsZero,
    I: HasStructure + Clone + StructureContract,
{
    type LCM = DenseTensor<U::Out, I>;

    fn exterior_product(&self, other: &DenseTensor<T, I>) -> Self::LCM {
        let mut final_structure = self.structure().clone();
        final_structure.merge(other.structure());
        let zero = self.data[0].try_upgrade().unwrap().into_owned().zero();
        let mut out = DenseTensor {
            data: vec![zero.clone(); final_structure.size()],
            structure: final_structure,
        };

        let stride = other.size();

        for (i, u) in self.flat_iter() {
            for (j, t) in other.flat_iter() {
                let _ = out.set_flat(i * stride + j, u.mul_fallible(t).unwrap());
            }
        }

        out
    }
}

impl<T, U, I, O> ExteriorProduct<DenseTensor<T, I>> for SparseTensor<U, I>
where
    U: ContractableWith<T, Out = O>,
    T: ContractableWith<U, Out = O>,
    O: FallibleAddAssign<O> + FallibleSubAssign<O> + Clone + RefZero + IsZero,
    I: HasStructure + Clone + StructureContract,
{
    type LCM = DenseTensor<U::Out, I>;

    fn exterior_product(&self, other: &DenseTensor<T, I>) -> Self::LCM {
        let mut final_structure = self.structure().clone();
        final_structure.merge(other.structure());
        let zero = other.data[0].try_upgrade().unwrap().into_owned().zero();
        let mut out = DenseTensor {
            data: vec![zero.clone(); final_structure.size()],
            structure: final_structure,
        };

        let stride = other.size();

        for (i, u) in self.flat_iter() {
            for (j, t) in other.flat_iter() {
                let _ = out.set_flat(i * stride + j, u.mul_fallible(t).unwrap());
            }
        }

        out
    }
}

impl<T, U, I, O> ExteriorProduct<SparseTensor<T, I>> for DenseTensor<U, I>
where
    U: ContractableWith<T, Out = O>,
    T: ContractableWith<U, Out = O>,
    O: FallibleAddAssign<O> + FallibleSubAssign<O> + Clone + RefZero + IsZero,
    I: HasStructure + Clone + StructureContract,
{
    type LCM = DenseTensor<U::Out, I>;

    fn exterior_product(&self, other: &SparseTensor<T, I>) -> Self::LCM {
        let mut final_structure = self.structure().clone();
        final_structure.merge(other.structure());
        let zero = self.data[0].try_upgrade().unwrap().into_owned().zero();
        let mut out = DenseTensor {
            data: vec![zero.clone(); final_structure.size()],
            structure: final_structure,
        };
        let stride = other.size();

        for (i, u) in self.flat_iter() {
            for (j, t) in other.flat_iter() {
                let _ = out.set_flat(i * stride + j, u.mul_fallible(t).unwrap());
            }
        }

        out
    }
}

impl<T, U, I, O> ExteriorProduct<SparseTensor<T, I>> for SparseTensor<U, I>
where
    U: ContractableWith<T, Out = O>,
    T: ContractableWith<U, Out = O>,
    O: FallibleAddAssign<O> + FallibleSubAssign<O> + Clone + RefZero + IsZero,
    I: HasStructure + Clone + StructureContract,
{
    type LCM = SparseTensor<U::Out, I>;

    fn exterior_product(&self, other: &SparseTensor<T, I>) -> Self::LCM {
        let mut final_structure = self.structure().clone();
        final_structure.merge(other.structure());

        let mut out = SparseTensor::empty(final_structure);
        let stride = other.size();

        for (i, u) in self.flat_iter() {
            for (j, t) in other.flat_iter() {
                let _ = out.set_flat(i * stride + j, u.mul_fallible(t).unwrap());
            }
        }

        out
    }
}

impl<T, I> DenseTensor<T, I>
where
    T: ContractableWith<T, Out = T> + Clone + RefZero + FallibleAddAssign<T> + FallibleSubAssign<T>,
    I: HasStructure + Clone + StructureContract,
{
    #[must_use]

    /// Contract the tensor with itself, i.e. trace over all matching indices.
    pub fn internal_contract(&self) -> Self {
        let mut result: DenseTensor<T, I> = self.clone();
        for trace in self.traces() {
            let mut new_structure = self.structure.clone();
            new_structure.trace(trace[0], trace[1]);

            let mut new_result = DenseTensor::from_data_coerced(&self.data, new_structure)
                .unwrap_or_else(|_| unreachable!());
            for (idx, t) in result.iter_trace(trace) {
                new_result.set(&idx, t).unwrap_or_else(|_| unreachable!());
            }
            result = new_result;
        }
        result
    }
}

pub trait IsZero {
    fn is_zero(&self) -> bool;

    fn is_non_zero(&self) -> bool {
        !self.is_zero()
    }
}

impl<T: RefZero + PartialEq> IsZero for T {
    fn is_zero(&self) -> bool {
        self.zero() == *self
    }
}

pub trait RefZero {
    fn zero(&self) -> Self;
}

// impl<T: num::Zero> RefZero for T { future impls Grrr
//     fn zero(&self) -> Self {
//         T::zero()
//     }
// }

impl RefZero for f64 {
    fn zero(&self) -> Self {
        0.0
    }
}

impl RefZero for f32 {
    fn zero(&self) -> Self {
        0.0
    }
}

impl RefZero for i8 {
    fn zero(&self) -> Self {
        0
    }
}

impl RefZero for i16 {
    fn zero(&self) -> Self {
        0
    }
}

impl RefZero for i32 {
    fn zero(&self) -> Self {
        0
    }
}

impl RefZero for i64 {
    fn zero(&self) -> Self {
        0
    }
}

impl RefZero for i128 {
    fn zero(&self) -> Self {
        0
    }
}

impl RefZero for u8 {
    fn zero(&self) -> Self {
        0
    }
}

impl RefZero for u16 {
    fn zero(&self) -> Self {
        0
    }
}

impl RefZero for u32 {
    fn zero(&self) -> Self {
        0
    }
}

impl RefZero for u64 {
    fn zero(&self) -> Self {
        0
    }
}

impl RefZero for u128 {
    fn zero(&self) -> Self {
        0
    }
}

impl<T, I> SparseTensor<T, I>
where
    T: ContractableWith<T, Out = T>
        + Clone
        + RefZero
        + IsZero
        + FallibleAddAssign<T>
        + FallibleSubAssign<T>,
    I: HasStructure + Clone + StructureContract,
{
    #[must_use]
    /// Contract the tensor with itself, i.e. trace over all matching indices.
    pub fn internal_contract(&self) -> Self {
        let trace = self.traces()[0];

        // println!("trace {:?}", trace);
        let mut new_structure = self.structure.clone();
        new_structure.trace(trace[0], trace[1]);

        let mut new_result = SparseTensor::empty(new_structure);
        for (idx, t) in self.iter_trace(trace).filter(|(_, t)| !t.is_zero()) {
            new_result.set(&idx, t).unwrap_or_else(|_| unreachable!());
        }

        if new_result.traces().is_empty() {
            new_result
        } else {
            new_result.internal_contract()
        }
    }
}
pub trait Contract<T> {
    type LCM;
    fn contract(&self, other: &T) -> Option<Self::LCM>;
}

pub trait SingleContract<T> {
    type LCM;
    fn single_contract(&self, other: &T, i: usize, j: usize) -> Option<Self::LCM>;
}

pub trait MultiContract<T> {
    type LCM;
    fn multi_contract(&self, other: &T) -> Option<Self::LCM>;
}
pub trait ContractableWith<T>
where
    Self: FallibleMul<T, Output = Self::Out> + Sized + TrySmallestUpgrade<T, LCM = Self::Out>,
    T: FallibleMul<Self, Output = Self::Out>,
{
    type Out: FallibleAddAssign<Self::Out> + FallibleSubAssign<Self::Out> + Clone + RefZero;
}

impl<T, U, Out> ContractableWith<T> for U
where
    U: FallibleMul<T, Output = Out> + TrySmallestUpgrade<T, LCM = Out>,
    T: FallibleMul<U, Output = Out>,
    Out: FallibleAddAssign<Out> + FallibleSubAssign<Out> + Clone + RefZero,
{
    type Out = Out;
}

impl<T, U, I, O> SingleContract<DenseTensor<T, I>> for DenseTensor<U, I>
where
    U: ContractableWith<T, Out = O>,
    T: ContractableWith<U, Out = O>,
    O: FallibleAddAssign<O> + FallibleSubAssign<O> + Clone + RefZero + IsZero,
    I: HasStructure + Clone + StructureContract,
{
    type LCM = DenseTensor<U::Out, I>;

    fn single_contract(&self, other: &DenseTensor<T, I>, i: usize, j: usize) -> Option<Self::LCM> {
        // println!("single contract dense dense");
        let zero = self.data[0].try_upgrade().unwrap().into_owned().zero();
        let final_structure = self.structure.merge_at(&other.structure, (i, j));
        let mut result_data = vec![zero.clone(); final_structure.size()];
        let mut result_index = 0;

        let mut self_iter = self.fiber_class(i.into()).iter();
        let mut other_iter = other.fiber_class(j.into()).iter();

        let fiber_representation: Representation = self.reps()[i];

        for mut fiber_a in self_iter.by_ref() {
            for fiber_b in other_iter.by_ref() {
                for (k, ((a, _), (b, _))) in (fiber_a.by_ref()).zip(fiber_b).enumerate() {
                    if fiber_representation.is_neg(k) {
                        result_data[result_index].sub_assign_fallible(a.mul_fallible(&b).unwrap());
                    } else {
                        result_data[result_index].add_assign_fallible(a.mul_fallible(b).unwrap());
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

        Some(result)
    }
}

impl<T, U, I, O> MultiContract<DenseTensor<T, I>> for DenseTensor<U, I>
where
    U: ContractableWith<T, Out = O>,
    T: ContractableWith<U, Out = O>,
    O: FallibleAddAssign<O> + FallibleSubAssign<O> + Clone + RefZero + IsZero,
    I: HasStructure + Clone + StructureContract,
{
    type LCM = DenseTensor<U::Out, I>;
    fn multi_contract(&self, other: &DenseTensor<T, I>) -> Option<Self::LCM> {
        // println!("multi contract dense dense");
        let zero = self.data[0].try_upgrade().unwrap().into_owned().zero();
        let (permutation, self_matches, other_matches) =
            self.structure().match_indices(other.structure()).unwrap();

        let mut final_structure = self.structure.clone();
        final_structure.merge(&other.structure);

        // Initialize result tensor with default values
        let mut result_data = vec![zero.clone(); final_structure.size()];
        let mut result_index = 0;

        let selfiter = self
            .fiber_class(self_matches.as_slice().into())
            .iter_perm_metric(permutation);
        let mut other_iter = other.fiber_class(other_matches.as_slice().into()).iter();

        for mut fiber_a in selfiter {
            for mut fiber_b in other_iter.by_ref() {
                for (a, (neg, _)) in fiber_a.by_ref() {
                    if let Some((b, _)) = fiber_b.next() {
                        if neg {
                            result_data[result_index]
                                .sub_assign_fallible(a.mul_fallible(b).unwrap());
                        } else {
                            result_data[result_index]
                                .add_assign_fallible(a.mul_fallible(b).unwrap());
                        }
                    }
                }
                result_index += 1;
                fiber_a.reset();
            }
            other_iter.reset();
        }
        let result: DenseTensor<U::Out, I> = DenseTensor {
            data: result_data,
            structure: final_structure,
        };

        Some(result)
    }
}

impl<T, U, O> Contract<T> for U
where
    U: SingleContract<T, LCM = O>
        + MultiContract<T, LCM = O>
        + ExteriorProduct<T, LCM = O>
        + HasStructure,
    U::Structure: HasStructure,
    T: SingleContract<U, LCM = O>
        + MultiContract<U, LCM = O>
        + ExteriorProduct<U, LCM = O>
        + HasStructure<Structure = U::Structure>,
{
    type LCM = O;
    fn contract(&self, other: &T) -> Option<Self::LCM> {
        if let Some((single, i, j)) = self.structure().match_index(other.structure()) {
            if i >= j {
                if single {
                    // println!("single");
                    return self.single_contract(other, i, j);
                }
                // println!("multi");
                return self.multi_contract(other);
            }
            // println!("flip");
            return other.contract(self);
        }
        // println!("exterior");
        let result = self.exterior_product(other);
        Some(result)
    }
}

impl<T, U, I, O> SingleContract<DenseTensor<T, I>> for SparseTensor<U, I>
where
    U: ContractableWith<T, Out = O>,
    T: ContractableWith<U, Out = O>,
    O: FallibleAddAssign<O> + FallibleSubAssign<O> + Clone + RefZero + IsZero,
    I: HasStructure + Clone + StructureContract,
{
    type LCM = DenseTensor<U::Out, I>;

    fn single_contract(&self, other: &DenseTensor<T, I>, i: usize, j: usize) -> Option<Self::LCM> {
        // println!("single contract sparse dense");
        let zero = other.data[0].try_upgrade().unwrap().into_owned().zero();
        let final_structure = self.structure.merge_at(&other.structure, (i, j));
        let mut result_data = vec![zero.clone(); final_structure.size()];
        let mut result_index = 0;

        let mut self_iter = self.fiber_class(i.into()).iter();
        let mut other_iter = other.fiber_class(j.into()).iter();

        let fiber_representation: Representation = self.reps()[i];

        for mut fiber_a in self_iter.by_ref() {
            for mut fiber_b in other_iter.by_ref() {
                for (k, (a, skip, _)) in fiber_a.by_ref().enumerate() {
                    if let Some((b, _)) = fiber_b.by_ref().nth(skip) {
                        if fiber_representation.is_neg(k + skip) {
                            result_data[result_index]
                                .sub_assign_fallible(a.mul_fallible(b).unwrap());
                        } else {
                            result_data[result_index]
                                .add_assign_fallible(a.mul_fallible(b).unwrap());
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

        Some(result)
    }
}

impl<T, U, I, O> SingleContract<SparseTensor<T, I>> for DenseTensor<U, I>
where
    U: ContractableWith<T, Out = O>,
    T: ContractableWith<U, Out = O>,
    O: FallibleAddAssign<O> + FallibleSubAssign<O> + Clone + RefZero + IsZero,
    I: HasStructure + Clone + StructureContract,
{
    type LCM = DenseTensor<U::Out, I>;

    fn single_contract(&self, other: &SparseTensor<T, I>, i: usize, j: usize) -> Option<Self::LCM> {
        // println!("single contract dense sparse");
        let zero = self.data[0].try_upgrade().unwrap().into_owned().zero();
        let final_structure = self.structure.merge_at(&other.structure, (i, j));
        let mut result_data = vec![zero.clone(); final_structure.size()];
        let mut result_index = 0;

        let mut self_iter = self.fiber_class(i.into()).iter();
        let mut other_iter = other.fiber_class(j.into()).iter();

        let fiber_representation: Representation = self.reps()[i];

        for mut fiber_a in self_iter.by_ref() {
            for mut fiber_b in other_iter.by_ref() {
                for (k, (b, skip, _)) in fiber_b.by_ref().enumerate() {
                    if let Some((a, _)) = fiber_a.by_ref().nth(skip) {
                        if fiber_representation.is_neg(k + skip) {
                            result_data[result_index]
                                .sub_assign_fallible(a.mul_fallible(b).unwrap());
                        } else {
                            result_data[result_index]
                                .add_assign_fallible(a.mul_fallible(b).unwrap());
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

        Some(result)
    }
}

impl<T, U, I, O> MultiContract<DenseTensor<T, I>> for SparseTensor<U, I>
where
    U: ContractableWith<T, Out = O>,
    T: ContractableWith<U, Out = O>,
    O: FallibleAddAssign<O> + FallibleSubAssign<O> + Clone + RefZero + IsZero,
    I: HasStructure + Clone + StructureContract,
{
    type LCM = DenseTensor<U::Out, I>;
    fn multi_contract(&self, other: &DenseTensor<T, I>) -> Option<Self::LCM> {
        // println!("multi contract sparse dense");
        let zero = other.data[0].try_upgrade().unwrap().as_ref().zero();
        let (permutation, self_matches, other_matches) =
            self.structure().match_indices(other.structure()).unwrap();

        let mut final_structure = self.structure.clone();
        let _ = final_structure.merge(&other.structure);

        let mut result_data = vec![zero.clone(); final_structure.size()];
        let mut result_index = 0;

        let selfiter = self
            .fiber_class(self_matches.as_slice().into())
            .iter_perm_metric(permutation);
        let mut other_iter = other.fiber_class(other_matches.as_slice().into()).iter();

        for mut fiber_a in selfiter {
            for mut fiber_b in other_iter.by_ref() {
                for (a, skip, (neg, _)) in fiber_a.by_ref() {
                    if let Some((b, _)) = fiber_b.by_ref().nth(skip) {
                        if neg {
                            result_data[result_index]
                                .sub_assign_fallible(a.mul_fallible(b).unwrap());
                        } else {
                            result_data[result_index]
                                .add_assign_fallible(a.mul_fallible(b).unwrap());
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

        Some(result)
    }
}

impl<T, U, I, O> MultiContract<SparseTensor<T, I>> for DenseTensor<U, I>
where
    U: ContractableWith<T, Out = O>,
    T: ContractableWith<U, Out = O>,
    O: FallibleAddAssign<O> + FallibleSubAssign<O> + Clone + RefZero + IsZero,
    I: HasStructure + Clone + StructureContract,
{
    type LCM = DenseTensor<U::Out, I>;

    fn multi_contract(&self, other: &SparseTensor<T, I>) -> Option<Self::LCM> {
        // println!("multi contract dense sparse");
        let zero = self.data[0].try_upgrade().unwrap().as_ref().zero();
        let (permutation, self_matches, other_matches) =
            self.structure().match_indices(other.structure()).unwrap();

        let mut final_structure = self.structure.clone();
        final_structure.merge(&other.structure);

        let mut result_data = vec![zero.clone(); final_structure.size()];
        let mut result_index = 0;

        let selfiter = self
            .fiber_class(self_matches.as_slice().into())
            .iter_perm_metric(permutation);
        let mut other_iter = other.fiber_class(other_matches.as_slice().into()).iter();

        for mut fiber_a in selfiter {
            for mut fiber_b in other_iter.by_ref() {
                for (b, skip, _) in fiber_b.by_ref() {
                    if let Some((a, (neg, _))) = fiber_a.by_ref().nth(skip) {
                        if neg {
                            result_data[result_index]
                                .sub_assign_fallible(a.mul_fallible(b).unwrap());
                        } else {
                            result_data[result_index]
                                .add_assign_fallible(a.mul_fallible(b).unwrap());
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

        Some(result)
    }
}

impl<T, U, I, O> SingleContract<SparseTensor<T, I>> for SparseTensor<U, I>
where
    U: ContractableWith<T, Out = O>,
    T: ContractableWith<U, Out = O>,
    O: FallibleAddAssign<O> + FallibleSubAssign<O> + Clone + RefZero + IsZero,
    I: HasStructure + Clone + StructureContract,
{
    type LCM = SparseTensor<U::Out, I>;

    fn single_contract(&self, other: &SparseTensor<T, I>, i: usize, j: usize) -> Option<Self::LCM> {
        // println!("single contract sparse sparse");

        let final_structure = self.structure.merge_at(&other.structure, (i, j));
        let mut result_data = AHashMap::default();
        if let Some((_, s)) = self.flat_iter().next() {
            let zero = s.try_upgrade().unwrap().as_ref().zero();
            let mut result_index = 0;

            let self_iter = self.fiber_class(i.into()).iter();
            let mut other_iter = other.fiber_class(j.into()).iter();

            let metric = self.external_structure()[i].representation.negative();

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
                                value.sub_assign_fallible(a.mul_fallible(b).unwrap());
                            } else {
                                value.add_assign_fallible(a.mul_fallible(b).unwrap());
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

        Some(result)
    }
}

impl<T, U, I, O> MultiContract<SparseTensor<T, I>> for SparseTensor<U, I>
where
    U: ContractableWith<T, Out = O>,
    T: ContractableWith<U, Out = O>,
    O: FallibleAddAssign<O> + FallibleSubAssign<O> + Clone + RefZero + IsZero,
    I: HasStructure + Clone + StructureContract,
{
    type LCM = SparseTensor<U::Out, I>;

    fn multi_contract(&self, other: &SparseTensor<T, I>) -> Option<Self::LCM> {
        // println!("multi contract sparse sparse");
        let (permutation, self_matches, other_matches) =
            self.structure().match_indices(other.structure()).unwrap();

        let mut final_structure = self.structure.clone();
        let _ = final_structure.merge(&other.structure);
        let mut result_data = AHashMap::default();

        if let Some((_, s)) = self.flat_iter().next() {
            let zero = s.try_upgrade().unwrap().as_ref().zero();
            let mut result_index = 0;

            let self_iter = self
                .fiber_class(self_matches.as_slice().into())
                .iter_perm_metric(permutation);
            let mut other_iter = other.fiber_class(other_matches.as_slice().into()).iter();

            for mut fiber_a in self_iter {
                for mut fiber_b in other_iter.by_ref() {
                    let mut items = fiber_a
                        .next()
                        .map(|(a, skip, (neg, _))| (a, skip, neg))
                        .zip(fiber_b.next().map(|(b, skip, _)| (b, skip)));

                    let mut value = zero.clone();
                    let mut nonzero = false;

                    while let Some(((a, skip_a, neg), (b, skip_b))) = items {
                        if skip_a > skip_b {
                            let b = fiber_b
                                .by_ref()
                                .next()
                                .map(|(b, skip, _)| (b, skip + skip_b + 1));
                            items = Some((a, skip_a, neg)).zip(b);
                        } else if skip_b > skip_a {
                            let a = fiber_a
                                .by_ref()
                                .next()
                                .map(|(a, skip, (neg, _))| (a, skip + skip_a + 1, neg));
                            items = a.zip(Some((b, skip_b)));
                        } else {
                            // println!("v{:?}", value);
                            if neg {
                                value.sub_assign_fallible(a.mul_fallible(b).unwrap());
                            } else {
                                value.add_assign_fallible(a.mul_fallible(b).unwrap());
                            }
                            let b = fiber_b
                                .by_ref()
                                .next()
                                .map(|(b, skip, _)| (b, skip + skip_b + 1));
                            let a = fiber_a
                                .by_ref()
                                .next()
                                .map(|(a, skip, (neg, _))| (a, skip + skip_a + 1, neg));
                            items = a.zip(b);
                            nonzero = true;
                        }
                    }
                    if nonzero && value.is_non_zero() {
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

        Some(result)
    }
}

impl<T, U, I, O> Contract<DataTensor<T, I>> for DataTensor<U, I>
where
    U: ContractableWith<T, Out = O>,
    T: ContractableWith<U, Out = O>,
    O: FallibleAddAssign<O> + FallibleSubAssign<O> + Clone + RefZero + IsZero,
    I: HasStructure + Clone + StructureContract,
{
    type LCM = DataTensor<U::Out, I>;
    fn contract(&self, other: &DataTensor<T, I>) -> Option<DataTensor<U::Out, I>> {
        match (self, other) {
            (DataTensor::Dense(s), DataTensor::Dense(o)) => Some(DataTensor::Dense(s.contract(o)?)),
            (DataTensor::Dense(s), DataTensor::Sparse(o)) => {
                Some(DataTensor::Dense(s.contract(o)?))
            }
            (DataTensor::Sparse(s), DataTensor::Dense(o)) => {
                Some(DataTensor::Dense(s.contract(o)?))
            }
            (DataTensor::Sparse(s), DataTensor::Sparse(o)) => {
                Some(DataTensor::Sparse(s.contract(o)?))
            }
        }
    }
}

impl<I> Contract<NumTensor<I>> for NumTensor<I>
where
    I: Clone + HasStructure + StructureContract,
{
    type LCM = NumTensor<I>;
    fn contract(&self, other: &NumTensor<I>) -> Option<Self::LCM> {
        match (self, other) {
            (NumTensor::Float(a), NumTensor::Float(b)) => Some(NumTensor::Float(a.contract(b)?)),
            (NumTensor::Float(a), NumTensor::Complex(b)) => {
                Some(NumTensor::Complex(a.contract(b)?))
            }
            (NumTensor::Complex(a), NumTensor::Float(b)) => {
                Some(NumTensor::Complex(a.contract(b)?))
            }
            (NumTensor::Complex(a), NumTensor::Complex(b)) => {
                Some(NumTensor::Complex(a.contract(b)?))
            }
        }
    }
}
