use ahash::AHashMap;
use duplicate::duplicate;
// use num::Zero;
#[cfg(feature = "shadowing")]
use symbolica::{atom::Atom, domains::float::Real};

use crate::{IteratableTensor, TensorStructure, TrySmallestUpgrade};

use super::{
    DataIterator, DataTensor, DenseTensor, FallibleAddAssign, FallibleMul, FallibleSubAssign,
    HasStructure, NumTensor, Representation, SetTensorData, SparseTensor, StructureContract,
};

use std::iter::Iterator;
use thiserror::Error;

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
//     I: TensorStructure + StructureContract + Clone,
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
//     I: TensorStructure + StructureContract + Clone,
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
//     I: TensorStructure + StructureContract + Clone,
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
//     I: TensorStructure + StructureContract + Clone,
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
    fn exterior_product(&self, other: &T) -> Result<Self::LCM, ContractionError>;
}

impl<T, U, I, O> ExteriorProduct<DenseTensor<T, I>> for DenseTensor<U, I>
where
    U: ContractableWith<T, Out = O>,
    // T: ContractableWith<U, Out = O>,
    O: FallibleAddAssign<O> + FallibleSubAssign<O> + Clone + RefZero + IsZero,
    I: TensorStructure + Clone + StructureContract,
{
    type LCM = DenseTensor<U::Out, I>;

    fn exterior_product(&self, other: &DenseTensor<T, I>) -> Result<Self::LCM, ContractionError> {
        let mut final_structure = self.structure().clone();
        final_structure.merge(other.structure());
        let zero = self.data[0].try_upgrade().unwrap().into_owned().ref_zero();
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

        Ok(out)
    }
}

// impl<T, U, I, O> ExteriorProduct<DenseTensor<U, I>> for DenseTensor<T, I>
// where
//     U: ContractableWith<T, Out = O>,
//     // T: ContractableWith<U, Out = O>,
//     O: FallibleAddAssign<O> + FallibleSubAssign<O> + Clone + RefZero + IsZero,
//     I: TensorStructure + Clone + StructureContract,
// {
//     type LCM = DenseTensor<U::Out, I>;

//     fn exterior_product(&self, other: &DenseTensor<T, I>) -> Option<Self::LCM> {
//         let mut final_structure = self.structure().clone();
//         final_structure.merge(other.structure());
//         let zero = self.data[0].try_upgrade().unwrap().into_owned().ref_zero();
//         let mut out = DenseTensor {
//             data: vec![zero.clone(); final_structure.size()],
//             structure: final_structure,
//         };

//         let stride = other.size();

//         for (i, u) in self.flat_iter() {
//             for (j, t) in other.flat_iter() {
//                 let _ = out.set_flat(i * stride + j, u.mul_fallible(t).unwrap());
//             }
//         }

//         Ok(out)
//     }
// }

impl<T, U, I, O> ExteriorProduct<DenseTensor<T, I>> for SparseTensor<U, I>
where
    U: ContractableWith<T, Out = O>,
    // T: ContractableWith<U, Out = O>,
    O: FallibleAddAssign<O> + FallibleSubAssign<O> + Clone + RefZero + IsZero,
    I: TensorStructure + Clone + StructureContract,
{
    type LCM = DenseTensor<U::Out, I>;

    fn exterior_product(&self, other: &DenseTensor<T, I>) -> Result<Self::LCM, ContractionError> {
        let mut final_structure = self.structure().clone();
        final_structure.merge(other.structure());
        if let Some((_, s)) = self.flat_iter().next() {
            let zero = s.try_upgrade().unwrap().as_ref().ref_zero();
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

            Ok(out)
        } else {
            Err(ContractionError::EmptySparse)
        }
    }
}

impl<T, U, I, O> ExteriorProduct<SparseTensor<T, I>> for DenseTensor<U, I>
where
    U: ContractableWith<T, Out = O>,
    // T: ContractableWith<U, Out = O>,
    O: FallibleAddAssign<O> + FallibleSubAssign<O> + Clone + RefZero + IsZero,
    I: TensorStructure + Clone + StructureContract,
{
    type LCM = DenseTensor<U::Out, I>;

    fn exterior_product(&self, other: &SparseTensor<T, I>) -> Result<Self::LCM, ContractionError> {
        let mut final_structure = self.structure().clone();
        final_structure.merge(other.structure());
        let zero = self.data[0].try_upgrade().unwrap().into_owned().ref_zero();
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

        Ok(out)
    }
}

impl<T, U, I, O> ExteriorProduct<SparseTensor<T, I>> for SparseTensor<U, I>
where
    U: ContractableWith<T, Out = O>,
    // T: ContractableWith<U, Out = O>,
    O: FallibleAddAssign<O> + FallibleSubAssign<O> + Clone + RefZero + IsZero,
    I: TensorStructure + Clone + StructureContract,
{
    type LCM = SparseTensor<U::Out, I>;

    fn exterior_product(&self, other: &SparseTensor<T, I>) -> Result<Self::LCM, ContractionError> {
        let mut final_structure = self.structure().clone();
        final_structure.merge(other.structure());

        let mut out = SparseTensor::empty(final_structure);
        let stride = other.size();

        for (i, u) in self.flat_iter() {
            for (j, t) in other.flat_iter() {
                let _ = out.set_flat(i * stride + j, u.mul_fallible(t).unwrap());
            }
        }

        Ok(out)
    }
}

impl<T, I> DenseTensor<T, I>
where
    T: ContractableWith<T, Out = T> + Clone + RefZero + FallibleAddAssign<T> + FallibleSubAssign<T>,
    I: TensorStructure + Clone + StructureContract,
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
        self.ref_zero() == *self
    }
}

pub trait RefZero {
    fn ref_zero(&self) -> Self;
}

pub trait RefOne {
    fn ref_one(&self) -> Self;
}

// #[cfg(feature = "shadowing")]
// impl<T: RefZero + NumericalFloatLike> RefZero for T {
//     fn ref_zero(&self) -> Self {
//         self.zero()
//     }
// } fu

#[cfg(feature = "shadowing")]
impl RefZero for Atom {
    fn ref_zero(&self) -> Self {
        Atom::new_num(0)
    }
}

#[cfg(feature = "shadowing")]
impl<T: RefOne + Real + RefZero> RefOne for symbolica::domains::float::Complex<T> {
    fn ref_one(&self) -> Self {
        Self::new(self.re.ref_one(), self.im.ref_zero())
    }
}

#[cfg(feature = "shadowing")]
impl<T: Real + RefZero> RefZero for symbolica::domains::float::Complex<T> {
    fn ref_zero(&self) -> Self {
        Self::new(self.re.ref_zero(), self.im.ref_zero())
    }
}

// impl<T: num::Zero> RefZero for T { future impls Grrr
//     fn zero(&self) -> Self {
//         T::zero()
//     }
// }

duplicate! {
    [types zero one;
        [f32] [0.0] [1.];
        [f64] [0.0] [1.];
        [i8] [0] [1];
        [i16] [0] [1] ;
        [i32] [0] [1];
        [i64] [0] [1];
        [i128] [0] [1];
        [u8] [0] [1];
        [u16] [0] [1];
        [u32] [0] [1];
        [u64] [0] [1];
        [u128] [0] [1];
        ]

    impl RefZero for types{
        fn ref_zero(&self)-> Self{
            zero
        }
    }

    impl RefOne for types{
        fn ref_one(&self)-> Self{
            one
        }
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
    I: TensorStructure + Clone + StructureContract,
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
    fn contract(&self, other: &T) -> Result<Self::LCM, ContractionError>;
}

#[derive(Error, Debug)]
pub enum ContractionError {
    #[error("Sparse tensor is empty")]
    EmptySparse,
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub trait SingleContract<T> {
    type LCM;
    fn single_contract(&self, other: &T, i: usize, j: usize)
        -> Result<Self::LCM, ContractionError>;
}

pub trait MultiContract<T> {
    type LCM;
    fn multi_contract(&self, other: &T) -> Result<Self::LCM, ContractionError>;
}
pub trait ContractableWith<T>
where
    Self: FallibleMul<T, Output = Self::Out> + Sized + TrySmallestUpgrade<T, LCM = Self::Out>,
{
    type Out: FallibleAddAssign<Self::Out> + FallibleSubAssign<Self::Out> + Clone + RefZero;
}

impl<T, U, Out> ContractableWith<T> for U
where
    U: FallibleMul<T, Output = Out> + TrySmallestUpgrade<T, LCM = Out>,
    // T: FallibleMul<U, Output = Out>,
    Out: FallibleAddAssign<Out> + FallibleSubAssign<Out> + Clone + RefZero,
{
    type Out = Out;
}

impl<T, U, I, O> SingleContract<DenseTensor<T, I>> for DenseTensor<U, I>
where
    U: ContractableWith<T, Out = O>,
    // T: ContractableWith<U, Out = O>,
    O: FallibleAddAssign<O> + FallibleSubAssign<O> + Clone + RefZero + IsZero,
    I: TensorStructure + Clone + StructureContract,
{
    type LCM = DenseTensor<U::Out, I>;

    fn single_contract(
        &self,
        other: &DenseTensor<T, I>,
        i: usize,
        j: usize,
    ) -> Result<Self::LCM, ContractionError> {
        // println!("single contract dense dense");
        let zero = self.data[0].try_upgrade().unwrap().into_owned().ref_zero();
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
                        result_data[result_index]
                            .sub_assign_fallible(&(a.mul_fallible(&b).unwrap()));
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

impl<T, U, I, O> MultiContract<DenseTensor<T, I>> for DenseTensor<U, I>
where
    U: ContractableWith<T, Out = O>,
    // T: ContractableWith<U, Out = O>,
    O: FallibleAddAssign<O> + FallibleSubAssign<O> + Clone + RefZero + IsZero,
    I: TensorStructure + Clone + StructureContract,
{
    type LCM = DenseTensor<U::Out, I>;
    fn multi_contract(&self, other: &DenseTensor<T, I>) -> Result<Self::LCM, ContractionError> {
        // println!("multi contract dense dense");
        let zero = self.data[0].try_upgrade().unwrap().into_owned().ref_zero();
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
        let result: DenseTensor<U::Out, I> = DenseTensor {
            data: result_data,
            structure: final_structure,
        };

        Ok(result)
    }
}

impl<T, U, O> Contract<T> for U
where
    U: SingleContract<T, LCM = O>
        + MultiContract<T, LCM = O>
        + ExteriorProduct<T, LCM = O>
        + HasStructure,
    // U::Structure: TensorStructure,
    T: HasStructure<Structure = U::Structure>,
    T: SingleContract<U, LCM = O>
        + MultiContract<U, LCM = O>
        + ExteriorProduct<U, LCM = O>
        + HasStructure<Structure = U::Structure>,
{
    type LCM = O;
    fn contract(&self, other: &T) -> Result<Self::LCM, ContractionError> {
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
        self.exterior_product(other)
    }
}

impl<T, U, I, O> SingleContract<DenseTensor<T, I>> for SparseTensor<U, I>
where
    U: ContractableWith<T, Out = O>,
    // T: ContractableWith<U, Out = O>,
    O: FallibleAddAssign<O> + FallibleSubAssign<O> + Clone + RefZero + IsZero,
    I: TensorStructure + Clone + StructureContract,
{
    type LCM = DenseTensor<U::Out, I>;

    fn single_contract(
        &self,
        other: &DenseTensor<T, I>,
        i: usize,
        j: usize,
    ) -> Result<Self::LCM, ContractionError> {
        // println!("single contract sparse dense");
        if let Some((_, s)) = self.flat_iter().next() {
            let zero = s.try_upgrade().unwrap().as_ref().ref_zero();
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

            return Ok(result);
        } else {
            Err(ContractionError::EmptySparse)
        }
    }
}

impl<T, U, I, O> SingleContract<SparseTensor<T, I>> for DenseTensor<U, I>
where
    U: ContractableWith<T, Out = O>,
    // T: ContractableWith<U, Out = O>,
    O: FallibleAddAssign<O> + FallibleSubAssign<O> + Clone + RefZero + IsZero,
    I: TensorStructure + Clone + StructureContract,
{
    type LCM = DenseTensor<U::Out, I>;

    fn single_contract(
        &self,
        other: &SparseTensor<T, I>,
        i: usize,
        j: usize,
    ) -> Result<Self::LCM, ContractionError> {
        // println!("single contract dense sparse");
        let zero = self.data[0].try_upgrade().unwrap().into_owned().ref_zero();
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

impl<T, U, I, O> MultiContract<DenseTensor<T, I>> for SparseTensor<U, I>
where
    U: ContractableWith<T, Out = O>,
    // T: ContractableWith<U, Out = O>,
    O: FallibleAddAssign<O> + FallibleSubAssign<O> + Clone + RefZero + IsZero,
    I: TensorStructure + Clone + StructureContract,
{
    type LCM = DenseTensor<U::Out, I>;
    fn multi_contract(&self, other: &DenseTensor<T, I>) -> Result<Self::LCM, ContractionError> {
        // println!("multi contract sparse dense");
        if let Some((_, s)) = self.flat_iter().next() {
            let zero = s.try_upgrade().unwrap().as_ref().ref_zero();
            // let zero = other.data[0].try_upgrade().unwrap().as_ref().ref_zero();
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
        } else {
            Err(ContractionError::EmptySparse)
        }
    }
}

impl<T, U, I, O> MultiContract<SparseTensor<T, I>> for DenseTensor<U, I>
where
    U: ContractableWith<T, Out = O>,
    // T: ContractableWith<U, Out = O>,
    O: FallibleAddAssign<O> + FallibleSubAssign<O> + Clone + RefZero + IsZero,
    I: TensorStructure + Clone + StructureContract,
{
    type LCM = DenseTensor<U::Out, I>;

    fn multi_contract(&self, other: &SparseTensor<T, I>) -> Result<Self::LCM, ContractionError> {
        // println!("multi contract dense sparse");
        let zero = self.data[0].try_upgrade().unwrap().as_ref().ref_zero();
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

impl<T, U, I, O> SingleContract<SparseTensor<T, I>> for SparseTensor<U, I>
where
    U: ContractableWith<T, Out = O>,
    // T: ContractableWith<U, Out = O>,
    O: FallibleAddAssign<O> + FallibleSubAssign<O> + Clone + RefZero + IsZero,
    I: TensorStructure + Clone + StructureContract,
{
    type LCM = SparseTensor<U::Out, I>;

    fn single_contract(
        &self,
        other: &SparseTensor<T, I>,
        i: usize,
        j: usize,
    ) -> Result<Self::LCM, ContractionError> {
        // println!("single contract sparse sparse");

        let final_structure = self.structure.merge_at(&other.structure, (i, j));
        let mut result_data = AHashMap::default();
        if let Some((_, s)) = self.flat_iter().next() {
            let zero = s.try_upgrade().unwrap().as_ref().ref_zero();
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

impl<T, U, I, O> MultiContract<SparseTensor<T, I>> for SparseTensor<U, I>
where
    U: ContractableWith<T, Out = O>,
    // T: ContractableWith<U, Out = O>,
    O: FallibleAddAssign<O> + FallibleSubAssign<O> + Clone + RefZero + IsZero,
    I: TensorStructure + Clone + StructureContract,
{
    type LCM = SparseTensor<U::Out, I>;

    fn multi_contract(&self, other: &SparseTensor<T, I>) -> Result<Self::LCM, ContractionError> {
        // println!("multi contract sparse sparse");
        let (permutation, self_matches, other_matches) =
            self.structure().match_indices(other.structure()).unwrap();

        let mut final_structure = self.structure.clone();
        let _ = final_structure.merge(&other.structure);
        let mut result_data = AHashMap::default();

        if let Some((_, s)) = self.flat_iter().next() {
            let zero = s.try_upgrade().unwrap().as_ref().ref_zero();
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

        Ok(result)
    }
}

impl<T, U, I, O> Contract<DataTensor<T, I>> for DataTensor<U, I>
where
    U: ContractableWith<T, Out = O>,
    T: ContractableWith<U, Out = O>,
    O: FallibleAddAssign<O> + FallibleSubAssign<O> + Clone + RefZero + IsZero,
    I: TensorStructure + Clone + StructureContract,
{
    type LCM = DataTensor<U::Out, I>;
    fn contract(&self, other: &DataTensor<T, I>) -> Result<Self::LCM, ContractionError> {
        match (self, other) {
            (DataTensor::Dense(s), DataTensor::Dense(o)) => Ok(DataTensor::Dense(s.contract(o)?)),
            (DataTensor::Dense(s), DataTensor::Sparse(o)) => Ok(DataTensor::Dense(s.contract(o)?)),
            (DataTensor::Sparse(s), DataTensor::Dense(o)) => Ok(DataTensor::Dense(s.contract(o)?)),
            (DataTensor::Sparse(s), DataTensor::Sparse(o)) => {
                Ok(DataTensor::Sparse(s.contract(o)?))
            }
        }
    }
}

impl<I> Contract<NumTensor<I>> for NumTensor<I>
where
    I: Clone + TensorStructure + StructureContract,
{
    type LCM = NumTensor<I>;
    fn contract(&self, other: &NumTensor<I>) -> Result<Self::LCM, ContractionError> {
        match (self, other) {
            (NumTensor::Float(a), NumTensor::Float(b)) => Ok(NumTensor::Float(a.contract(b)?)),
            (NumTensor::Float(a), NumTensor::Complex(b)) => Ok(NumTensor::Complex(a.contract(b)?)),
            (NumTensor::Complex(a), NumTensor::Float(b)) => Ok(NumTensor::Complex(a.contract(b)?)),
            (NumTensor::Complex(a), NumTensor::Complex(b)) => {
                Ok(NumTensor::Complex(a.contract(b)?))
            }
        }
    }
}
