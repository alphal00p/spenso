use std::{borrow::Cow, ops::Neg, sync::LazyLock};

use crate::{
    complex::{Complex, RealOrComplex, RealOrComplexTensor},
    data::{DataTensor, DenseTensor, SetTensorData, SparseTensor, StorageTensor},
    parametric::{ConcreteOrParam, MixedTensor, ParamOrConcrete, ParamTensor},
    scalar::Scalar,
    structure::{
        abstract_index::{AbstractIndex, ABSTRACTIND},
        concrete_index::ConcreteIndex,
        representation::{Rep, RepName, REPS},
        slot::{IsAbstractSlot, Slot, SlotError},
        AtomStructure, HasName, HasStructure, IndexlessNamedStructure, NamedStructure,
        StructureError, TensorShell, TensorStructure, VecStructure,
    },
    symbolica_utils::{IntoArgs, IntoSymbol},
};
use ahash::AHashMap;
use anyhow::{anyhow, Result};
use symbolica::{
    atom::{representation::FunView, Atom, AtomView, Symbol},
    symbol,
};
use thiserror::Error;

pub type ExplicitKey = IndexlessNamedStructure<Symbol, Vec<Atom>, Rep>;

impl ExplicitKey {
    pub fn from_structure<S: TensorStructure + HasName<Name: IntoSymbol, Args: IntoArgs>>(
        structure: S,
    ) -> Option<Self>
    where
        Rep: From<<S::Slot as IsAbstractSlot>::R>,
    {
        Some(IndexlessNamedStructure::from_iter(
            structure.reps().into_iter().map(|r| r.cast()),
            structure.name()?.ref_into_symbol(),
            structure.args().map(|a| a.args()),
        ))
    }
}

pub trait LibraryTensor: SetTensorData + HasStructure<Structure = ExplicitKey> + Sized {
    type WithIndices: HasStructure;
    fn empty(key: ExplicitKey) -> Self;

    fn from_dense(key: ExplicitKey, data: Vec<Self::SetData>) -> Result<Self>;

    fn from_sparse(
        key: ExplicitKey,
        data: impl IntoIterator<Item = (Vec<ConcreteIndex>, Self::SetData)>,
    ) -> Result<Self>;

    fn with_indices(&self, indices: &[AbstractIndex]) -> Self::WithIndices;
}

pub struct DataStoreRefTensor<'a, T, S> {
    data: &'a T,
    structure: S,
}

impl<D: Clone> LibraryTensor for DataTensor<D, ExplicitKey> {
    type WithIndices = DataTensor<D, ShadowedStructure>;

    fn empty(key: ExplicitKey) -> Self {
        DataTensor::Sparse(SparseTensor::empty(key))
    }

    fn from_dense(key: ExplicitKey, data: Vec<Self::SetData>) -> Result<Self> {
        Ok(DataTensor::Dense(DenseTensor::from_data(data, key)?))
    }

    fn from_sparse(
        key: ExplicitKey,
        data: impl IntoIterator<Item = (Vec<ConcreteIndex>, Self::SetData)>,
    ) -> Result<Self> {
        Ok(DataTensor::Sparse(SparseTensor::from_data(data, key)?))
    }

    fn with_indices(&self, indices: &[AbstractIndex]) -> Self::WithIndices {
        let new_structure = self.structure().clone().to_indexed(indices);

        match self {
            DataTensor::Dense(d) => DataTensor::Dense(DenseTensor {
                data: d.data.clone(),
                structure: new_structure,
            }),
            DataTensor::Sparse(s) => DataTensor::Sparse(SparseTensor {
                elements: s.elements.clone(),
                structure: new_structure,
            }),
        }
    }
}

pub type ShadowedStructure = NamedStructure<Symbol, Vec<Atom>, Rep>;

#[cfg(feature = "shadowing")]
impl<'a> TryFrom<FunView<'a>> for ShadowedStructure {
    type Error = StructureError;
    fn try_from(value: FunView<'a>) -> Result<Self, Self::Error> {
        match value.get_symbol() {
            s if s == symbol!(ABSTRACTIND) => {
                let mut structure: Vec<Slot<Rep>> = vec![];

                for arg in value.iter() {
                    structure.push(arg.try_into()?);
                }

                Ok(VecStructure::from(structure).into())
            }
            name => {
                let mut structure: ShadowedStructure = VecStructure::default().into();
                structure.set_name(name.into());
                let mut args = vec![];
                let mut is_structure = Some(SlotError::EmptyStructure);

                for arg in value.iter() {
                    let slot: Result<Slot<Rep>, _> = arg.try_into();

                    match slot {
                        Ok(slot) => {
                            is_structure = None;
                            structure.structure.push(slot);
                        }
                        Err(e) => {
                            if let AtomView::Fun(f) = arg {
                                if f.get_symbol() == symbol!(ABSTRACTIND) {
                                    let internal_s = Self::try_from(f);

                                    if let Ok(s) = internal_s {
                                        structure.extend(s);
                                        is_structure = None;
                                        continue;
                                    }
                                }
                            }
                            is_structure = Some(e);
                            args.push(arg.to_owned().into());
                        }
                    }
                }

                if !args.is_empty() {
                    structure.additional_args = Some(args);
                }
                if let Some(e) = is_structure {
                    Err(StructureError::EmptyStructure(e))
                } else {
                    Ok(structure)
                }
            }
        }
    }
}

impl<D: Clone + Default> LibraryTensor for RealOrComplexTensor<D, ExplicitKey> {
    type WithIndices = RealOrComplexTensor<D, ShadowedStructure>;

    fn empty(key: ExplicitKey) -> Self {
        RealOrComplexTensor::Real(DataTensor::Sparse(SparseTensor::empty(key)))
    }

    fn from_dense(key: ExplicitKey, data: Vec<Self::SetData>) -> Result<Self> {
        let complex_data = data.into_iter().map(|a| a.to_complex()).collect();
        let complex_tensor = DataTensor::<Complex<D>, ExplicitKey>::from_dense(key, complex_data)?;

        Ok(RealOrComplexTensor::Complex(complex_tensor))
    }

    fn from_sparse(
        key: ExplicitKey,
        data: impl IntoIterator<Item = (Vec<ConcreteIndex>, Self::SetData)>,
    ) -> Result<Self> {
        let complex_tensor = DataTensor::<Complex<D>, ExplicitKey>::from_sparse(
            key,
            data.into_iter()
                .map(|(indices, value)| (indices, value.to_complex())),
        )?;

        Ok(RealOrComplexTensor::Complex(complex_tensor))
    }

    fn with_indices(&self, indices: &[AbstractIndex]) -> Self::WithIndices {
        match self {
            RealOrComplexTensor::Real(real_tensor) => {
                let new_real_tensor = real_tensor.with_indices(indices);
                RealOrComplexTensor::Real(new_real_tensor)
            }
            RealOrComplexTensor::Complex(complex_tensor) => {
                let new_complex_tensor = complex_tensor.with_indices(indices);
                RealOrComplexTensor::Complex(new_complex_tensor)
            }
        }
    }
}

impl LibraryTensor for ParamTensor<ExplicitKey> {
    type WithIndices = ParamTensor<ShadowedStructure>;
    fn empty(key: ExplicitKey) -> Self {
        ParamTensor::composite(DataTensor::empty(key))
    }

    fn from_dense(key: ExplicitKey, data: Vec<Self::SetData>) -> Result<Self> {
        Ok(ParamTensor::composite(DataTensor::from_dense(key, data)?))
    }

    fn from_sparse(
        key: ExplicitKey,
        data: impl IntoIterator<Item = (Vec<ConcreteIndex>, Self::SetData)>,
    ) -> Result<Self> {
        Ok(ParamTensor::composite(DataTensor::from_sparse(key, data)?))
    }

    fn with_indices(&self, indices: &[AbstractIndex]) -> Self::WithIndices {
        let new_tensor = self.tensor.with_indices(indices);
        ParamTensor {
            tensor: new_tensor,
            param_type: self.param_type,
        }
    }
}

impl<D: Default + Clone> LibraryTensor for MixedTensor<D, ExplicitKey> {
    type WithIndices = MixedTensor<D, ShadowedStructure>;

    fn empty(key: ExplicitKey) -> Self {
        MixedTensor::Concrete(RealOrComplexTensor::empty(key))
    }

    fn from_dense(key: ExplicitKey, data: Vec<Self::SetData>) -> Result<Self> {
        let data: Result<Vec<_>> = data
            .into_iter()
            .map(|v| match v {
                ConcreteOrParam::Concrete(c) => Ok(c),
                ConcreteOrParam::Param(p) => {
                    return Err(anyhow!("Only concrete data allowed, not {p}"))
                }
            })
            .collect();

        Ok(MixedTensor::Concrete(RealOrComplexTensor::from_dense(
            key, data?,
        )?))
    }

    fn from_sparse(
        key: ExplicitKey,
        data: impl IntoIterator<Item = (Vec<ConcreteIndex>, Self::SetData)>,
    ) -> Result<Self> {
        let data: Result<Vec<_>> = data
            .into_iter()
            .map(|(i, v)| match v {
                ConcreteOrParam::Concrete(c) => Ok((i, c)),
                ConcreteOrParam::Param(p) => {
                    return Err(anyhow!("Only concrete data allowed, not {p}"))
                }
            })
            .collect();

        Ok(MixedTensor::Concrete(RealOrComplexTensor::from_sparse(
            key, data?,
        )?))
    }

    fn with_indices(&self, indices: &[AbstractIndex]) -> Self::WithIndices {
        match self {
            ParamOrConcrete::Concrete(c) => ParamOrConcrete::Concrete(c.with_indices(indices)),
            ParamOrConcrete::Param(p) => ParamOrConcrete::Param(p.with_indices(indices)),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct GenericKey {
    global_name: Symbol,
    args: Option<Vec<Atom>>,
    reps: Vec<Rep>,
}

impl GenericKey {
    pub fn new(global_name: Symbol, args: Option<Vec<Atom>>, reps: Vec<Rep>) -> Self {
        Self {
            global_name,
            args,
            reps,
        }
    }
}

impl From<ExplicitKey> for GenericKey {
    fn from(key: ExplicitKey) -> Self {
        Self {
            global_name: key.global_name.unwrap(),
            reps: key.reps().into_iter().map(|r| r.rep).collect(),
            args: key.additional_args,
        }
    }
}

#[allow(clippy::type_complexity)]
pub struct TensorLibrary<T: HasStructure<Structure = ExplicitKey>> {
    explicit_dimension: AHashMap<ExplicitKey, T>,
    generic_dimension: AHashMap<GenericKey, fn(ExplicitKey) -> T>,
}

#[derive(Debug, Error)]
pub enum LibraryError {
    #[error("The key {0} is not present")]
    KeyNotFound(ExplicitKey),
}

pub struct ExplicitTensorSymbols {
    pub id: Symbol,
    pub metric: Symbol,
}

pub static ETS: LazyLock<ExplicitTensorSymbols> = LazyLock::new(|| ExplicitTensorSymbols {
    id: symbol!(TensorLibrary::<TensorShell<ExplicitKey>>::ID_NAME;Symmetric).unwrap(),
    metric: symbol!(TensorLibrary::<TensorShell<ExplicitKey>>::METRIC_NAME;Symmetric).unwrap(),
});

impl<T: HasStructure<Structure = ExplicitKey>> TensorLibrary<T> {
    pub const ID_NAME: &'static str = "𝟙";
    pub const METRIC_NAME: &'static str = "g";

    pub fn new() -> Self {
        Self {
            explicit_dimension: AHashMap::new(),
            generic_dimension: AHashMap::new(),
        }
    }
}

pub trait TensorLibraryData: Neg<Output = Self> {
    fn one() -> Self;
    fn zero() -> Self;
}
macro_rules! impl_tensor_library_data {
    ( $target_type:ty, $zero_val:expr, $one_val:expr ) => {
        // Assuming TensorLibraryData is defined in spenso::tensor_library
        // If the macro is defined *inside* tensor_library.rs, `TensorLibraryData` might suffice.
        // Using `$crate::spenso::tensor_library::TensorLibraryData` is more robust
        // if the macro can be called from other modules within the spenso crate.
        // Adjust path as needed based on your crate structure.
        impl TensorLibraryData for $target_type {
            /// Returns the additive identity element (zero) of this type.
            #[inline]
            fn zero() -> Self {
                $zero_val
            }

            /// Returns the multiplicative identity element (one) of this type.
            #[inline]
            fn one() -> Self {
                $one_val
            }
        }
    };
}

impl_tensor_library_data!(f32, 0.0, 1.0);
impl_tensor_library_data!(f64, 0.0, 1.0);
impl_tensor_library_data!(i32, 0, 1);
impl_tensor_library_data!(i64, 0, 1);
impl_tensor_library_data!(Atom, Atom::Zero, Atom::new_num(1));
impl<T: TensorLibraryData> TensorLibraryData for RealOrComplex<T> {
    fn zero() -> Self {
        RealOrComplex::Real(T::zero())
    }

    fn one() -> Self {
        RealOrComplex::Real(T::one())
    }
}

impl<T: TensorLibraryData> TensorLibraryData for ConcreteOrParam<T> {
    fn one() -> Self {
        ConcreteOrParam::Concrete(T::one())
    }

    fn zero() -> Self {
        ConcreteOrParam::Concrete(T::zero())
    }
}

impl<T: LibraryTensor> TensorLibrary<T> {
    fn metric_key(rep: Rep) -> ExplicitKey {
        ExplicitKey::from_iter(
            [rep.new_rep(4), rep.new_rep(4)],
            symbol!(Self::METRIC_NAME),
            None,
        )
    }

    fn generic_mink_metric(key: ExplicitKey) -> T
    where
        T::SetData: TensorLibraryData,
    {
        let dim: usize = key.get_dim(0).unwrap().try_into().unwrap();
        let mut tensor = T::empty(key);

        for i in 0..dim {
            if i > 0 {
                tensor.set(&[i, i], -T::SetData::one()).unwrap();
            } else {
                tensor.set(&[i, i], T::SetData::one()).unwrap();
            }
        }
        tensor.into()
    }

    pub fn update_ids(&mut self)
    where
        T::SetData: TensorLibraryData,
    {
        for rep in REPS.read().unwrap().reps() {
            self.insert_generic(Self::id(*rep), Self::checked_identity);
            if rep.dual() != *rep {
                let id_metric =
                    GenericKey::new(symbol!(Self::METRIC_NAME), None, vec![*rep, rep.dual()]);
                self.insert_generic(id_metric, Self::checked_identity);
                self.insert_generic(Self::id(rep.dual()), Self::checked_identity);
                let id_metric =
                    GenericKey::new(symbol!(Self::METRIC_NAME), None, vec![rep.dual(), *rep]);
                self.insert_generic(id_metric, Self::checked_identity);
            }
        }
    }

    pub fn id(rep: Rep) -> GenericKey {
        GenericKey::new(symbol!(Self::ID_NAME), None, vec![rep, rep.dual()])
    }

    pub fn checked_identity(key: ExplicitKey) -> T
    where
        T::SetData: TensorLibraryData,
    {
        debug_assert!(key.order() == 2);
        debug_assert!(key.get_rep(0).map(|r| r.dual()) == key.get_rep(1));

        Self::identity(key)
    }

    pub fn identity(key: ExplicitKey) -> T
    where
        T::SetData: TensorLibraryData,
    {
        let dim: usize = key.get_dim(0).unwrap().try_into().unwrap();
        let mut tensor = T::empty(key);

        for i in 0..dim {
            tensor.set(&[i, i], T::SetData::one()).unwrap();
        }
        tensor.into()
    }

    pub fn insert_explicit(&mut self, data: T) {
        let key = data.structure().clone();
        self.explicit_dimension.insert(key, data);
    }

    pub fn insert_explicit_dense(&mut self, key: ExplicitKey, data: Vec<T::SetData>) -> Result<()> {
        let tensor = T::from_dense(key.clone(), data)?;
        self.explicit_dimension.insert(key, tensor);
        Ok(())
    }

    pub fn insert_explicit_sparse(
        &mut self,
        key: ExplicitKey,
        data: impl IntoIterator<Item = (Vec<ConcreteIndex>, T::SetData)>,
    ) -> Result<()> {
        let tensor = T::from_sparse(key.clone(), data)?;
        self.explicit_dimension.insert(key, tensor);
        Ok(())
    }

    pub fn insert_generic(&mut self, key: GenericKey, data: fn(ExplicitKey) -> T) {
        self.generic_dimension.insert(key, data);
    }

    pub fn get(&self, key: &ExplicitKey) -> Result<Cow<T>>
    where
        T: Clone,
    {
        if let Some(tensor) = self.explicit_dimension.get(key) {
            Ok(Cow::Borrowed(tensor))
        } else if let Some(builder) = self.generic_dimension.get(&key.clone().into()) {
            Ok(Cow::Owned(builder(key.clone())))
        } else {
            Err(LibraryError::KeyNotFound(key.clone()).into())
        }
    }
}
#[cfg(test)]
mod test {}
