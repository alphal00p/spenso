use std::{borrow::Cow, fmt::Display, marker::PhantomData, ops::Neg, sync::LazyLock};

use crate::{
    complex::{Complex, RealOrComplex, RealOrComplexTensor},
    data::{DataTensor, DenseTensor, SetTensorData, SparseTensor},
    structure::{
        abstract_index::AbstractIndex,
        concrete_index::ConcreteIndex,
        dimension::Dimension,
        representation::{LibraryRep, RepName, Representation, REPS},
        slot::{IsAbstractSlot, Slot, SlotError},
        HasName, HasStructure, IndexlessNamedStructure, NamedStructure, StructureError,
        TensorShell, TensorStructure, VecStructure,
    },
};
use ahash::AHashMap;
use anyhow::{anyhow, Result};

use thiserror::Error;

#[derive(Debug, Clone, Copy, Error)]
pub enum LibraryError<Key: Display> {
    #[error("Not found {0}")]
    NotFound(Key),
    #[error("Invalid key")]
    InvalidKey,
}

pub trait Library<S> {
    type Key: Display;
    type Value: Clone;
    // type Structure: TensorStructure;

    fn key_for_structure(&self, structure: S) -> Result<Self::Key, S>
    where
        S: TensorStructure;
    fn get<'a>(&'a self, key: &Self::Key) -> Result<Cow<'a, Self::Value>, LibraryError<Self::Key>>;
}

#[derive(Default, Clone)]
pub struct DummyLibrary<V, K = DummyKey> {
    key: PhantomData<K>,
    value: PhantomData<V>,
}

impl<V, K> DummyLibrary<V, K> {
    pub fn new() -> Self {
        DummyLibrary {
            key: PhantomData,
            value: PhantomData,
        }
    }
}

#[derive(Default, Clone, Copy, Debug)]
pub struct DummyKey {}

impl Display for DummyKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DUMMY")
    }
}

impl<K: Display + Clone, V: Clone, S> Library<S> for DummyLibrary<V, K> {
    type Key = K;
    type Value = DummyLibraryTensor<V>;
    fn get<'a>(&'a self, key: &Self::Key) -> Result<Cow<'a, Self::Value>, LibraryError<Self::Key>> {
        Err(LibraryError::NotFound(key.clone()))
    }

    fn key_for_structure(&self, structure: S) -> Result<Self::Key, S>
    where
        S: TensorStructure,
    {
        Err(structure)
    }
}

#[derive(Default, Clone, Copy, Debug)]
pub struct DummyLibraryTensor<T> {
    with_indices: PhantomData<T>,
}

pub enum DummyIter<T> {
    None,
    Some(T),
}

impl<T> Iterator for DummyIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

impl<T: TensorStructure> TensorStructure for DummyLibraryTensor<T> {
    type Slot = T::Slot;
    type Indexed = T::Indexed;
    fn reindex(self, indices: &[AbstractIndex]) -> Result<Self::Indexed, StructureError> {
        unimplemented!()
    }
    fn dual(self) -> Self {
        unimplemented!()
    }

    fn external_structure_iter(&self) -> impl Iterator<Item = Self::Slot> {
        DummyIter::<T::Slot>::None
    }
    fn external_dims_iter(&self) -> impl Iterator<Item = Dimension> {
        DummyIter::<Dimension>::None
    }
    fn external_reps_iter(
        &self,
    ) -> impl Iterator<Item = Representation<<Self::Slot as IsAbstractSlot>::R>> {
        DummyIter::<Representation<<T::Slot as IsAbstractSlot>::R>>::None
    }

    fn external_indices_iter(&self) -> impl Iterator<Item = AbstractIndex> {
        DummyIter::<AbstractIndex>::None
    }
    fn get_aind(&self, i: usize) -> Option<AbstractIndex> {
        unimplemented!()
    }
    fn get_rep(&self, i: usize) -> Option<Representation<<Self::Slot as IsAbstractSlot>::R>> {
        unimplemented!()
    }
    fn get_dim(&self, i: usize) -> Option<Dimension> {
        unimplemented!()
    }
    fn get_slot(&self, i: usize) -> Option<Self::Slot> {
        unimplemented!()
    }
    fn order(&self) -> usize {
        unimplemented!()
    }
}

impl<T: HasStructure + TensorStructure> HasStructure for DummyLibraryTensor<T> {
    type Store<S>
        = T::Store<S>
    where
        S: TensorStructure;

    type Scalar = T::Scalar;
    type ScalarRef<'a>
    where
        Self: 'a,
    = T::ScalarRef<'a>;
    type Structure = T::Structure;

    fn map_structure<O: TensorStructure>(self, f: impl Fn(Self::Structure) -> O) -> Self::Store<O> {
        unimplemented!()
    }

    fn map_structure_result<O: TensorStructure, Er>(
        self,
        f: impl Fn(Self::Structure) -> Result<O, Er>,
    ) -> std::result::Result<Self::Store<O>, Er> {
        unimplemented!()
    }

    fn structure(&self) -> &Self::Structure {
        unimplemented!()
    }

    fn mut_structure(&mut self) -> &mut Self::Structure {
        unimplemented!()
    }

    fn scalar(self) -> Option<Self::Scalar> {
        unimplemented!()
    }

    fn scalar_ref(&self) -> Option<Self::ScalarRef<'_>> {
        unimplemented!()
    }

    fn map_same_structure(self, f: impl FnOnce(Self::Structure) -> Self::Structure) -> Self {
        unimplemented!()
    }
}

impl<T: SetTensorData> SetTensorData for DummyLibraryTensor<T> {
    type SetData = T::SetData;

    fn set(&mut self, indices: &[ConcreteIndex], value: Self::SetData) -> Result<()> {
        unimplemented!()
    }

    fn set_flat(
        &mut self,
        index: crate::structure::concrete_index::FlatIndex,
        value: Self::SetData,
    ) -> Result<()> {
        unimplemented!()
    }
}

impl<T: HasStructure + TensorStructure> LibraryTensor for DummyLibraryTensor<T> {
    type WithIndices = T;
    type Data = ();

    fn empty(key: Self::Structure) -> Self {
        unimplemented!()
    }

    fn from_dense(key: Self::Structure, data: Vec<Self::Data>) -> Result<Self> {
        unimplemented!()
    }

    fn from_sparse(
        key: Self::Structure,
        data: impl IntoIterator<Item = (Vec<ConcreteIndex>, Self::Data)>,
    ) -> Result<Self> {
        unimplemented!()
    }

    fn with_indices(&self, indices: &[AbstractIndex]) -> Result<Self::WithIndices, StructureError> {
        unimplemented!()
    }
}

pub trait LibraryTensor: HasStructure + Sized + TensorStructure {
    type WithIndices: HasStructure;
    type Data;
    fn empty(key: Self::Structure) -> Self;

    fn from_dense(key: Self::Structure, data: Vec<Self::Data>) -> Result<Self>;

    fn from_sparse(
        key: Self::Structure,
        data: impl IntoIterator<Item = (Vec<ConcreteIndex>, Self::Data)>,
    ) -> Result<Self>;

    fn with_indices(&self, indices: &[AbstractIndex]) -> Result<Self::WithIndices, StructureError>;
}

impl<D: Clone, S: TensorStructure + Clone> LibraryTensor for DataTensor<D, S> {
    type WithIndices = DataTensor<D, S::Indexed>;
    type Data = D;

    fn empty(key: S) -> Self {
        DataTensor::Sparse(SparseTensor::empty(key))
    }

    fn from_dense(key: S, data: Vec<Self::Data>) -> Result<Self> {
        Ok(DataTensor::Dense(DenseTensor::from_data(data, key)?))
    }

    fn from_sparse(
        key: S,
        data: impl IntoIterator<Item = (Vec<ConcreteIndex>, Self::Data)>,
    ) -> Result<Self> {
        Ok(DataTensor::Sparse(SparseTensor::from_data(data, key)?))
    }

    fn with_indices(&self, indices: &[AbstractIndex]) -> Result<Self::WithIndices, StructureError> {
        let new_structure = self.structure().clone().reindex(indices)?;

        Ok(match self {
            DataTensor::Dense(d) => DataTensor::Dense(DenseTensor {
                data: d.data.clone(),
                structure: new_structure,
            }),
            DataTensor::Sparse(s) => DataTensor::Sparse(SparseTensor {
                elements: s.elements.clone(),
                structure: new_structure,
            }),
        })
    }
}

impl<D: Clone + Default, S: TensorStructure + Clone> LibraryTensor for RealOrComplexTensor<D, S> {
    type WithIndices = RealOrComplexTensor<D, S::Indexed>;
    type Data = RealOrComplex<D>;

    fn empty(key: S) -> Self {
        RealOrComplexTensor::Real(DataTensor::Sparse(SparseTensor::empty(key)))
    }

    fn from_dense(key: S, data: Vec<Self::Data>) -> Result<Self> {
        let complex_data = data.into_iter().map(|a| a.to_complex()).collect();
        let complex_tensor =
            <DataTensor<Complex<D>, S> as LibraryTensor>::from_dense(key, complex_data)?;

        Ok(RealOrComplexTensor::Complex(complex_tensor))
    }

    fn from_sparse(
        key: S,
        data: impl IntoIterator<Item = (Vec<ConcreteIndex>, Self::Data)>,
    ) -> Result<Self> {
        let complex_tensor = <DataTensor<Complex<D>, S> as LibraryTensor>::from_sparse(
            key,
            data.into_iter()
                .map(|(indices, value)| (indices, value.to_complex())),
        )?;

        Ok(RealOrComplexTensor::Complex(complex_tensor))
    }

    fn with_indices(&self, indices: &[AbstractIndex]) -> Result<Self::WithIndices, StructureError> {
        match self {
            RealOrComplexTensor::Real(real_tensor) => {
                let new_real_tensor =
                    <DataTensor<D, S> as LibraryTensor>::with_indices(&real_tensor, indices)?;
                Ok(RealOrComplexTensor::Real(new_real_tensor))
            }
            RealOrComplexTensor::Complex(complex_tensor) => {
                let new_complex_tensor =
                    <DataTensor<Complex<D>, S> as LibraryTensor>::with_indices(
                        &complex_tensor,
                        indices,
                    )?;
                Ok(RealOrComplexTensor::Complex(new_complex_tensor))
            }
        }
    }
}

#[cfg(feature = "shadowing")]
pub mod symbolic {
    use super::*;
    use symbolica::{
        atom::{representation::FunView, Atom, AtomView, Symbol},
        symbol,
    };

    use crate::{
        parametric::{ConcreteOrParam, MixedTensor, ParamOrConcrete, ParamTensor},
        structure::abstract_index::AIND_SYMBOLS,
        symbolica_utils::{IntoArgs, IntoSymbol},
    };

    pub type ExplicitKey = IndexlessNamedStructure<Symbol, Vec<Atom>, LibraryRep>;

    impl ExplicitKey {
        pub fn from_structure<S: TensorStructure + HasName<Name: IntoSymbol, Args: IntoArgs>>(
            structure: &S,
        ) -> Option<Self> {
            Some(IndexlessNamedStructure::from_iter(
                structure.reps().into_iter().map(|r| r.to_lib()),
                structure.name()?.ref_into_symbol(),
                structure.args().map(|a| a.args()),
            ))
        }
    }

    // pub struct DataStoreRefTensor<'a, T, S> {
    //     data: &'a T,
    //     structure: S,
    // }

    pub type ShadowedStructure = NamedStructure<Symbol, Vec<Atom>, LibraryRep>;

    #[cfg(feature = "shadowing")]
    impl<'a> TryFrom<FunView<'a>> for ShadowedStructure {
        type Error = StructureError;
        fn try_from(value: FunView<'a>) -> Result<Self, Self::Error> {
            match value.get_symbol() {
                s if s == AIND_SYMBOLS.aind => {
                    let mut structure: Vec<Slot<LibraryRep>> = vec![];

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
                        let slot: Result<Slot<LibraryRep>, _> = arg.try_into();

                        match slot {
                            Ok(slot) => {
                                is_structure = None;
                                structure.structure.push(slot);
                            }
                            Err(e) => {
                                if let AtomView::Fun(f) = arg {
                                    if f.get_symbol() == AIND_SYMBOLS.aind {
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

    impl<S: TensorStructure + Clone> LibraryTensor for ParamTensor<S> {
        type WithIndices = ParamTensor<S::Indexed>;
        type Data = Atom;
        fn empty(key: S) -> Self {
            ParamTensor::composite(<DataTensor<Atom, S> as LibraryTensor>::empty(key))
        }

        fn from_dense(key: S, data: Vec<Self::Data>) -> Result<Self> {
            Ok(ParamTensor::composite(
                <DataTensor<Atom, S> as LibraryTensor>::from_dense(key, data)?,
            ))
        }

        fn from_sparse(
            key: S,
            data: impl IntoIterator<Item = (Vec<ConcreteIndex>, Self::Data)>,
        ) -> Result<Self> {
            Ok(ParamTensor::composite(
                <DataTensor<Atom, S> as LibraryTensor>::from_sparse(key, data)?,
            ))
        }

        fn with_indices(
            &self,
            indices: &[AbstractIndex],
        ) -> Result<Self::WithIndices, StructureError> {
            let new_tensor =
                <DataTensor<Atom, S> as LibraryTensor>::with_indices(&self.tensor, indices)?;
            Ok(ParamTensor {
                tensor: new_tensor,
                param_type: self.param_type,
            })
        }
    }

    impl<D: Default + Clone, S: TensorStructure + Clone> LibraryTensor for MixedTensor<D, S> {
        type WithIndices = MixedTensor<D, S::Indexed>;
        type Data = ConcreteOrParam<RealOrComplex<D>>;

        fn empty(key: S) -> Self {
            MixedTensor::Concrete(<RealOrComplexTensor<D, S> as LibraryTensor>::empty(key))
        }

        fn from_dense(key: S, data: Vec<Self::Data>) -> Result<Self> {
            let data: Result<Vec<_>> = data
                .into_iter()
                .map(|v| match v {
                    ConcreteOrParam::Concrete(c) => Ok(c),
                    ConcreteOrParam::Param(p) => {
                        return Err(anyhow!("Only concrete data allowed, not {p}"))
                    }
                })
                .collect();

            Ok(MixedTensor::Concrete(
                <RealOrComplexTensor<D, S> as LibraryTensor>::from_dense(key, data?)?,
            ))
        }

        fn from_sparse(
            key: S,
            data: impl IntoIterator<Item = (Vec<ConcreteIndex>, Self::Data)>,
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

            Ok(MixedTensor::Concrete(
                <RealOrComplexTensor<D, S> as LibraryTensor>::from_sparse(key, data?)?,
            ))
        }

        fn with_indices(
            &self,
            indices: &[AbstractIndex],
        ) -> Result<Self::WithIndices, StructureError> {
            Ok(match self {
                ParamOrConcrete::Concrete(c) => {
                    ParamOrConcrete::Concrete(
                        <RealOrComplexTensor<D, S> as LibraryTensor>::with_indices(c, indices)?,
                    )
                }
                ParamOrConcrete::Param(p) => ParamOrConcrete::Param(
                    <ParamTensor<S> as LibraryTensor>::with_indices(p, indices)?,
                ),
            })
        }
    }

    #[derive(Clone, PartialEq, Eq, Debug, Hash)]
    pub struct GenericKey {
        global_name: Symbol,
        args: Option<Vec<Atom>>,
        reps: Vec<LibraryRep>,
    }

    impl GenericKey {
        pub fn new(global_name: Symbol, args: Option<Vec<Atom>>, reps: Vec<LibraryRep>) -> Self {
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

        pub fn merge(&mut self, other: &mut Self) {
            self.explicit_dimension
                .extend(other.explicit_dimension.drain());
            self.generic_dimension
                .extend(other.generic_dimension.drain());
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

    impl<
            T: HasStructure<Structure = ExplicitKey> + Clone,
            S: TensorStructure + HasName<Name: IntoSymbol, Args: IntoArgs>,
        > Library<S> for TensorLibrary<T>
    {
        type Key = ExplicitKey;
        type Value = T;
        // type Structure = ExplicitKey;

        fn get<'a>(
            &'a self,
            key: &Self::Key,
        ) -> Result<Cow<'a, Self::Value>, LibraryError<Self::Key>> {
            if let Some(tensor) = self.explicit_dimension.get(key) {
                Ok(Cow::Borrowed(tensor))
            } else if let Some(builder) = self.generic_dimension.get(&key.clone().into()) {
                Ok(Cow::Owned(builder(key.clone())))
            } else {
                Err(LibraryError::NotFound(key.clone()))
            }
        }

        fn key_for_structure(&self, structure: S) -> Result<Self::Key, S>
        where
            S: TensorStructure,
        {
            let a = ExplicitKey::from_structure(&structure);
            if let Some(key) = a {
                if <TensorLibrary<T> as Library<S>>::get(&self, &key).is_ok() {
                    Ok(key)
                } else {
                    Err(structure)
                }
            } else {
                Err(structure)
            }
        }
    }

    impl<T: HasStructure<Structure = ExplicitKey> + SetTensorData + Clone + LibraryTensor>
        TensorLibrary<T>
    {
        pub fn metric_key(rep: LibraryRep) -> ExplicitKey {
            ExplicitKey::from_iter(
                [rep.new_rep(4), rep.new_rep(4)],
                symbol!(Self::METRIC_NAME),
                None,
            )
        }

        pub fn generic_mink_metric(key: ExplicitKey) -> T
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
                    let id_metric = GenericKey::new(ETS.metric, None, vec![*rep, rep.dual()]);
                    self.insert_generic(id_metric, Self::checked_identity);
                    self.insert_generic(Self::id(rep.dual()), Self::checked_identity);
                    let id_metric = GenericKey::new(ETS.metric, None, vec![rep.dual(), *rep]);
                    self.insert_generic(id_metric, Self::checked_identity);
                }
            }
        }

        pub fn id(rep: LibraryRep) -> GenericKey {
            GenericKey::new(ETS.id, None, vec![rep, rep.dual()])
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

        pub fn insert_explicit_dense(
            &mut self,
            key: ExplicitKey,
            data: Vec<T::Data>,
        ) -> Result<()> {
            let tensor = T::from_dense(key.clone(), data)?;
            self.explicit_dimension.insert(key, tensor);
            Ok(())
        }

        pub fn insert_explicit_sparse(
            &mut self,
            key: ExplicitKey,
            data: impl IntoIterator<Item = (Vec<ConcreteIndex>, T::Data)>,
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
                Err(LibraryError::NotFound(key.clone()).into())
            }
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
#[cfg(feature = "shadowing")]
impl_tensor_library_data!(
    symbolica::atom::Atom,
    symbolica::atom::Atom::Zero,
    symbolica::atom::Atom::new_num(1)
);

impl<T: TensorLibraryData> TensorLibraryData for RealOrComplex<T> {
    fn zero() -> Self {
        RealOrComplex::Real(T::zero())
    }

    fn one() -> Self {
        RealOrComplex::Real(T::one())
    }
}

#[cfg(test)]
mod test {}
