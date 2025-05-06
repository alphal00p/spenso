use std::sync::LazyLock;

use super::*;
use ahash::AHashMap;
use symbolica::{
    atom::{Atom, Symbol},
    symbol,
};

use anyhow::anyhow;

use crate::{
    parametric::{ConcreteOrParam, MixedTensor, ParamOrConcrete, ParamTensor},
    structure::{
        representation::{LibraryRep, RepName, REPS},
        HasName, IndexlessNamedStructure, TensorShell,
    },
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

    fn with_indices(&self, indices: &[AbstractIndex]) -> Result<Self::WithIndices, StructureError> {
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

    fn with_indices(&self, indices: &[AbstractIndex]) -> Result<Self::WithIndices, StructureError> {
        Ok(match self {
            ParamOrConcrete::Concrete(c) => ParamOrConcrete::Concrete(
                <RealOrComplexTensor<D, S> as LibraryTensor>::with_indices(c, indices)?,
            ),
            ParamOrConcrete::Param(p) => {
                ParamOrConcrete::Param(<ParamTensor<S> as LibraryTensor>::with_indices(p, indices)?)
            }
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

    fn get<'a>(&'a self, key: &Self::Key) -> Result<Cow<'a, Self::Value>, LibraryError<Self::Key>> {
        if let Some(tensor) = self.explicit_dimension.get(key) {
            Ok(Cow::Borrowed(tensor))
        } else if let Some(builder) = self.generic_dimension.get(&key.clone().into()) {
            Ok(Cow::Owned(builder(key.clone())))
        } else {
            Err(LibraryError::NotFound(key.clone()))
        }
    }

    fn key_for_structure(&self, structure: &S) -> Result<Self::Key, LibraryError<Self::Key>>
    where
        S: TensorStructure,
    {
        let a = ExplicitKey::from_structure(structure);
        if let Some(key) = a {
            if <TensorLibrary<T> as Library<S>>::get(&self, &key).is_ok() {
                Ok(key)
            } else {
                Err(LibraryError::InvalidKey)
            }
        } else {
            Err(LibraryError::InvalidKey)
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

    pub fn insert_explicit_dense(&mut self, key: ExplicitKey, data: Vec<T::Data>) -> Result<()> {
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

#[cfg(test)]
mod test {
    use crate::{
        complex::RealOrComplexRef,
        network::{
            parsing::ShadowedStructure, store::NetworkStore, Network, Sequential, SmallestDegree,
            TensorOrScalarOrKey,
        },
        parametric::ConcreteOrParamRef,
        shadowing::Concretize,
        structure::{
            representation::{Euclidean, Minkowski},
            ToSymbolic,
        },
    };

    use super::*;

    #[test]
    fn add_to_lib() {
        let mut lib = TensorLibrary::<MixedTensor<f64, ExplicitKey>>::new();
        let key = ExplicitKey::from_iter(
            [
                LibraryRep::from(Minkowski {}).new_rep(4),
                Euclidean {}.new_rep(4).cast(),
                Euclidean {}.new_rep(4).cast(),
            ],
            symbol!("gamma"),
            None,
        );

        let one = ConcreteOrParam::Concrete(RealOrComplex::Real(1.));
        lib.insert_explicit_sparse(key.clone(), [(vec![0, 0, 1], one)])
            .unwrap();

        lib.get(&key).unwrap();
        let indexed = key
            .clone()
            .to_indexed(&[0.into(), 1.into(), 2.into()])
            .unwrap();
        let expr = indexed.to_symbolic().unwrap();
        let mut net = Network::<
            NetworkStore<MixedTensor<f64, ShadowedStructure>, ConcreteOrParam<RealOrComplex<f64>>>,
            _,
        >::try_from_view(expr.as_view(), &lib)
        .unwrap();

        println!(
            "{}",
            net.dot_display_impl::<_, ExplicitKey>(
                &lib,
                |a| a.to_string(),
                |_| "".to_string(),
                |a| a.to_string()
            )
        );

        net.execute::<Sequential, SmallestDegree, _>(&lib).unwrap();
        println!(
            "{}",
            net.dot_display_impl::<_, ExplicitKey>(
                &lib,
                |a| a.to_string(),
                |_| "".to_string(),
                |a| a.to_string()
            )
        );

        if let TensorOrScalarOrKey::Key {
            key: res_key,
            graph_slots,
        } = net.result().unwrap()
        {
            // println!("YaY:{a}");
            assert_eq!(graph_slots, indexed.structure.structure);
            assert_eq!(&key, res_key);
        } else {
            panic!("Not Key")
        }
    }

    #[test]
    fn not_in_lib() {
        let lib = TensorLibrary::<MixedTensor<f64, ExplicitKey>>::new();
        let key = ExplicitKey::from_iter(
            [
                LibraryRep::from(Minkowski {}).new_rep(4),
                Euclidean {}.new_rep(4).cast(),
                Euclidean {}.new_rep(4).cast(),
            ],
            symbol!("gamma"),
            None,
        );

        let indexed = key
            .clone()
            .to_indexed(&[0.into(), 1.into(), 2.into()])
            .unwrap();
        let expr = indexed.to_symbolic().unwrap();
        let mut net = Network::<
            NetworkStore<MixedTensor<f64, ShadowedStructure>, ConcreteOrParam<RealOrComplex<f64>>>,
            _,
        >::try_from_view(expr.as_view(), &lib)
        .unwrap();

        println!(
            "{}",
            net.dot_display_impl::<_, ExplicitKey>(
                &lib,
                |a| a.to_string(),
                |_| "".to_string(),
                |a| a.name().unwrap().to_string()
            )
        );

        net.execute::<Sequential, SmallestDegree, _>(&lib).unwrap();
        println!(
            "{}",
            net.dot_display_impl::<_, ExplicitKey>(
                &lib,
                |a| a.to_string(),
                |_| "".to_string(),
                |a| a.name().unwrap().to_string()
            )
        );

        if let TensorOrScalarOrKey::Tensor { tensor, .. } = net.result().unwrap() {
            // println!("YaY:{a}");

            assert_eq!(tensor, &indexed.to_shell().concretize());
        } else {
            panic!("Not Key")
        }
    }
}
