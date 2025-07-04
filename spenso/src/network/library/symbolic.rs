use std::sync::LazyLock;

use super::*;
use ahash::AHashMap;
use linnet::permutation::Permutation;
use symbolica::{
    atom::{Atom, Symbol},
    symbol,
};

use anyhow::anyhow;

use crate::{
    shadowing::symbolica_utils::{IntoArgs, IntoSymbol},
    structure::{
        named::{IdentityName, METRIC_NAME},
        permuted::{Perm, PermuteTensor},
        representation::{initialize, LibraryRep, RepName},
        HasName, IndexlessNamedStructure,
    },
    tensors::parametric::{ConcreteOrParam, MixedTensor, ParamOrConcrete, ParamTensor},
};

pub type ExplicitKey = IndexlessNamedStructure<Symbol, Vec<Atom>, LibraryRep>;
pub type LibraryKey = PermutedStructure<ExplicitKey>;
impl ExplicitKey {
    pub fn from_structure<S: TensorStructure + HasName<Name: IntoSymbol, Args: IntoArgs>>(
        structure: &PermutedStructure<S>,
    ) -> Option<Self> {
        let rep_structure: Vec<_> = structure
            .structure
            .reps()
            .into_iter()
            .map(|r| r.to_lib())
            .collect();

        Some(
            IndexlessNamedStructure::from_iter(
                rep_structure,
                structure.structure.name()?.ref_into_symbol(),
                structure.structure.args().map(|a| a.args()),
            )
            .structure,
        )
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

    fn with_indices(
        &self,
        indices: &[<<<Self::WithIndices as HasStructure>::Structure as TensorStructure>::Slot as IsAbstractSlot>::Aind],
    ) -> Result<PermutedStructure<Self::WithIndices>, StructureError> {
        let new_tensor =
            <DataTensor<Atom, S> as LibraryTensor>::with_indices(&self.tensor, indices)?;
        Ok(PermutedStructure {
            structure: ParamTensor {
                tensor: new_tensor.structure,
                param_type: self.param_type,
            },
            rep_permutation: new_tensor.rep_permutation,
            index_permutation: new_tensor.index_permutation,
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
        indices: &[<<<Self::WithIndices as HasStructure>::Structure as TensorStructure>::Slot as IsAbstractSlot>::Aind],
    ) -> Result<PermutedStructure<Self::WithIndices>, StructureError> {
        Ok(match self {
            ParamOrConcrete::Concrete(c) => {
                let strct = <RealOrComplexTensor<D, S> as LibraryTensor>::with_indices(c, indices)?;
                PermutedStructure {
                    structure: ParamOrConcrete::Concrete(strct.structure),
                    rep_permutation: strct.rep_permutation,
                    index_permutation: strct.index_permutation,
                }
            }
            ParamOrConcrete::Param(p) => {
                let strct = <ParamTensor<S> as LibraryTensor>::with_indices(p, indices)?;
                PermutedStructure {
                    structure: ParamOrConcrete::Param(strct.structure),
                    rep_permutation: strct.rep_permutation,
                    index_permutation: strct.index_permutation,
                }
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
    explicit_dimension: AHashMap<ExplicitKey, PermutedStructure<T>>,
    generic_dimension: AHashMap<GenericKey, fn(ExplicitKey) -> T>,
}

pub struct ExplicitTensorSymbols {
    pub flat: Symbol,
    pub metric: Symbol,
}

pub static ETS: LazyLock<ExplicitTensorSymbols> = LazyLock::new(|| ExplicitTensorSymbols {
    flat: symbol!("♭";Symmetric),
    // sharp: symbol!("♯";Symmetric),
    metric: Symbol::id(),
});

impl IdentityName for Symbol {
    fn id() -> Self {
        symbol!(METRIC_NAME;Symmetric)
    }
}

impl<T: HasStructure<Structure = ExplicitKey>> TensorLibrary<T> {
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

    fn minus_one() -> Self {
        ConcreteOrParam::Concrete(T::minus_one())
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
    type Value = PermutedStructure<T>;
    // type Structure = ExplicitKey;

    fn get<'a>(&'a self, key: &Self::Key) -> Result<Cow<'a, Self::Value>, LibraryError<Self::Key>> {
        // println!("Trying:{}", key);
        if let Some(tensor) = self.explicit_dimension.get(key) {
            // println!("found explicit");
            Ok(Cow::Borrowed(tensor))
        } else if let Some(builder) = self.generic_dimension.get(&key.clone().into()) {
            let permutation = PermutedStructure {
                structure: builder(key.clone()),
                rep_permutation: Permutation::id(key.order()),
                index_permutation: Permutation::id(key.order()),
            };
            // println!("found generic");
            Ok(Cow::Owned(permutation))
        } else {
            Err(LibraryError::NotFound(key.clone()))
        }
    }

    fn key_for_structure(
        &self,
        structure: &PermutedStructure<S>,
    ) -> Result<Self::Key, LibraryError<Self::Key>>
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

impl<
        T: HasStructure<Structure = ExplicitKey>
            + SetTensorData
            + Clone
            + LibraryTensor
            + PermuteTensor<Permuted = T>,
    > TensorLibrary<T>
{
    pub fn get_key_from_name(
        &self,
        name: Symbol,
    ) -> Result<ExplicitKey, LibraryError<ExplicitKey>> {
        let keys: Vec<_> = self
            .explicit_dimension
            .keys()
            .filter(|k| k.name().unwrap() == name)
            .collect();

        match keys.len() {
            0 => Err(LibraryError::InvalidKey),
            1 => Ok(keys[0].clone()),
            _ => Err(LibraryError::MultipleKeys(name.to_string())),
        }
    }
    pub fn metric_key(rep: LibraryRep) -> ExplicitKey {
        ExplicitKey::from_iter([rep.new_rep(4), rep.new_rep(4)], ETS.metric, None).structure
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
        initialize();
        for r in LibraryRep::all_dualizables() {
            self.insert_generic(Self::id(*r), Self::checked_identity);
        }

        for r in LibraryRep::all_self_duals() {
            let id_metric = GenericKey::new(ETS.metric, None, vec![*r, *r]);
            self.insert_generic(id_metric, Self::checked_identity);
        }

        for r in LibraryRep::all_inline_metrics() {
            self.insert_generic(
                GenericKey::new(ETS.flat, None, vec![*r, *r]),
                Self::checked_identity,
            );
            let id_metric = GenericKey::new(ETS.metric, None, vec![*r, *r]);
            self.insert_generic(id_metric, Self::diag_unimodular_metric);
        }
    }

    pub fn id(rep: LibraryRep) -> GenericKey {
        GenericKey::new(ETS.metric, None, vec![rep, rep.dual()])
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

    pub fn diag_unimodular_metric(key: ExplicitKey) -> T
    where
        T::SetData: TensorLibraryData,
    {
        let dim: usize = key.get_dim(0).unwrap().try_into().unwrap();
        let rep = key.get_rep(0).unwrap();
        let mut tensor = T::empty(key);

        for i in 0..dim {
            if rep.is_neg(i) {
                tensor.set(&[i, i], T::SetData::minus_one()).unwrap();
            } else {
                tensor.set(&[i, i], T::SetData::one()).unwrap();
            }
        }
        tensor.into()
    }

    pub fn insert_explicit(&mut self, data: PermutedStructure<T>) {
        let key = data.structure.structure().clone();
        self.explicit_dimension.insert(key, data);
    }

    pub fn insert_explicit_dense(
        &mut self,
        key: PermutedStructure<ExplicitKey>,
        data: Vec<T::Data>,
    ) -> Result<()> {
        let tensor = T::from_dense(key.structure.clone(), data)?;
        let perm_tensor = PermutedStructure {
            rep_permutation: key.rep_permutation,
            index_permutation: key.index_permutation,
            structure: tensor,
        };

        self.explicit_dimension
            .insert(key.structure, perm_tensor.permute_inds_wrapped());
        Ok(())
    }

    pub fn insert_explicit_sparse(
        &mut self,
        key: PermutedStructure<ExplicitKey>,
        data: impl IntoIterator<Item = (Vec<ConcreteIndex>, T::Data)>,
    ) -> Result<()> {
        let tensor = T::from_sparse(key.structure.clone(), data)?;

        let perm_tensor = PermutedStructure {
            rep_permutation: key.rep_permutation.clone(),
            index_permutation: key.index_permutation.clone(),
            structure: tensor,
        };

        self.explicit_dimension
            .insert(key.structure, perm_tensor.permute_inds_wrapped());
        Ok(())
    }

    pub fn insert_generic(&mut self, key: GenericKey, data: fn(ExplicitKey) -> T) {
        self.generic_dimension.insert(key, data);
    }

    pub fn get(&self, key: &ExplicitKey) -> Result<Cow<T>>
    where
        T: Clone,
    {
        // println!("Trying:{}", key);

        if let Some(tensor) = self.explicit_dimension.get(key) {
            Ok(Cow::Borrowed(&tensor.structure))
        } else if let Some(builder) = self.generic_dimension.get(&key.clone().into()) {
            Ok(Cow::Owned(builder(key.clone())))
        } else {
            Err(LibraryError::NotFound(key.clone()).into())
        }
    }
}

#[cfg(test)]
mod test {
    use symbolica::{function, parse};

    use crate::{
        network::{
            parsing::ShadowedStructure, store::NetworkStore, ExecutionResult, Network, Sequential,
            SmallestDegree, TensorOrScalarOrKey,
        },
        shadowing::Concretize,
        structure::{
            representation::{Euclidean, Minkowski},
            ToSymbolic,
        },
        tensors::data::SparseOrDense,
    };

    use super::*;

    #[test]
    fn add_to_lib() {
        let mut lib = TensorLibrary::<MixedTensor<f64, ExplicitKey>>::new();
        let key = ExplicitKey::from_iter(
            [
                Euclidean {}.new_rep(4).cast(),
                Euclidean {}.new_rep(4).cast(),
                LibraryRep::from(Minkowski {}).new_rep(4),
            ],
            symbol!("gamma"),
            None,
        );

        println!("{}", key.structure);
        println!("{}", key.rep_permutation);

        let one = ConcreteOrParam::Concrete(RealOrComplex::Real(1.));
        lib.insert_explicit_sparse((key).clone(), [(vec![0, 0, 1], one)])
            .unwrap();

        lib.get(&key.structure).unwrap();
        let indexed = key.clone().reindex([0, 1, 2]).unwrap().structure;
        let expr = indexed.to_symbolic(None).unwrap();
        let mut net = Network::<
            NetworkStore<MixedTensor<f64, ShadowedStructure>, ConcreteOrParam<RealOrComplex<f64>>>,
            _,
        >::try_from_view(expr.as_view(), &lib)
        .unwrap();

        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.name().map(|a| a.to_string()).unwrap_or("".to_owned())
            )
        );

        net.execute::<Sequential, SmallestDegree, _, _>(&lib)
            .unwrap();
        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.name().map(|a| a.to_string()).unwrap_or("".to_owned())
            )
        );
    }
    #[test]
    fn libperm() {
        let mut lib = TensorLibrary::<MixedTensor<f64, ExplicitKey>>::new();
        let key = ExplicitKey::from_iter(
            [
                Euclidean {}.new_rep(2).cast(),
                Euclidean {}.new_rep(2).cast(),
                LibraryRep::from(Minkowski {}).new_rep(2),
            ],
            symbol!("gamma"),
            None,
        );

        println!("{}", key.structure);
        println!("{}", key.rep_permutation);

        let tensor = MixedTensor::Param(
            ParamTensor::from_sparse(
                key.structure.clone(),
                [
                    (vec![0, 0, 0], parse!("a").into()),
                    (vec![0, 0, 1], parse!("b").into()),
                    (vec![0, 1, 0], parse!("c").into()),
                    (vec![0, 1, 1], parse!("d").into()),
                    (vec![1, 0, 0], parse!("e").into()),
                    (vec![1, 0, 1], parse!("f").into()),
                    (vec![1, 1, 0], parse!("g").into()),
                    (vec![1, 1, 1], parse!("h").into()),
                ],
            )
            .unwrap()
            .to_dense(),
        );

        lib.insert_explicit(PermutedStructure {
            structure: tensor,
            rep_permutation: key.rep_permutation.clone(),
            index_permutation: key.index_permutation.clone(),
        });

        lib.get(&key.structure).unwrap();
        let mut net = Network::<
            NetworkStore<MixedTensor<f64, ShadowedStructure>, ConcreteOrParam<RealOrComplex<f64>>>,
            _,
        >::try_from_view(
            parse!("gamma(euc(2,1),euc(2,2),mink(2,0))-gamma(mink(2,0),euc(2,2),euc(2,1))")
                .as_view(),
            &lib,
        )
        .unwrap();

        net.execute::<Sequential, SmallestDegree, _, _>(&lib)
            .unwrap();

        print!("One {}", net.result_tensor(&lib).unwrap());
        let mut net = Network::<
            NetworkStore<MixedTensor<f64, ShadowedStructure>, ConcreteOrParam<RealOrComplex<f64>>>,
            _,
        >::try_from_view(
            parse!("gamma(mink(2,0),euc(2,2),euc(2,1))").as_view(), &lib
        )
        .unwrap();

        net.execute::<Sequential, SmallestDegree, _, _>(&lib)
            .unwrap();

        print!("Two{}", net.result_tensor(&lib).unwrap())
    }

    #[test]
    fn not_in_lib() {
        let lib = TensorLibrary::<MixedTensor<f64, ExplicitKey>>::new();
        let key = ExplicitKey::from_iter(
            [
                Euclidean {}.new_rep(4).cast(),
                LibraryRep::from(Minkowski {}).new_rep(4),
                Euclidean {}.new_rep(4).cast(),
            ],
            symbol!("gamma"),
            None,
        );

        let indexed = key.reindex([0, 1, 2]).unwrap().structure;
        let expr = indexed.to_symbolic(None).unwrap();
        let mut net = Network::<
            NetworkStore<MixedTensor<f64, ShadowedStructure>, ConcreteOrParam<RealOrComplex<f64>>>,
            _,
        >::try_from_view(expr.as_view(), &lib)
        .unwrap();

        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.name().map(|a| a.to_string()).unwrap_or("".to_owned())
            )
        );

        net.execute::<Sequential, SmallestDegree, _, _>(&lib)
            .unwrap();
        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.name().map(|a| a.to_string()).unwrap_or("".to_owned())
            )
        );

        if let ExecutionResult::Val(TensorOrScalarOrKey::Tensor { tensor, .. }) =
            net.result().unwrap()
        {
            // println!("YaY:{a}");
            println!("{tensor}");
            assert_eq!(tensor, &indexed.to_shell().concretize(None));
        } else {
            panic!("Not Key")
        }
    }

    #[test]
    fn flat() {
        let mut lib = TensorLibrary::<MixedTensor<f64, ExplicitKey>>::new();
        lib.update_ids();
        fn p(m: impl Into<AbstractIndex>) -> Atom {
            let m_atom: AbstractIndex = m.into();
            let m_atom: Atom = m_atom.into();
            let mink = Minkowski {}.new_rep(4);
            function!(symbol!("spenso::p"), mink.to_symbolic([m_atom]))
        }

        let mink = Minkowski {}.new_rep(4);

        let a = mink.flat(1, 2) * p(2);

        let mut net = Network::<
            NetworkStore<MixedTensor<f64, ShadowedStructure>, ConcreteOrParam<RealOrComplex<f64>>>,
            _,
        >::try_from_view(a.as_view(), &lib)
        .map_err(|a| a.to_string())
        .unwrap();

        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.name().map(|a| a.to_string()).unwrap_or("".to_owned())
            )
        );

        net.execute::<Sequential, SmallestDegree, _, _>(&lib)
            .unwrap();

        if let Ok(ExecutionResult::Val(v)) = net.result_tensor(&lib) {
            println!("{}", v)
        }
    }

    #[test]
    fn dot() {
        let lib = TensorLibrary::<MixedTensor<f64, ExplicitKey>>::new();

        let expr = parse!("p(1,mink(4,2))*q(2,mink(4,2))");
        let mut net = Network::<
            NetworkStore<MixedTensor<f64, ShadowedStructure>, ConcreteOrParam<RealOrComplex<f64>>>,
            _,
        >::try_from_view(expr.as_view(), &lib)
        .map_err(|a| a.to_string())
        .unwrap();

        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.name().map(|a| a.to_string()).unwrap_or("".to_owned())
            )
        );

        net.execute::<Sequential, SmallestDegree, _, _>(&lib)
            .unwrap();
        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.name().map(|a| a.to_string()).unwrap_or("".to_owned())
            )
        );

        if let Ok(ExecutionResult::Val(a)) = net.result_scalar() {
            if let ConcreteOrParam::Param(a) = a.as_ref() {
                let res= parse!("p(1,cind(0))*q(2,cind(0))-p(1,cind(1))*q(2,cind(1))-p(1,cind(2))*q(2,cind(2))-p(1,cind(3))*q(2,cind(3))");
                assert_eq!(a, &res);
            } else {
                panic!("Not Key")
            }
        } else {
            panic!("Not Key")
        }
    }

    #[test]
    fn big_expr() {
        initialize();
        let mut lib = TensorLibrary::<MixedTensor<f64, ExplicitKey>>::new();
        lib.update_ids();

        let expr = parse!(" -G^2*(-g(mink(4,5),mink(4,6))*Q(2,mink(4,7))+g(mink(4,5),mink(4,6))*Q(3,mink(4,7))+g(mink(4,5),mink(4,7))*Q(2,mink(4,6))+g(mink(4,5),mink(4,7))*Q(4,mink(4,6))-g(mink(4,6),mink(4,7))*Q(3,mink(4,5))-g(mink(4,6),mink(4,7))*Q(4,mink(4,5)))*g(mink(4,2),mink(4,5))*g(mink(4,3),mink(4,6))*g(euc(4,0),euc(4,5))*g(euc(4,1),euc(4,4))*g(mink(4,4),mink(4,7))*vbar(1,euc(4,1))*u(0,euc(4,0))*ϵbar(2,mink(4,2))*ϵbar(3,mink(4,3))*gamma(euc(4,5),euc(4,4),mink(4,4))");
        let mut net = Network::<
            NetworkStore<MixedTensor<f64, ShadowedStructure>, ConcreteOrParam<RealOrComplex<f64>>>,
            _,
        >::try_from_view(expr.as_view(), &lib)
        .map_err(|a| a.to_string())
        .unwrap();

        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.name().map(|a| a.to_string()).unwrap_or("".to_owned())
            )
        );

        net.execute::<Sequential, SmallestDegree, _, _>(&lib)
            .unwrap();

        if let Ok(ExecutionResult::Val(v)) = net.result_scalar() {
            println!("Hi{}", v)
        }
        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.name().map(|a| a.to_string()).unwrap_or("".to_owned())
            )
        );
    }

    #[test]
    fn small_expr() {
        initialize();
        let mut lib = TensorLibrary::<MixedTensor<f64, ExplicitKey>>::new();
        lib.update_ids();

        let expr =
            parse!("(-g(mink(4,5),mink(4,6))*Q(2,mink(4,7))+g(mink(4,5),mink(4,6))*Q(3,mink(4,7)))*g(mink(4,2),mink(4,5))*g(mink(4,3),mink(4,6))*g(mink(4,4),mink(4,7))*ϵbar(2,mink(4,2))*ϵbar(3,mink(4,3))")
                ;
        let mut net = Network::<
            NetworkStore<MixedTensor<f64, ShadowedStructure>, ConcreteOrParam<RealOrComplex<f64>>>,
            _,
        >::try_from_view(expr.as_view(), &lib)
        .map_err(|a| a.to_string())
        .unwrap();

        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.name().map(|a| a.to_string()).unwrap_or("".to_owned())
            )
        );

        net.execute::<Sequential, SmallestDegree, _, _>(&lib)
            .unwrap();

        if let Ok(ExecutionResult::Val(v)) = net.result_tensor(&lib) {
            println!("{}", v)
        }
        println!(
            "{}",
            net.dot_display_impl(
                |a| a.to_string(),
                |_| None,
                |a| a.name().map(|a| a.to_string()).unwrap_or("".to_owned())
            )
        );
    }

    #[test]
    fn one_times_x() {
        let mut a: Network<NetworkStore<MixedTensor<f64, ShadowedStructure>, Atom>, DummyKey> =
            Network::one() * Network::from_scalar(Atom::var(symbol!("x")));

        // a.merge_ops();
        a.execute::<Sequential, SmallestDegree, _, _>(&DummyLibrary::default())
            .unwrap();

        let res = a.result_scalar();
        if let Ok(ExecutionResult::Val(v)) = res {
            println!("Hi{}", v)
        } else {
            // panic!("AAAAA{}", a.dot())
        }
    }

    #[test]
    fn transposition() {
        initialize();
        let mut lib = TensorLibrary::<MixedTensor<f64, ExplicitKey>>::new();
        lib.update_ids();

        let key = ExplicitKey::from_iter(
            [Euclidean {}.new_rep(4), Euclidean {}.new_rep(4)],
            symbol!("A"),
            None,
        )
        .structure;

        let mut a: DataTensor<_, _> = DenseTensor::fill(key.clone(), Atom::num(1)).into();
        a.set(&[3, 0], parse!("a")).unwrap();
        let a = PermutedStructure::identity(MixedTensor::<f64, ExplicitKey>::param(a));

        lib.insert_explicit(a);
        #[allow(non_snake_case)]
        fn A(i: impl Into<AbstractIndex>, j: impl Into<AbstractIndex>) -> Atom {
            let euc = Euclidean {}.new_rep(4);
            function!(symbol!("A"), euc.slot(i).to_atom(), euc.slot(j).to_atom())
        }

        let expr = A(0, 1) - A(1, 0);

        let mut net = Network::<
            NetworkStore<MixedTensor<f64, ShadowedStructure>, ConcreteOrParam<RealOrComplex<f64>>>,
            _,
        >::try_from_view(expr.as_view(), &lib)
        .map_err(|a| a.to_string())
        .unwrap();

        net.execute::<Sequential, SmallestDegree, _, _>(&lib)
            .unwrap();

        if let Ok(ExecutionResult::Val(v)) = net.result_tensor(&lib) {
            println!("{}", v)
        }
    }
    #[test]
    fn trace_metric_kron() {
        initialize();
        let mut lib = TensorLibrary::<MixedTensor<f64, ExplicitKey>>::new();
        lib.update_ids();

        let mink = Minkowski {}.new_rep(4);

        let expr = mink.id(1, 0) * mink.id(0, 1);

        let mut net = Network::<
            NetworkStore<MixedTensor<f64, ShadowedStructure>, ConcreteOrParam<RealOrComplex<f64>>>,
            _,
        >::try_from_view(expr.as_view(), &lib)
        .map_err(|a| a.to_string())
        .unwrap();

        net.execute::<Sequential, SmallestDegree, _, _>(&lib)
            .unwrap();

        if let Ok(ExecutionResult::Val(v)) = net.result_tensor(&lib) {
            println!("{}", v)
        }
    }
}
