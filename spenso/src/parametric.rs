extern crate derive_more;

use std::{
    fmt::{Debug, Display},
    io::Cursor,
};

use crate::structure::abstract_index::AbstractIndex;
use crate::structure::dimension::Dimension;
use crate::structure::representation::Representation;
use crate::structure::slot::IsAbstractSlot;
use crate::structure::StructureError;
use ahash::HashMap;
use delegate::delegate;

use anyhow::anyhow;
use anyhow::{Error, Result};

use atomcore::{ReplaceBuilderGeneric, TensorAtomMaps};
// use anyhow::Ok;
use enum_try_as_inner::EnumTryAsInner;
use log::trace;
use serde::{de, ser::SerializeStruct, Deserialize, Serialize, Serializer};

use crate::{
    arithmetic::ScalarMul,
    complex::{Complex, RealOrComplex, RealOrComplexTensor},
    contraction::{Contract, ContractableWith, ContractionError, IsZero, RefZero, Trace},
    data::{
        DataIterator, DataTensor, DenseTensor, GetTensorData, HasTensorData, SetTensorData,
        SparseOrDense, SparseTensor, StorageTensor,
    },
    iterators::{IteratableTensor, IteratorEnum},
    shadowing::{ShadowMapping, Shadowable},
    structure::{
        concrete_index::{ConcreteIndex, DualConciousExpandedIndex, ExpandedIndex, FlatIndex},
        representation::LibraryRep,
        slot::Slot,
        CastStructure, HasName, HasStructure, NamedStructure, ScalarStructure, ScalarTensor,
        StructureContract, TensorStructure, ToSymbolic, TracksCount, VecStructure,
    },
    symbolica_utils::{IntoArgs, IntoSymbol, SerializableAtom},
    upgrading_arithmetic::{FallibleAddAssign, FallibleMul, FallibleSubAssign, TrySmallestUpgrade},
};
use bincode::{Decode, Encode};

use symbolica::{
    atom::{representation::FunView, Atom, AtomCore, AtomView, FunctionBuilder, KeyLookup, Symbol},
    coefficient::Coefficient,
    domains::{
        float::{NumericalFloatLike, Real},
        rational::Rational,
    },
    evaluate::{
        CompileOptions, CompiledCode, CompiledEvaluator, CompiledEvaluatorFloat, EvalTree,
        EvaluationFn, ExportedCode, Expression, ExpressionEvaluator, FunctionMap, InlineASM,
    },
    id::Pattern,
    state::{State, StateMap},
    symbol,
    utils::BorrowedOrOwned,
};

use std::hash::Hash;

use symbolica::domains::float::Complex as SymComplex;

// impl RefZero for Atom {
//     fn ref_zero(&self) -> Self {
//         Atom::new_num(0)
//     }
// }

pub trait TensorCoefficient: Display {
    fn cooked_name(&self) -> Option<String>;
    fn name(&self) -> Option<Symbol>;
    fn tags(&self) -> Vec<Atom>;
    fn to_atom(&self) -> Option<Atom>;
    fn to_atom_re(&self) -> Option<Atom>;
    fn to_atom_im(&self) -> Option<Atom>;
    fn add_tagged_function<T>(
        &self,
        fn_map: &mut FunctionMap<T>,
        body: Atom,
    ) -> Result<(), String> {
        let (name, cooked_name) = self
            .name()
            .zip(self.cooked_name())
            .ok_or(format!("unnamed {}", self))?;

        fn_map
            .add_tagged_function(name, self.tags(), cooked_name, vec![], body)
            .map_err(String::from)
    }
}

#[derive(Debug)]
pub struct FlatCoefficent<Args: IntoArgs> {
    pub name: Option<Symbol>,
    pub index: FlatIndex,
    pub args: Option<Args>,
}

impl<Arg: IntoArgs> Display for FlatCoefficent<Arg> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(name) = self.name {
            write!(f, "{}", name)?
        }
        write!(f, "(")?;
        if let Some(ref args) = self.args {
            let args: Vec<String> = args.ref_into_args().map(|s| s.to_string()).collect();
            write!(f, "{},", args.join(","))?
        }

        write!(f, "{})", self.index)?;
        Result::Ok(())
    }
}

impl<Args: IntoArgs> TensorCoefficient for FlatCoefficent<Args> {
    fn name(&self) -> Option<Symbol> {
        self.name
    }

    fn cooked_name(&self) -> Option<String> {
        let mut name = self.name?.to_string();
        if let Some(ref args) = self.args {
            name += args.cooked_name().as_str();
        }
        Some(name)
    }

    fn tags(&self) -> Vec<Atom> {
        let mut tags: Vec<Atom> = if let Some(ref args) = self.args {
            args.args()
        } else {
            vec![]
        };
        tags.push(Atom::from(self.index));
        tags
    }

    fn to_atom(&self) -> Option<Atom> {
        let mut fn_builder = FunctionBuilder::new(self.name?);
        if let Some(ref args) = self.args {
            for arg in args.ref_into_args() {
                fn_builder = fn_builder.add_arg(arg.as_view());
            }
        }
        fn_builder = fn_builder.add_arg(Atom::from(self.index).as_view());
        Some(fn_builder.finish())
    }

    fn to_atom_re(&self) -> Option<Atom> {
        let name = symbol!(self.name?.to_string() + "_re");

        let mut fn_builder = FunctionBuilder::new(name);
        if let Some(ref args) = self.args {
            for arg in args.ref_into_args() {
                fn_builder = fn_builder.add_arg(arg.as_view());
            }
        }
        fn_builder = fn_builder.add_arg(Atom::from(self.index).as_view());
        Some(fn_builder.finish())
    }

    fn to_atom_im(&self) -> Option<Atom> {
        let name = symbol!(self.name?.to_string() + "_im");

        let mut fn_builder = FunctionBuilder::new(name);
        if let Some(ref args) = self.args {
            for arg in args.ref_into_args() {
                fn_builder = fn_builder.add_arg(arg.as_view());
            }
        }
        fn_builder = fn_builder.add_arg(Atom::from(self.index).as_view());
        Some(fn_builder.finish())
    }
}

pub struct ExpandedCoefficent<Args: IntoArgs> {
    pub name: Option<Symbol>,
    pub index: DualConciousExpandedIndex,
    pub args: Option<Args>,
}

impl<Arg: IntoArgs> Display for ExpandedCoefficent<Arg> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(name) = self.name {
            write!(f, "{}", name)?
        }
        write!(f, "(")?;
        if let Some(ref args) = self.args {
            let args: Vec<String> = args.ref_into_args().map(|s| s.to_string()).collect();
            write!(f, "{},", args.join(","))?
        }
        write!(f, "{})", self.index)?;
        Result::Ok(())
    }
}

impl<Args: IntoArgs> TensorCoefficient for ExpandedCoefficent<Args> {
    fn name(&self) -> Option<Symbol> {
        self.name
    }
    fn cooked_name(&self) -> Option<String> {
        let mut name = self.name?.to_string();
        if let Some(ref args) = self.args {
            name += args.cooked_name().as_str();
        }
        Some(name)
    }

    fn tags(&self) -> Vec<Atom> {
        let mut tags: Vec<Atom> = if let Some(ref args) = self.args {
            args.args()
        } else {
            vec![]
        };
        tags.push(Atom::from(self.index.clone()));
        tags
    }

    fn to_atom(&self) -> Option<Atom> {
        let mut fn_builder = FunctionBuilder::new(self.name?);
        if let Some(ref args) = self.args {
            for arg in args.ref_into_args() {
                fn_builder = fn_builder.add_arg(arg.as_view());
            }
        }
        fn_builder = fn_builder.add_arg(Atom::from(self.index.clone()).as_view());
        Some(fn_builder.finish())
    }
    fn to_atom_re(&self) -> Option<Atom> {
        let name = symbol!(self.name?.to_string() + "_re");

        let mut fn_builder = FunctionBuilder::new(name);
        if let Some(ref args) = self.args {
            for arg in args.ref_into_args() {
                fn_builder = fn_builder.add_arg(arg.as_view());
            }
        }
        fn_builder = fn_builder.add_arg(Atom::from(self.index.clone()).as_view());
        Some(fn_builder.finish())
    }

    fn to_atom_im(&self) -> Option<Atom> {
        let name = symbol!(self.name?.to_string() + "_im");

        let mut fn_builder = FunctionBuilder::new(name);
        if let Some(ref args) = self.args {
            for arg in args.ref_into_args() {
                fn_builder = fn_builder.add_arg(arg.as_view());
            }
        }
        fn_builder = fn_builder.add_arg(Atom::from(self.index.clone()).as_view());
        Some(fn_builder.finish())
    }
}

impl<'a> TryFrom<FunView<'a>> for DenseTensor<Atom, NamedStructure<Symbol, Vec<Atom>>> {
    type Error = Error;

    fn try_from(f: FunView<'a>) -> Result<Self> {
        let mut structure: Vec<Slot<LibraryRep>> = vec![];
        let f_id = f.get_symbol();
        let mut args = vec![];

        for arg in f.iter() {
            if let Ok(arg) = arg.try_into() {
                structure.push(arg);
            } else {
                args.push(arg.to_owned());
            }
        }
        let s = NamedStructure::from_iter(structure, f_id, Some(args));
        s.to_dense_expanded_labels()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Encode, Decode)]
#[bincode(decode_context = "symbolica::state::StateMap")]
pub struct ParamTensor<S = VecStructure> {
    pub tensor: DataTensor<Atom, S>,
    pub param_type: ParamOrComposite,
    // Param(DataTensor<Atom, S>),
    // // Concrete(DataTensor<T, S>),
    // Composite(DataTensor<Atom, S>),
}

impl<S> TensorStructure for ParamTensor<S>
where
    S: TensorStructure,
{
    // type R = <T::Structure as TensorStructure>::R;
    type Indexed = ParamTensor<S::Indexed>;
    type Slot = S::Slot;

    fn reindex(self, indices: &[AbstractIndex]) -> Result<Self::Indexed, StructureError> {
        self.map_structure_result(|s| s.reindex(indices))
    }

    fn dual(self) -> Self {
        self.map_same_structure(|s| s.dual())
    }

    delegate! {
        to self.structure() {
            fn external_reps_iter(&self)-> impl Iterator<Item = Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn external_indices_iter(&self)-> impl Iterator<Item = AbstractIndex>;
            fn external_dims_iter(&self)-> impl Iterator<Item = Dimension>;
            fn external_structure_iter(&self)-> impl Iterator<Item = Self::Slot>;
            fn get_slot(&self, i: usize)-> Option<Self::Slot>;
            fn get_rep(&self, i: usize)-> Option<Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn get_dim(&self, i: usize)-> Option<Dimension>;
            fn get_aind(&self, i: usize)-> Option<AbstractIndex>;
            fn order(&self)-> usize;
        }
    }
}

impl<S: TensorStructure + Clone> StorageTensor for ParamTensor<S> {
    type Data = Atom;
    type ContainerData<Data> = DataTensor<Data, S>;

    fn map_data<U>(self, f: impl Fn(Self::Data) -> U) -> Self::ContainerData<U> {
        self.tensor.map_data(f)
    }

    fn map_data_ref_mut_result<U, E>(
        &mut self,
        f: impl FnMut(&mut Self::Data) -> Result<U, E>,
    ) -> Result<Self::ContainerData<U>, E> {
        self.tensor.map_data_ref_mut_result(f)
    }

    fn map_data_ref_result_self<E>(
        &self,
        f: impl Fn(&Self::Data) -> Result<Self::Data, E>,
    ) -> Result<Self, E> {
        Ok(ParamTensor {
            tensor: self.tensor.map_data_ref_result_self(f)?,
            param_type: self.param_type,
        })
    }

    fn map_data_mut(&mut self, f: impl FnMut(&mut Self::Data)) {
        self.tensor.map_data_mut(f)
    }

    fn map_data_ref<U>(&self, f: impl Fn(&Self::Data) -> U) -> Self::ContainerData<U> {
        self.tensor.map_data_ref(f)
    }

    fn map_data_ref_mut<U>(
        &mut self,
        f: impl FnMut(&mut Self::Data) -> U,
    ) -> Self::ContainerData<U> {
        self.tensor.map_data_ref_mut(f)
    }

    fn map_data_ref_self(&self, f: impl Fn(&Self::Data) -> Self::Data) -> Self {
        ParamTensor {
            param_type: self.param_type,
            tensor: self.tensor.map_data_ref(f),
        }
    }

    fn map_data_ref_result<U, E>(
        &self,
        f: impl Fn(&Self::Data) -> Result<U, E>,
    ) -> Result<Self::ContainerData<U>, E> {
        self.tensor.map_data_ref_result(f)
    }

    fn map_data_ref_mut_self(&mut self, f: impl FnMut(&mut Self::Data) -> Self::Data) -> Self {
        ParamTensor {
            param_type: self.param_type,
            tensor: self.tensor.map_data_ref_mut_self(f),
        }
    }

    fn map_data_self(self, f: impl Fn(Self::Data) -> Self::Data) -> Self {
        ParamTensor {
            param_type: self.param_type,
            tensor: self.tensor.map_data(f),
        }
    }
}

impl<S> From<DataTensor<Atom, S>> for ParamTensor<S>
where
    S: TensorStructure + Clone,
{
    fn from(tensor: DataTensor<Atom, S>) -> Self {
        ParamTensor {
            tensor,
            param_type: ParamOrComposite::Composite,
        }
    }
}

impl<S> From<SparseTensor<Atom, S>> for ParamTensor<S>
where
    S: TensorStructure + Clone,
{
    fn from(tensor: SparseTensor<Atom, S>) -> Self {
        ParamTensor {
            tensor: tensor.into(),
            param_type: ParamOrComposite::Composite,
        }
    }
}

impl<S> From<DenseTensor<Atom, S>> for ParamTensor<S>
where
    S: TensorStructure + Clone,
{
    fn from(tensor: DenseTensor<Atom, S>) -> Self {
        ParamTensor {
            tensor: tensor.into(),
            param_type: ParamOrComposite::Composite,
        }
    }
}

impl<S: TensorStructure + Clone> ScalarMul<SerializableAtom> for ParamTensor<S> {
    type Output = ParamTensor<S>;
    fn scalar_mul(&self, rhs: &SerializableAtom) -> Option<Self::Output> {
        Some(ParamTensor {
            tensor: self.tensor.scalar_mul(&rhs.0)?,
            param_type: ParamOrComposite::Composite,
        })
    }
}

// impl<C: HasStructure<Structure = S> + Clone, S: TensorStructure + Clone> ScalarMul<SerializableAtom>
//     for ParamOrConcrete<C, S> where C
// {
//     type Output = ParamTensor<S>;
//     fn scalar_mul(&self, rhs: &SerializableAtom) -> Option<Self::Output> {
//         match self{

//         }
//     }
// }

// impl<Structure: TensorStructure + Serialize + Clone> Serialize for ParamTensor<Structure> {
//     fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
//     where
//         S: serde::Serializer,
//     {
//         let mut state = serializer.serialize_struct("ParamTensor", 3)?;

//         state.serialize_field("param_type", &self.param_type)?;

//         let serialized_tensor = self.tensor.map_data_ref(|a| {
//             let mut v = Vec::new();
//             a.as_view().write(&mut v).unwrap();
//             v
//         });
//         state.serialize_field("tensor", &serialized_tensor)?;

//         let mut symbolica_state = Vec::new();

//         State::export(&mut symbolica_state).unwrap();

//         state.serialize_field("state", &symbolica_state)?;
//         state.end()
//     }
// }

impl<'a, Structure: TensorStructure + Deserialize<'a> + Clone> Deserialize<'a>
    for ParamTensor<Structure>
{
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'a>,
    {
        #[derive(Deserialize)]
        struct ParamTensorHelper<Structure: TensorStructure> {
            param_type: ParamOrComposite,
            tensor: DataTensor<Vec<u8>, Structure>,
            state: Vec<u8>,
        }

        let helper = ParamTensorHelper::deserialize(deserializer)?;

        let state = helper.state;

        let mut export = vec![];
        State::export(&mut export).unwrap();

        let map = State::import(&mut Cursor::new(&state), None).unwrap();

        Ok(ParamTensor {
            tensor: helper
                .tensor
                .map_data_ref_result(|a| Atom::import_with_map(a.as_slice(), &map))
                .map_err(serde::de::Error::custom)?,
            param_type: helper.param_type,
        })
    }
}

impl<S: TensorStructure + Clone> HasTensorData for ParamTensor<S> {
    type Data = Atom;
    fn data(&self) -> Vec<Self::Data> {
        self.tensor.data()
    }

    fn hashmap(&self) -> indexmap::IndexMap<ExpandedIndex, Self::Data> {
        self.tensor.hashmap()
    }

    fn indices(&self) -> Vec<ExpandedIndex> {
        self.tensor.indices()
    }

    fn symhashmap(
        &self,
        name: Symbol,
        args: &[Atom],
    ) -> std::collections::HashMap<Atom, Self::Data> {
        self.tensor.symhashmap(name, args)
    }
}
pub enum TensorSet<T: HasStructure> {
    Tensors(Vec<T>),
    Scalars(Vec<T::Scalar>),
}

impl<T: HasStructure> TensorSet<T> {
    pub fn len(&self) -> usize {
        match self {
            TensorSet::Tensors(t) => t.len(),
            TensorSet::Scalars(t) => t.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            TensorSet::Tensors(t) => t.is_empty(),
            TensorSet::Scalars(t) => t.is_empty(),
        }
    }

    pub fn push(&mut self, tensor: T) {
        match self {
            TensorSet::Scalars(t) => t.push(tensor.scalar().unwrap()),
            TensorSet::Tensors(t) => t.push(tensor),
        }
    }
}

impl<T: HasStructure + Clone> FromIterator<T> for TensorSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let (scalars, set): (Vec<Option<T::Scalar>>, Vec<T>) =
            iter.into_iter().map(|t| (t.clone().scalar(), t)).collect();
        let scalars: Option<Vec<_>> = scalars.into_iter().collect();
        if let Some(s) = scalars {
            TensorSet::Scalars(s)
        } else {
            TensorSet::Tensors(set)
        }
    }
}

pub struct ParamTensorSet<S: TensorStructure + Clone> {
    pub tensors: TensorSet<ParamTensor<S>>,
    size: usize,
}

impl<S: TensorStructure + Clone> ParamTensorSet<S> {
    pub fn new(tensors: Vec<ParamTensor<S>>) -> Self {
        let size = tensors
            .iter()
            .map(|t| t.tensor.actual_size())
            .reduce(|acc, a| acc + a)
            .unwrap();

        ParamTensorSet {
            tensors: tensors.into_iter().collect(),
            size,
        }
    }

    pub fn empty() -> Self {
        ParamTensorSet {
            tensors: [].into_iter().collect(),
            size: 0,
        }
    }

    // pub fn push(&mut self, tensor: ParamTensor<S>) {
    //     self.size += tensor.tensor.actual_size();
    //     self.tensors.push(tensor);
    // }
}

impl<S: TensorStructure> ParamTensor<S> {
    pub fn param(tensor: DataTensor<Atom, S>) -> Self {
        ParamTensor {
            tensor,
            param_type: ParamOrComposite::Param,
        }
    }

    pub fn composite(tensor: DataTensor<Atom, S>) -> Self {
        ParamTensor {
            tensor,
            param_type: ParamOrComposite::Composite,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Copy, PartialEq, Eq, Hash, Encode, Decode)]
pub enum ParamOrComposite {
    Param,
    Composite,
}

impl<S: TensorStructure, O: From<S> + TensorStructure> CastStructure<ParamTensor<O>>
    for ParamTensor<S>
{
    fn cast_structure(self) -> ParamTensor<O> {
        ParamTensor {
            tensor: self.tensor.cast_structure(),
            param_type: self.param_type,
        }
    }
}

impl<S: TensorStructure> Shadowable for ParamTensor<S>
where
    S: HasName + Clone,
    S::Name: IntoSymbol,
    S::Args: IntoArgs,
{
}

impl<S: TensorStructure, Const> ShadowMapping<Const> for ParamTensor<S>
where
    S: HasName + Clone,
    S::Name: IntoSymbol,
    S::Args: IntoArgs,
{
    fn append_map<T>(
        &self,
        fn_map: &mut FunctionMap<Const>,
        index_to_atom: impl Fn(&Self::Structure, FlatIndex) -> T,
    ) where
        T: TensorCoefficient,
    {
        match self.param_type {
            ParamOrComposite::Param => {}
            ParamOrComposite::Composite => match &self.tensor {
                DataTensor::Dense(d) => {
                    for (i, a) in d.flat_iter() {
                        let labeled_coef = index_to_atom(self.structure(), i);

                        labeled_coef.add_tagged_function(fn_map, a.clone()).unwrap();
                    }
                }
                DataTensor::Sparse(d) => {
                    for (i, a) in d.flat_iter() {
                        let labeled_coef = index_to_atom(self.structure(), i);

                        labeled_coef.add_tagged_function(fn_map, a.clone()).unwrap();
                    }
                }
            },
        }
    }
}

pub mod atomcore;

use symbolica::id::BorrowReplacement;

impl<S: TensorStructure + Clone> ParamTensorSet<S> {
    pub fn eval_tree(
        &self,
        fn_map: &FunctionMap,
        params: &[Atom],
    ) -> Result<EvalTreeTensorSet<Rational, S>> {
        match &self.tensors {
            TensorSet::Scalars(s) => {
                trace!("turning {} scalars into eval tree", s.len());
                let exprs = s.iter().map(|a| a.as_view()).collect::<Vec<_>>();
                Ok(EvalTreeTensorSet {
                    tensors: TensorsOrScalars::Scalars,
                    eval: (
                        AtomView::to_eval_tree_multiple(&exprs, fn_map, params)
                            .map_err(|s| anyhow!(s))?,
                        None,
                    ),
                    size: self.size,
                })
            }
            TensorSet::Tensors(in_tensors) => {
                trace!("turning {} tensors into eval tree", in_tensors.len());
                let mut tensors = vec![];

                let mut atoms = vec![];
                let mut id = 0;
                for t in in_tensors.iter() {
                    let structure = t.structure().clone();
                    let usize_tensor = match &t.tensor {
                        DataTensor::Dense(d) => {
                            let oldid = id;
                            id += d.size().unwrap();
                            for (_, a) in d.flat_iter() {
                                atoms.push(a.as_view());
                            }
                            DataTensor::Dense(DenseTensor::from_data(
                                Vec::from_iter(oldid..id),
                                structure,
                            )?)
                        }
                        DataTensor::Sparse(s) => {
                            let mut t = SparseTensor::empty(structure);
                            for (i, a) in s.flat_iter() {
                                t.set_flat(i, id)?;
                                atoms.push(a.as_view());
                                id += 1;
                            }
                            DataTensor::Sparse(t)
                        }
                    };
                    tensors.push(usize_tensor);
                }

                Ok(EvalTreeTensorSet {
                    tensors: TensorsOrScalars::Tensors(tensors),
                    eval: (
                        AtomView::to_eval_tree_multiple(&atoms, fn_map, params)
                            .map_err(|s| anyhow!(s))?,
                        None,
                    ),
                    size: self.size,
                })
            }
        }
    }
}

impl<S: TensorStructure> IteratableTensor for ParamTensor<S> {
    type Data<'a>
        = AtomView<'a>
    where
        Self: 'a;

    fn iter_expanded(&self) -> impl Iterator<Item = (ExpandedIndex, Self::Data<'_>)> {
        self.tensor.iter_expanded().map(|(i, x)| (i, x.as_view()))
    }

    fn iter_flat(&self) -> impl Iterator<Item = (FlatIndex, Self::Data<'_>)> {
        self.tensor.iter_flat().map(|(i, x)| (i, x.as_view()))
    }
}

impl<S> Display for ParamTensor<S>
where
    S: TensorStructure,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.tensor)
    }
}

impl<S: TensorStructure> HasName for ParamTensor<S>
where
    S: HasName,
{
    type Args = S::Args;
    type Name = S::Name;

    fn args(&self) -> Option<Self::Args> {
        self.tensor.args()
    }

    fn name(&self) -> Option<Self::Name> {
        self.tensor.name()
    }

    fn set_name(&mut self, name: Self::Name) {
        if let ParamOrComposite::Composite = self.param_type {
            self.tensor.set_name(name);
        } // never set the name of a param tensor, it is always set by construction
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Encode)]
#[bincode(decode_context = "symbolica::state::StateMap")]
pub enum ParamOrConcrete<C, S> {
    Concrete(C),
    Param(ParamTensor<S>),
}

impl<C, S> Decode<StateMap> for ParamOrConcrete<C, S>
where
    C: Decode<StateMap>,
    S: Decode<StateMap>,
{
    fn decode<__D: ::bincode::de::Decoder<Context = StateMap>>(
        decoder: &mut __D,
    ) -> core::result::Result<Self, ::bincode::error::DecodeError> {
        let variant_index = <u32 as ::bincode::Decode<__D::Context>>::decode(decoder)?;
        match variant_index {
            0u32 => core::result::Result::Ok(Self::Concrete {
                0: ::bincode::Decode::<__D::Context>::decode(decoder)?,
            }),
            1u32 => core::result::Result::Ok(Self::Param {
                0: ::bincode::Decode::<__D::Context>::decode(decoder)?,
            }),
            variant => {
                core::result::Result::Err(::bincode::error::DecodeError::UnexpectedVariant {
                    found: variant,
                    type_name: "ParamOrConcrete",
                    allowed: &::bincode::error::AllowedEnumVariants::Range { min: 0, max: 1 },
                })
            }
        }
    }
}
impl<'__de, C, S> ::bincode::BorrowDecode<'__de, symbolica::state::StateMap>
    for ParamOrConcrete<C, S>
where
    C: ::bincode::de::BorrowDecode<'__de, symbolica::state::StateMap>,
    S: ::bincode::de::BorrowDecode<'__de, symbolica::state::StateMap>,
{
    fn borrow_decode<
        __D: ::bincode::de::BorrowDecoder<'__de, Context = symbolica::state::StateMap>,
    >(
        decoder: &mut __D,
    ) -> core::result::Result<Self, ::bincode::error::DecodeError> {
        let variant_index = <u32 as ::bincode::Decode<__D::Context>>::decode(decoder)?;
        match variant_index {
            0u32 => core::result::Result::Ok(Self::Concrete {
                0: ::bincode::BorrowDecode::<__D::Context>::borrow_decode(decoder)?,
            }),
            1u32 => core::result::Result::Ok(Self::Param {
                0: ::bincode::BorrowDecode::<__D::Context>::borrow_decode(decoder)?,
            }),
            variant => {
                core::result::Result::Err(::bincode::error::DecodeError::UnexpectedVariant {
                    found: variant,
                    type_name: "ParamOrConcrete",
                    allowed: &::bincode::error::AllowedEnumVariants::Range { min: 0, max: 1 },
                })
            }
        }
    }
}

impl<C: HasStructure<Structure = S> + Clone, S: TensorStructure + Clone> From<ParamTensor<S>>
    for ParamOrConcrete<C, S>
{
    fn from(tensor: ParamTensor<S>) -> Self {
        ParamOrConcrete::Param(tensor)
    }
}

impl<C: HasStructure<Structure = S> + Clone, S: TensorStructure + Clone> ParamOrConcrete<C, S> {
    pub fn replace<'b, P: Into<BorrowedOrOwned<'b, Pattern>>>(
        &self,
        pattern: P,
    ) -> ReplaceBuilderGeneric<'b, &'_ Self, Self> {
        ReplaceBuilderGeneric::new(self, pattern)
    }

    pub fn replace_multiple<T: BorrowReplacement>(&self, replacements: &[T]) -> Self {
        match self {
            ParamOrConcrete::Param(p) => ParamOrConcrete::Param(p.replace_multiple(replacements)),
            _ => self.clone(),
        }
    }
}

impl<C: Display + HasStructure<Structure = S> + Clone, S: TensorStructure + Clone> Display
    for ParamOrConcrete<C, S>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamOrConcrete::Concrete(c) => c.fmt(f),
            ParamOrConcrete::Param(p) => p.fmt(f),
        }
    }
}

impl<
        U: HasStructure<Structure = O> + Clone,
        C: CastStructure<U> + HasStructure<Structure = S> + Clone,
        S: TensorStructure + Clone,
        O: From<S> + TensorStructure + Clone,
    > CastStructure<ParamOrConcrete<U, O>> for ParamOrConcrete<C, S>
{
    fn cast_structure(self) -> ParamOrConcrete<U, O> {
        match self {
            ParamOrConcrete::Concrete(c) => ParamOrConcrete::Concrete(c.cast_structure()),
            ParamOrConcrete::Param(p) => ParamOrConcrete::Param(p.cast_structure()),
        }
    }
}

impl<
        C: HasStructure<Structure = S> + Clone + Shadowable,
        S: TensorStructure + Clone + HasName<Args: IntoArgs, Name: IntoSymbol>,
    > Shadowable for ParamOrConcrete<C, S>
{
}

impl<
        U,
        C: HasStructure<Structure = S> + Clone + ShadowMapping<U>,
        S: TensorStructure + Clone + HasName<Args: IntoArgs, Name: IntoSymbol>,
    > ShadowMapping<U> for ParamOrConcrete<C, S>
{
    // fn shadow_with_map<'a, T>(
    //     &'a self,
    //     fn_map: &mut FunctionMap<'a, U>,
    //     index_to_atom: impl Fn(&Self::Structure, FlatIndex) -> T,
    // ) -> Option<ParamTensor<Self::Structure>>
    // where
    //     T: TensorCoefficient,
    // {
    //     match self {
    //         ParamOrConcrete::Concrete(c) => c.shadow_with_map(fn_map, index_to_atom),
    //         ParamOrConcrete::Param(p) => p.shadow_with_map(fn_map, index_to_atom),
    //     }
    // }

    fn append_map<T>(
        &self,
        fn_map: &mut FunctionMap<U>,
        index_to_atom: impl Fn(&Self::Structure, FlatIndex) -> T,
    ) where
        T: TensorCoefficient,
    {
        match self {
            ParamOrConcrete::Concrete(c) => c.append_map(fn_map, index_to_atom),
            ParamOrConcrete::Param(p) => p.append_map(fn_map, index_to_atom),
        }
    }
}

pub enum AtomViewOrConcrete<'a, T> {
    Atom(AtomView<'a>),
    Concrete(T),
}

pub enum AtomOrConcrete<T> {
    Atom(Atom),
    Concrete(T),
}

pub trait Concrete {}

impl<T> From<Atom> for AtomOrConcrete<T> {
    fn from(value: Atom) -> Self {
        AtomOrConcrete::Atom(value)
    }
}

impl<T: Concrete> From<T> for AtomOrConcrete<T> {
    fn from(value: T) -> Self {
        AtomOrConcrete::Concrete(value)
    }
}

impl<C: HasStructure<Structure = S> + Clone, S: TensorStructure + Clone> ParamOrConcrete<C, S> {
    pub fn is_parametric(&self) -> bool {
        matches!(self, ParamOrConcrete::Param(_))
    }

    pub fn try_into_parametric(self) -> Result<ParamTensor<S>, Self> {
        match self {
            ParamOrConcrete::Param(x) => Ok(x),
            _ => Err(self),
        }
    }

    pub fn try_into_concrete(self) -> Result<C, Self> {
        match self {
            ParamOrConcrete::Concrete(x) => Ok(x),
            _ => Err(self),
        }
    }
}

#[derive(Clone, Debug, EnumTryAsInner)]
#[derive_err(Debug)]
pub enum ConcreteOrParam<C> {
    Concrete(C),
    Param(Atom),
}

#[derive(Clone, Debug, EnumTryAsInner)]
#[derive_err(Debug)]
pub enum ConcreteOrParamRef<'a, C> {
    Concrete(C),
    Param(AtomView<'a>),
}

impl<C: Default> Default for ConcreteOrParam<C> {
    fn default() -> Self {
        ConcreteOrParam::Concrete(C::default())
    }
}

impl<C: std::ops::Neg<Output = C>> std::ops::Neg for ConcreteOrParam<C> {
    type Output = ConcreteOrParam<C>;

    fn neg(self) -> Self {
        match self {
            ConcreteOrParam::Concrete(c) => ConcreteOrParam::Concrete(-c),
            ConcreteOrParam::Param(p) => ConcreteOrParam::Param(-p),
        }
    }
}

impl<C> From<SerializableAtom> for ConcreteOrParam<C> {
    fn from(value: SerializableAtom) -> Self {
        ConcreteOrParam::Param(value.into())
    }
}

#[derive(Clone, Debug, EnumTryAsInner)]
#[derive_err(Debug)]
pub enum ConcreteOrParamView<'a, C> {
    Concrete(C),
    Param(AtomView<'a>),
}

#[derive(Debug, EnumTryAsInner)]
#[derive_err(Debug)]
pub enum ConcreteOrParamViewMut<'a, C> {
    Concrete(C),
    Param(&'a mut Atom),
}

impl<D: Display> Display for ConcreteOrParam<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConcreteOrParam::Concrete(c) => c.fmt(f),
            ConcreteOrParam::Param(p) => write!(f, "{}", p),
        }
    }
}

impl<S: TensorStructure + Clone> SparseOrDense for ParamTensor<S> {
    fn to_dense(self) -> Self {
        ParamTensor {
            tensor: self.tensor.to_dense(),
            param_type: self.param_type,
        }
    }

    fn to_sparse(self) -> Self {
        ParamTensor {
            tensor: self.tensor.to_sparse(),
            param_type: self.param_type,
        }
    }
}

impl<C, S> SparseOrDense for ParamOrConcrete<C, S>
where
    C: SparseOrDense + Clone + HasStructure<Structure = S>,
    S: TensorStructure + Clone,
{
    fn to_dense(self) -> Self {
        match self {
            ParamOrConcrete::Concrete(x) => ParamOrConcrete::Concrete(x.to_dense()),
            ParamOrConcrete::Param(x) => ParamOrConcrete::Param(x.to_dense()),
        }
    }

    fn to_sparse(self) -> Self {
        match self {
            ParamOrConcrete::Concrete(x) => ParamOrConcrete::Concrete(x.to_sparse()),
            ParamOrConcrete::Param(x) => ParamOrConcrete::Param(x.to_sparse()),
        }
    }
}

impl<C, S> TensorStructure for ParamOrConcrete<C, S>
where
    C: HasStructure<Structure = S> + TensorStructure,
    C::Indexed: HasStructure<Structure = S::Indexed>,
    S: TensorStructure,
{
    // type R = <T::Structure as TensorStructure>::R;
    type Indexed = ParamOrConcrete<C::Indexed, S::Indexed>;
    type Slot = S::Slot;

    fn reindex(self, indices: &[AbstractIndex]) -> Result<Self::Indexed, StructureError> {
        Ok(match self {
            ParamOrConcrete::Concrete(c) => ParamOrConcrete::Concrete(c.reindex(indices)?),
            ParamOrConcrete::Param(p) => ParamOrConcrete::Param(p.reindex(indices)?),
        })
    }

    fn dual(self) -> Self {
        self.map_same_structure(|s| s.dual())
    }

    delegate! {
        to self.structure() {
            fn external_reps_iter(&self)-> impl Iterator<Item = Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn external_indices_iter(&self)-> impl Iterator<Item = AbstractIndex>;
            fn external_dims_iter(&self)-> impl Iterator<Item = Dimension>;
            fn external_structure_iter(&self)-> impl Iterator<Item = Self::Slot>;
            fn get_slot(&self, i: usize)-> Option<Self::Slot>;
            fn get_rep(&self, i: usize)-> Option<Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn get_dim(&self, i: usize)-> Option<Dimension>;
            fn get_aind(&self, i: usize)-> Option<AbstractIndex>;
            fn order(&self)-> usize;
        }
    }
}

impl<C, S> HasStructure for ParamOrConcrete<C, S>
where
    C: HasStructure<Structure = S>,
    S: TensorStructure,
{
    type Scalar = ConcreteOrParam<C::Scalar>;
    type ScalarRef<'a>
        = ConcreteOrParamRef<'a, C::ScalarRef<'a>>
    where
        Self: 'a;
    type Structure = S;
    type Store<U>
        = ParamOrConcrete<C::Store<U>, U>
    where
        U: TensorStructure;

    fn map_structure<O: TensorStructure>(self, f: impl Fn(Self::Structure) -> O) -> Self::Store<O> {
        match self {
            ParamOrConcrete::Concrete(c) => ParamOrConcrete::Concrete(c.map_structure(f)),
            ParamOrConcrete::Param(p) => ParamOrConcrete::Param(p.map_structure(f)),
        }
    }

    fn map_structure_result<O: TensorStructure, Er>(
        self,
        f: impl Fn(Self::Structure) -> Result<O, Er>,
    ) -> std::result::Result<Self::Store<O>, Er> {
        todo!()
    }

    fn structure(&self) -> &Self::Structure {
        match self {
            ParamOrConcrete::Concrete(x) => x.structure(),
            ParamOrConcrete::Param(x) => x.structure(),
        }
    }

    fn map_same_structure(self, f: impl FnOnce(Self::Structure) -> Self::Structure) -> Self {
        match self {
            ParamOrConcrete::Concrete(x) => ParamOrConcrete::Concrete(x.map_same_structure(f)),
            ParamOrConcrete::Param(x) => ParamOrConcrete::Param(x.map_same_structure(f)),
        }
    }

    fn mut_structure(&mut self) -> &mut Self::Structure {
        match self {
            ParamOrConcrete::Concrete(x) => x.mut_structure(),
            ParamOrConcrete::Param(x) => x.mut_structure(),
        }
    }

    fn scalar(self) -> Option<Self::Scalar> {
        match self {
            ParamOrConcrete::Concrete(x) => x.scalar().map(ConcreteOrParam::Concrete),
            ParamOrConcrete::Param(x) => x.scalar().map(ConcreteOrParam::Param),
        }
    }

    fn scalar_ref(&self) -> Option<Self::ScalarRef<'_>> {
        match self {
            ParamOrConcrete::Concrete(x) => x.scalar_ref().map(ConcreteOrParamRef::Concrete),
            ParamOrConcrete::Param(x) => x.scalar_ref().map(ConcreteOrParamRef::Param),
        }
    }
}

impl<C, S> SetTensorData for ParamOrConcrete<C, S>
where
    C: HasStructure<Structure = S> + Clone + SetTensorData,
    S: TensorStructure + Clone,
{
    type SetData = ConcreteOrParam<C::SetData>;

    fn set(
        &mut self,
        indices: &[crate::structure::concrete_index::ConcreteIndex],
        value: Self::SetData,
    ) -> Result<()> {
        match self {
            ParamOrConcrete::Concrete(x) => x.set(
                indices,
                value
                    .try_into_concrete()
                    .map_err(|r| anyhow!(r.to_string()))?,
            ),
            ParamOrConcrete::Param(x) => x.set(
                indices,
                value.try_into_param().map_err(|r| anyhow!(r.to_string()))?,
            ),
        }
    }

    fn set_flat(&mut self, index: FlatIndex, value: Self::SetData) -> Result<()> {
        match self {
            ParamOrConcrete::Concrete(x) => x.set_flat(
                index,
                value
                    .try_into_concrete()
                    .map_err(|r| anyhow!(r.to_string()))?,
            ),
            ParamOrConcrete::Param(x) => x.set_flat(
                index,
                value.try_into_param().map_err(|r| anyhow!(r.to_string()))?,
            ),
        }
    }
}

impl<C, S> GetTensorData for ParamOrConcrete<C, S>
where
    C: HasStructure<Structure = S> + Clone + GetTensorData<GetDataOwned: Clone>,
    S: TensorStructure + Clone,
{
    type GetDataRef<'a>
        = ConcreteOrParamView<'a, C::GetDataRef<'a>>
    where
        Self: 'a;
    type GetDataRefMut<'a>
        = ConcreteOrParamViewMut<'a, C::GetDataRefMut<'a>>
    where
        Self: 'a;
    type GetDataOwned = ConcreteOrParam<C::GetDataOwned>;
    fn get_ref<D: AsRef<[ConcreteIndex]>>(&self, indices: D) -> Result<Self::GetDataRef<'_>> {
        match self {
            ParamOrConcrete::Concrete(x) => x.get_ref(indices).map(ConcreteOrParamView::Concrete),
            ParamOrConcrete::Param(x) => x.get_ref(indices).map(ConcreteOrParamView::Param),
        }
    }

    fn get_ref_linear(&self, index: FlatIndex) -> Option<Self::GetDataRef<'_>> {
        match self {
            ParamOrConcrete::Concrete(x) => {
                x.get_ref_linear(index).map(ConcreteOrParamView::Concrete)
            }
            ParamOrConcrete::Param(x) => x.get_ref_linear(index).map(ConcreteOrParamView::Param),
        }
    }

    fn get_mut_linear(&mut self, index: FlatIndex) -> Option<Self::GetDataRefMut<'_>> {
        match self {
            ParamOrConcrete::Concrete(x) => x
                .get_mut_linear(index)
                .map(ConcreteOrParamViewMut::Concrete),
            ParamOrConcrete::Param(x) => x.get_mut_linear(index).map(ConcreteOrParamViewMut::Param),
        }
    }

    fn get_owned<D: AsRef<[ConcreteIndex]>>(&self, indices: D) -> Result<Self::GetDataOwned>
    where
        Self::GetDataOwned: Clone,
    {
        match self {
            ParamOrConcrete::Concrete(x) => x.get_owned(indices).map(ConcreteOrParam::Concrete),
            ParamOrConcrete::Param(x) => x.get_owned(indices).map(ConcreteOrParam::Param),
        }
    }

    fn get_owned_linear(&self, index: FlatIndex) -> Option<Self::GetDataOwned>
    where
        Self::GetDataOwned: Clone,
    {
        match self {
            ParamOrConcrete::Concrete(x) => {
                x.get_owned_linear(index).map(ConcreteOrParam::Concrete)
            }
            ParamOrConcrete::Param(x) => x.get_owned_linear(index).map(ConcreteOrParam::Param),
        }
    }
}

impl<T> From<Atom> for ConcreteOrParam<T> {
    fn from(value: Atom) -> Self {
        ConcreteOrParam::Param(value)
    }
}

impl<T: Into<Atom>> From<ConcreteOrParam<T>> for Atom {
    fn from(value: ConcreteOrParam<T>) -> Self {
        match value {
            ConcreteOrParam::Concrete(x) => x.into(),
            ConcreteOrParam::Param(x) => x,
        }
    }
}

impl<T: Into<Coefficient>> From<RealOrComplex<T>> for Atom {
    fn from(value: RealOrComplex<T>) -> Self {
        match value {
            RealOrComplex::Real(x) => Atom::new_num(x),
            RealOrComplex::Complex(x) => {
                let (re, im) = (Atom::new_num(x.re), Atom::new_num(x.im));
                let i = Atom::new_var(Atom::I);
                re + im * i
            }
        }
    }
}

impl<C, S> ScalarTensor for ParamOrConcrete<C, S>
where
    C: HasStructure<Structure = S> + Clone + ScalarTensor + TensorStructure,
    C::Indexed: HasStructure + TensorStructure,
    S: TensorStructure + ScalarStructure + Clone,
{
    fn new_scalar(scalar: Self::Scalar) -> Self {
        match scalar {
            ConcreteOrParam::Concrete(x) => ParamOrConcrete::Concrete(C::new_scalar(x)),
            ConcreteOrParam::Param(x) => ParamOrConcrete::Param(ParamTensor::new_scalar(x)),
        }
    }
}

impl<C, S> TracksCount for ParamOrConcrete<C, S>
where
    C: TracksCount + HasStructure<Structure = S> + Clone,
    S: TensorStructure + TracksCount + Clone,
{
    fn contractions_num(&self) -> usize {
        match self {
            ParamOrConcrete::Concrete(x) => x.contractions_num(),
            ParamOrConcrete::Param(x) => x.contractions_num(),
        }
    }
}

impl<C, S> HasName for ParamOrConcrete<C, S>
where
    C: HasName + HasStructure<Structure = S> + Clone,
    S: TensorStructure + HasName<Name = C::Name, Args = C::Args> + Clone,
{
    type Args = C::Args;
    type Name = C::Name;

    fn args(&self) -> Option<Self::Args> {
        match self {
            ParamOrConcrete::Concrete(x) => x.args(),
            ParamOrConcrete::Param(x) => x.args(),
        }
    }

    fn name(&self) -> Option<Self::Name> {
        match self {
            ParamOrConcrete::Concrete(x) => x.name(),
            ParamOrConcrete::Param(x) => x.name(),
        }
    }

    fn set_name(&mut self, name: Self::Name) {
        match self {
            ParamOrConcrete::Concrete(x) => x.set_name(name),
            ParamOrConcrete::Param(x) => x.set_name(name),
        }
    }
}

impl<C: IteratableTensor + Clone, S: TensorStructure + Clone> IteratableTensor
    for ParamOrConcrete<C, S>
where
    C: HasStructure<Structure = S> + TensorStructure,
    C::Indexed: HasStructure<Structure = S::Indexed>,
    S: TensorStructure,
{
    type Data<'a>
        = AtomViewOrConcrete<'a, C::Data<'a>>
    where
        Self: 'a;

    fn iter_flat(&self) -> impl Iterator<Item = (FlatIndex, Self::Data<'_>)> {
        match self {
            ParamOrConcrete::Concrete(x) => IteratorEnum::A(
                x.iter_flat()
                    .map(|(i, x)| (i, AtomViewOrConcrete::Concrete(x))),
            ),
            ParamOrConcrete::Param(x) => {
                IteratorEnum::B(x.iter_flat().map(|(i, x)| (i, AtomViewOrConcrete::Atom(x))))
            }
        }
    }

    fn iter_expanded(&self) -> impl Iterator<Item = (ExpandedIndex, Self::Data<'_>)> {
        match self {
            ParamOrConcrete::Concrete(x) => IteratorEnum::A(
                x.iter_expanded()
                    .map(|(i, x)| (i, AtomViewOrConcrete::Concrete(x))),
            ),
            ParamOrConcrete::Param(x) => IteratorEnum::B(
                x.iter_expanded()
                    .map(|(i, x)| (i, AtomViewOrConcrete::Atom(x))),
            ),
        }
    }
}

pub type MixedTensor<T = f64, S = NamedStructure<Symbol, Vec<Atom>>> =
    ParamOrConcrete<RealOrComplexTensor<T, S>, S>;

impl<T: Clone, S: TensorStructure + Clone> PartialEq<MixedTensor<T, S>> for MixedTensor<T, S> {
    fn eq(&self, other: &MixedTensor<T, S>) -> bool {
        matches!(
            (self, other),
            (
                MixedTensor::Concrete(RealOrComplexTensor::Real(_)),
                MixedTensor::Concrete(RealOrComplexTensor::Real(_)),
            ) | (
                MixedTensor::Concrete(RealOrComplexTensor::Complex(_)),
                MixedTensor::Concrete(RealOrComplexTensor::Complex(_)),
            ) | (MixedTensor::Param(_), MixedTensor::Param(_))
        )
    }
}

impl<T: Clone, S: TensorStructure + Clone> Eq for MixedTensor<T, S> {}

impl<T: Clone, S: TensorStructure + Clone> PartialOrd<MixedTensor<T, S>> for MixedTensor<T, S> {
    fn partial_cmp(&self, other: &MixedTensor<T, S>) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Clone, S: TensorStructure + Clone> Ord for MixedTensor<T, S> {
    fn cmp(&self, other: &MixedTensor<T, S>) -> std::cmp::Ordering {
        match (self, other) {
            (
                MixedTensor::Concrete(RealOrComplexTensor::Real(_)),
                MixedTensor::Concrete(RealOrComplexTensor::Real(_)),
            ) => std::cmp::Ordering::Equal,
            (
                MixedTensor::Concrete(RealOrComplexTensor::Real(_)),
                MixedTensor::Concrete(RealOrComplexTensor::Complex(_)),
            ) => std::cmp::Ordering::Less,
            (MixedTensor::Concrete(RealOrComplexTensor::Real(_)), MixedTensor::Param(_)) => {
                std::cmp::Ordering::Less
            }
            (
                MixedTensor::Concrete(RealOrComplexTensor::Complex(_)),
                MixedTensor::Concrete(RealOrComplexTensor::Real(_)),
            ) => std::cmp::Ordering::Greater,
            (
                MixedTensor::Concrete(RealOrComplexTensor::Complex(_)),
                MixedTensor::Concrete(RealOrComplexTensor::Complex(_)),
            ) => std::cmp::Ordering::Equal,
            (MixedTensor::Concrete(RealOrComplexTensor::Complex(_)), MixedTensor::Param(_)) => {
                std::cmp::Ordering::Less
            }
            (MixedTensor::Param(_), MixedTensor::Concrete(RealOrComplexTensor::Real(_))) => {
                std::cmp::Ordering::Greater
            }
            (MixedTensor::Param(_), MixedTensor::Concrete(RealOrComplexTensor::Complex(_))) => {
                std::cmp::Ordering::Greater
            }
            (MixedTensor::Param(_), MixedTensor::Param(_)) => std::cmp::Ordering::Equal,
        }
    }
}

impl<'a, I: TensorStructure + Clone + 'a, T: Clone> MixedTensor<T, I> {
    pub fn evaluate_real<A: AtomCore + KeyLookup, F: Fn(&Rational) -> T + Copy>(
        &mut self,
        coeff_map: F,
        const_map: &HashMap<A, T>,
        function_map: &HashMap<Symbol, EvaluationFn<A, T>>,
    ) where
        T: Real + for<'c> From<&'c Rational>,
    {
        let content = match self {
            MixedTensor::Param(x) => Some(x),
            _ => None,
        };

        if let Some(x) = content {
            *self = MixedTensor::Concrete(RealOrComplexTensor::Real(
                x.evaluate(coeff_map, const_map, function_map).unwrap(),
            ));
        }
    }

    pub fn evaluate_complex<A: AtomCore + KeyLookup, F: Fn(&Rational) -> SymComplex<T> + Copy>(
        &mut self,
        coeff_map: F,
        const_map: &HashMap<A, SymComplex<T>>,
        function_map: &HashMap<Symbol, EvaluationFn<A, SymComplex<T>>>,
    ) where
        T: Real + for<'c> From<&'c Rational>,
        SymComplex<T>: Real + for<'c> From<&'c Rational>,
    {
        let content = match self {
            MixedTensor::Param(x) => Some(x),
            _ => None,
        };

        if let Some(x) = content {
            *self = MixedTensor::Concrete(RealOrComplexTensor::Complex(
                x.evaluate(coeff_map, const_map, function_map)
                    .unwrap()
                    .map_data(|c| c.into()),
            ));
        }
    }
}

impl<I> DenseTensor<Atom, I>
where
    I: Clone + TensorStructure,
{
    pub fn append_const_map<'a, 'b, T, U>(
        &'a self,
        data: &DenseTensor<T, I>,
        const_map: &mut HashMap<AtomView<'b>, U>,
    ) where
        I: TensorStructure,
        T: Copy,
        U: From<T>,
        'a: 'b,
    {
        for ((i, a), (j, v)) in self.flat_iter().zip(data.flat_iter()) {
            assert_eq!(i, j);
            const_map.insert(a.as_view(), (*v).into());
        }
    }
}

impl<S> HasStructure for ParamTensor<S>
where
    S: TensorStructure,
{
    type Structure = S;
    type Scalar = Atom;
    type ScalarRef<'a>
        = AtomView<'a>
    where
        Self: 'a;
    type Store<U>
        = ParamTensor<U>
    where
        U: TensorStructure;

    fn map_structure<Ss>(self, f: impl Fn(Self::Structure) -> Ss) -> Self::Store<Ss>
    where
        Ss: TensorStructure,
    {
        ParamTensor {
            param_type: self.param_type,
            tensor: self.tensor.map_structure(f),
        }
    }

    fn map_structure_result<O: TensorStructure, Er>(
        self,
        f: impl Fn(Self::Structure) -> Result<O, Er>,
    ) -> std::result::Result<Self::Store<O>, Er> {
        Ok(ParamTensor {
            param_type: self.param_type,
            tensor: self.tensor.map_structure_result(f)?,
        })
    }

    fn structure(&self) -> &Self::Structure {
        self.tensor.structure()
    }

    fn mut_structure(&mut self) -> &mut Self::Structure {
        self.tensor.mut_structure()
    }

    fn map_same_structure(self, f: impl FnOnce(Self::Structure) -> Self::Structure) -> Self {
        ParamTensor {
            tensor: self.tensor.map_same_structure(f),
            ..self
        }
    }

    fn scalar(self) -> Option<Self::Scalar> {
        self.tensor.scalar()
    }

    fn scalar_ref(&self) -> Option<Self::ScalarRef<'_>> {
        self.tensor.scalar_ref().map(|a| a.as_view())
    }
}

impl<S: TensorStructure> GetTensorData for ParamTensor<S> {
    type GetDataRef<'a>
        = AtomView<'a>
    where
        Self: 'a;
    type GetDataRefMut<'a>
        = &'a mut Atom
    where
        Self: 'a;
    type GetDataOwned = Atom;
    fn get_ref<C: AsRef<[ConcreteIndex]>>(&self, indices: C) -> Result<Self::GetDataRef<'_>> {
        self.tensor.get_ref(indices).map(|x| x.as_view())
    }

    fn get_ref_linear(&self, index: FlatIndex) -> Option<Self::GetDataRef<'_>> {
        self.tensor.get_ref_linear(index).map(|x| x.as_view())
    }

    fn get_mut_linear(&mut self, index: FlatIndex) -> Option<&mut Atom> {
        self.tensor.get_mut_linear(index)
    }

    fn get_owned<C: AsRef<[ConcreteIndex]>>(&self, indices: C) -> Result<Self::GetDataOwned>
    where
        Self::GetDataOwned: Clone,
    {
        self.tensor.get_owned(indices)
    }

    fn get_owned_linear(&self, index: FlatIndex) -> Option<Self::GetDataOwned>
    where
        Self::GetDataOwned: Clone,
    {
        self.tensor.get_owned_linear(index)
    }
}

impl<S: TensorStructure> SetTensorData for ParamTensor<S> {
    type SetData = Atom;

    fn set(
        &mut self,
        indices: &[crate::structure::concrete_index::ConcreteIndex],
        value: Self::SetData,
    ) -> Result<()> {
        self.tensor.set(indices, value)
    }

    fn set_flat(&mut self, index: FlatIndex, value: Self::SetData) -> Result<()> {
        self.tensor.set_flat(index, value)
    }
}

impl<S> ScalarTensor for ParamTensor<S>
where
    S: TensorStructure + ScalarStructure,
{
    fn new_scalar(scalar: Self::Scalar) -> Self {
        ParamTensor {
            tensor: DataTensor::new_scalar(scalar),
            param_type: ParamOrComposite::Composite,
        }
    }
}

impl<S> TracksCount for ParamTensor<S>
where
    S: TensorStructure + TracksCount,
{
    fn contractions_num(&self) -> usize {
        self.tensor.contractions_num()
    }
}

// pub type MixedTensors = MixedTensor<HistoryStructure<Symbol>>;

impl<I, T: Clone> From<DenseTensor<T, I>> for MixedTensor<T, I>
where
    I: TensorStructure + Clone,
{
    fn from(other: DenseTensor<T, I>) -> Self {
        MixedTensor::Concrete(RealOrComplexTensor::Real(DataTensor::Dense(other)))
    }
}

impl<I, T: Clone> From<SparseTensor<T, I>> for MixedTensor<T, I>
where
    I: TensorStructure + Clone,
{
    fn from(other: SparseTensor<T, I>) -> Self {
        MixedTensor::Concrete(RealOrComplexTensor::Real(DataTensor::Sparse(other)))
    }
}

impl<I, T: Clone> From<DenseTensor<Complex<T>, I>> for MixedTensor<T, I>
where
    I: TensorStructure + Clone,
{
    fn from(other: DenseTensor<Complex<T>, I>) -> Self {
        MixedTensor::Concrete(RealOrComplexTensor::Complex(DataTensor::Dense(other)))
    }
}

impl<I, T: Clone> From<SparseTensor<Complex<T>, I>> for MixedTensor<T, I>
where
    I: TensorStructure + Clone,
{
    fn from(other: SparseTensor<Complex<T>, I>) -> Self {
        MixedTensor::Concrete(RealOrComplexTensor::Complex(DataTensor::Sparse(other)))
    }
}

impl<I, T: Clone> MixedTensor<T, I>
where
    I: TensorStructure + Clone,
{
    pub fn param(other: DataTensor<Atom, I>) -> Self {
        MixedTensor::Param(ParamTensor::param(other))
    }

    pub fn composite(other: DataTensor<Atom, I>) -> Self {
        MixedTensor::Param(ParamTensor::composite(other))
    }
}

impl<I> Trace for ParamTensor<I>
where
    I: TensorStructure + Clone + StructureContract,
{
    fn internal_contract(&self) -> Self {
        ParamTensor {
            tensor: self.tensor.internal_contract(),
            param_type: self.param_type,
        }
    }
}

impl<I> Contract<ParamTensor<I>> for ParamTensor<I>
where
    I: TensorStructure + Clone + StructureContract,
{
    type LCM = ParamTensor<I>;
    fn contract(&self, other: &ParamTensor<I>) -> Result<Self::LCM, ContractionError> {
        let s = self.tensor.contract(&other.tensor)?;

        match (self.param_type, other.param_type) {
            (ParamOrComposite::Param, ParamOrComposite::Param) => Ok(ParamTensor::param(s)),
            (ParamOrComposite::Composite, ParamOrComposite::Composite) => {
                Ok(ParamTensor::composite(s))
            }
            (ParamOrComposite::Param, ParamOrComposite::Composite) => Ok(ParamTensor::composite(s)),
            (ParamOrComposite::Composite, ParamOrComposite::Param) => Ok(ParamTensor::composite(s)),
        }
    }
}

impl<I, T> Trace for ParamOrConcrete<DataTensor<T, I>, I>
where
    I: TensorStructure + Clone + StructureContract,
    T: ContractableWith<Atom, Out = Atom>
        + ContractableWith<T, Out = T>
        + Clone
        + FallibleMul<Output = T>
        + FallibleAddAssign<T>
        + FallibleSubAssign<T>
        + RefZero
        + IsZero,
{
    fn internal_contract(&self) -> Self {
        match self {
            ParamOrConcrete::Param(p) => ParamOrConcrete::Param(p.internal_contract()),
            ParamOrConcrete::Concrete(c) => ParamOrConcrete::Concrete(c.internal_contract()),
        }
    }
}

impl<I, T> Contract<ParamOrConcrete<DataTensor<T, I>, I>> for ParamOrConcrete<DataTensor<T, I>, I>
where
    I: TensorStructure + Clone + StructureContract,
    T: ContractableWith<Atom, Out = Atom>
        + ContractableWith<T, Out = T>
        + Clone
        + FallibleMul<Output = T>
        + FallibleAddAssign<T>
        + FallibleSubAssign<T>
        + RefZero
        + IsZero,
    Atom: TrySmallestUpgrade<T, LCM = Atom>, // Atom: ContractableWith<T, Out = Atom> + ContractableWith<Atom, Out = Atom>,
{
    type LCM = ParamOrConcrete<DataTensor<T, I>, I>;
    fn contract(
        &self,
        other: &ParamOrConcrete<DataTensor<T, I>, I>,
    ) -> Result<Self::LCM, ContractionError> {
        match (self, other) {
            (ParamOrConcrete::Param(s), ParamOrConcrete::Param(o)) => {
                Ok(ParamOrConcrete::Param(s.contract(o)?))
            }
            (ParamOrConcrete::Param(s), ParamOrConcrete::Concrete(o)) => match s.param_type {
                ParamOrComposite::Composite => Ok(ParamOrConcrete::Param(ParamTensor::composite(
                    s.tensor.contract(o)?,
                ))),
                ParamOrComposite::Param => Ok(ParamOrConcrete::Param(ParamTensor::composite(
                    s.tensor.contract(o)?,
                ))),
            },
            (ParamOrConcrete::Concrete(s), ParamOrConcrete::Param(o)) => match o.param_type {
                ParamOrComposite::Composite => Ok(ParamOrConcrete::Param(ParamTensor::composite(
                    s.contract(&o.tensor)?,
                ))),
                ParamOrComposite::Param => Ok(ParamOrConcrete::Param(ParamTensor::composite(
                    s.contract(&o.tensor)?,
                ))),
            },
            (ParamOrConcrete::Concrete(s), ParamOrConcrete::Concrete(o)) => {
                Ok(ParamOrConcrete::Concrete(s.contract(o)?))
            }
        }
    }
}
impl<I, T> Trace for ParamOrConcrete<RealOrComplexTensor<T, I>, I>
where
    I: TensorStructure + Clone + StructureContract,
    T: ContractableWith<Atom, Out = Atom>
        + ContractableWith<T, Out = T>
        + ContractableWith<Complex<T>, Out = Complex<T>>
        + Clone
        + FallibleMul<Output = T>
        + FallibleAddAssign<T>
        + FallibleSubAssign<T>
        + RefZero
        + IsZero,
    Complex<T>: ContractableWith<Atom, Out = Atom>
        + ContractableWith<T, Out = Complex<T>>
        + ContractableWith<Complex<T>, Out = Complex<T>>
        + Clone
        + FallibleMul<Output = Complex<T>>
        + FallibleAddAssign<Complex<T>>
        + FallibleSubAssign<Complex<T>>
        + RefZero
        + IsZero,
    Atom: TrySmallestUpgrade<T, LCM = Atom> + TrySmallestUpgrade<Complex<T>, LCM = Atom>,
{
    fn internal_contract(&self) -> Self {
        match self {
            ParamOrConcrete::Param(p) => ParamOrConcrete::Param(p.internal_contract()),
            ParamOrConcrete::Concrete(c) => ParamOrConcrete::Concrete(c.internal_contract()),
        }
    }
}

impl<I, T> Contract<ParamOrConcrete<RealOrComplexTensor<T, I>, I>>
    for ParamOrConcrete<RealOrComplexTensor<T, I>, I>
where
    I: TensorStructure + Clone + StructureContract,
    T: ContractableWith<Atom, Out = Atom>
        + ContractableWith<T, Out = T>
        + ContractableWith<Complex<T>, Out = Complex<T>>
        + Clone
        + FallibleMul<Output = T>
        + FallibleAddAssign<T>
        + FallibleSubAssign<T>
        + RefZero
        + IsZero,
    Complex<T>: ContractableWith<Atom, Out = Atom>
        + ContractableWith<T, Out = Complex<T>>
        + ContractableWith<Complex<T>, Out = Complex<T>>
        + Clone
        + FallibleMul<Output = Complex<T>>
        + FallibleAddAssign<Complex<T>>
        + FallibleSubAssign<Complex<T>>
        + RefZero
        + IsZero,
    Atom: TrySmallestUpgrade<T, LCM = Atom> + TrySmallestUpgrade<Complex<T>, LCM = Atom>,
{
    type LCM = ParamOrConcrete<RealOrComplexTensor<T, I>, I>;
    fn contract(
        &self,
        other: &ParamOrConcrete<RealOrComplexTensor<T, I>, I>,
    ) -> Result<Self::LCM, ContractionError> {
        match (self, other) {
            (ParamOrConcrete::Param(s), ParamOrConcrete::Param(o)) => {
                Ok(ParamOrConcrete::Param(s.contract(o)?))
            }
            (ParamOrConcrete::Param(s), ParamOrConcrete::Concrete(o)) => match (s.param_type, o) {
                (ParamOrComposite::Composite, RealOrComplexTensor::Real(o)) => Ok(
                    ParamOrConcrete::Param(ParamTensor::composite(s.tensor.contract(o)?)),
                ),
                (ParamOrComposite::Composite, RealOrComplexTensor::Complex(o)) => Ok(
                    ParamOrConcrete::Param(ParamTensor::composite(s.tensor.contract(o)?)),
                ),
                (ParamOrComposite::Param, RealOrComplexTensor::Real(o)) => Ok(
                    ParamOrConcrete::Param(ParamTensor::composite(s.tensor.contract(o)?)),
                ),
                (ParamOrComposite::Param, RealOrComplexTensor::Complex(o)) => Ok(
                    ParamOrConcrete::Param(ParamTensor::composite(s.tensor.contract(o)?)),
                ),
            },
            (ParamOrConcrete::Concrete(s), ParamOrConcrete::Param(o)) => match (o.param_type, s) {
                (ParamOrComposite::Composite, RealOrComplexTensor::Real(s)) => Ok(
                    ParamOrConcrete::Param(ParamTensor::composite(o.tensor.contract(s)?)),
                ),
                (ParamOrComposite::Composite, RealOrComplexTensor::Complex(s)) => Ok(
                    ParamOrConcrete::Param(ParamTensor::composite(o.tensor.contract(s)?)),
                ),
                (ParamOrComposite::Param, RealOrComplexTensor::Real(s)) => Ok(
                    ParamOrConcrete::Param(ParamTensor::composite(o.tensor.contract(s)?)),
                ),
                (ParamOrComposite::Param, RealOrComplexTensor::Complex(s)) => Ok(
                    ParamOrConcrete::Param(ParamTensor::composite(o.tensor.contract(s)?)),
                ),
            },
            (ParamOrConcrete::Concrete(s), ParamOrConcrete::Concrete(o)) => {
                Ok(ParamOrConcrete::Concrete(s.contract(o)?))
            }
        }
    }
}

pub type EvalTreeTensor<T, S> = EvalTensor<EvalTree<T>, S>;

pub type EvalTreeTensorSet<T, S> = EvalTensorSet<(EvalTree<T>, Option<Vec<Expression<T>>>), S>;

impl<S: Clone + TensorStructure> EvalTreeTensorSet<Rational, S> {
    pub fn horner_scheme(&mut self) {
        self.eval.0.horner_scheme()
    }
    pub fn optimize_horner_scheme(&mut self, iterations: usize, n_cores: usize, verbose: bool) {
        let scheme = self.eval.1.take();
        self.eval.1 = Some(
            self.eval
                .0
                .optimize_horner_scheme(iterations, n_cores, scheme, verbose),
        );
    }
}

impl<T, S: TensorStructure> EvalTreeTensorSet<T, S> {
    pub fn map_coeff<T2, F: Fn(&T) -> T2>(&self, f: &F) -> EvalTreeTensorSet<T2, S>
    where
        T: Clone + PartialEq,
        S: Clone,
    {
        EvalTreeTensorSet {
            eval: (self.eval.0.map_coeff(f), None),
            tensors: self.tensors.clone(),
            size: self.size,
        }
        // self.map_data_ref(|x| x.map_coeff(f))
    }

    #[allow(clippy::type_complexity)]
    pub fn linearize(
        self,
        cpe_rounds: Option<usize>,
    ) -> EvalTensorSet<(ExpressionEvaluator<T>, Option<Vec<Expression<T>>>), S>
    where
        T: Clone + Default + PartialEq,
    {
        EvalTensorSet {
            eval: (self.eval.0.linearize(cpe_rounds), self.eval.1),
            tensors: self.tensors,
            size: self.size,
        }
    }

    pub fn common_subexpression_elimination(&mut self)
    where
        T: Debug + Hash + Eq + Ord + Clone + Default,
    {
        self.eval.0.common_subexpression_elimination()
    }

    pub fn evaluate(&mut self, params: &[T]) -> TensorSet<DataTensor<T, S>>
    where
        T: Real,
        S: TensorStructure + Clone,
    {
        let zero = params[0].zero();

        let mut elements = vec![zero; self.size];
        self.eval.0.evaluate(params, &mut elements);

        match &self.tensors {
            TensorsOrScalars::Scalars => TensorSet::Scalars(elements),
            TensorsOrScalars::Tensors(t) => {
                let mut out_tensors = Vec::with_capacity(t.len());
                for t in t.iter() {
                    out_tensors.push(t.map_data_ref(|&i| elements[i].clone()));
                }
                TensorSet::Tensors(out_tensors)
            }
        }
    }
}

impl<S: Clone> EvalTreeTensor<Rational, S> {
    pub fn horner_scheme(&mut self) {
        self.eval.horner_scheme()
    }

    pub fn optimize_horner_scheme(
        &mut self,
        iterations: usize,
        n_cores: usize,
        scheme: Option<Vec<Expression<Rational>>>,
        verbose: bool,
    ) -> Vec<Expression<Rational>> {
        self.eval
            .optimize_horner_scheme(iterations, n_cores, scheme, verbose)
    }

    pub fn optimize(
        &mut self,
        iterations: usize,
        n_cores: usize,
        start_scheme: Option<Vec<Expression<Rational>>>,
        verbose: bool,
    ) -> EvalTensor<ExpressionEvaluator<Rational>, S> {
        let _ = self.optimize_horner_scheme(iterations, n_cores, start_scheme, verbose);
        self.common_subexpression_elimination();
        self.clone().linearize(None)
    }
}

impl<S: Clone> EvalTreeTensor<Rational, S> {
    pub fn from_dense(
        dense: &DenseTensor<Atom, S>,
        fn_map: &FunctionMap,
        params: &[Atom],
    ) -> Result<Self, String> {
        let atomviews: Vec<AtomView> = dense.data.iter().map(|a| a.as_view()).collect();
        let eval = AtomView::to_eval_tree_multiple(&atomviews, fn_map, params)?;

        Ok(EvalTreeTensor {
            eval,
            indexmap: None,
            structure: dense.structure.clone(),
        })
    }

    pub fn from_sparse(
        dense: &SparseTensor<Atom, S>,
        fn_map: &FunctionMap,
        params: &[Atom],
    ) -> Result<Self, String> {
        let atomviews: (Vec<FlatIndex>, Vec<AtomView>) = dense
            .elements
            .iter()
            .map(|(k, a)| (*k, a.as_view()))
            .unzip();
        let eval = AtomView::to_eval_tree_multiple(&atomviews.1, fn_map, params)?;

        Ok(EvalTreeTensor {
            eval,
            indexmap: Some(atomviews.0),
            structure: dense.structure.clone(),
        })
    }

    pub fn from_data(
        data: &DataTensor<Atom, S>,
        fn_map: &FunctionMap,
        params: &[Atom],
    ) -> Result<Self, String>
    where
        S: TensorStructure,
    {
        match data {
            DataTensor::Dense(d) => Self::from_dense(d, fn_map, params),
            DataTensor::Sparse(s) => Self::from_sparse(s, fn_map, params),
        }
    }
}

impl<S: Clone, T> EvalTreeTensor<T, S> {
    pub fn map_coeff<T2, F: Fn(&T) -> T2>(&self, f: &F) -> EvalTreeTensor<T2, S>
    where
        T: Clone + PartialEq,
    {
        EvalTreeTensor {
            eval: self.eval.map_coeff(f),
            indexmap: self.indexmap.clone(),
            structure: self.structure.clone(),
        }
        // self.map_data_ref(|x| x.map_coeff(f))
    }

    pub fn linearize(self, cpe_rounds: Option<usize>) -> EvalTensor<ExpressionEvaluator<T>, S>
    where
        T: Clone + Default + PartialEq,
    {
        EvalTensor {
            eval: self.eval.linearize(cpe_rounds),
            structure: self.structure,
            indexmap: self.indexmap,
        }
    }

    pub fn common_subexpression_elimination(&mut self)
    where
        T: Debug + Hash + Eq + Ord + Clone + Default,
    {
        self.eval.common_subexpression_elimination()
    }

    pub fn evaluate(&mut self, params: &[T]) -> DataTensor<T, S>
    where
        T: Real,
        S: TensorStructure,
    {
        let zero = params[0].zero();
        if let Some(ref indexmap) = self.indexmap {
            let mut elements = vec![zero; indexmap.len()];
            self.eval.evaluate(params, &mut elements);
            let s = SparseTensor {
                elements: indexmap.iter().cloned().zip(elements.drain(0..)).collect(),
                structure: self.structure.clone(),
            };
            DataTensor::Sparse(s)
        } else {
            let mut out_data = DenseTensor::repeat(self.structure.clone(), zero.clone());
            self.eval.evaluate(params, &mut out_data.data);
            DataTensor::Dense(out_data)
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalTensor<T, S> {
    eval: T,
    indexmap: Option<Vec<FlatIndex>>,
    structure: S,
}

impl<T, S: TensorStructure + Clone> EvalTensor<T, S> {
    pub fn usize_tensor(&self, shift: usize) -> DataTensor<usize, S> {
        if let Some(ref indexmap) = self.indexmap {
            let mut sparse_tensor = SparseTensor::empty(self.structure.clone());
            for (i, idx) in indexmap.iter().enumerate() {
                sparse_tensor.elements.insert(*idx, shift + i);
            }
            DataTensor::Sparse(sparse_tensor)
        } else {
            let data: Vec<usize> = (shift..shift + self.structure.size().unwrap()).collect();
            let dense_tensor = DenseTensor::from_data(data, self.structure.clone()).unwrap();
            DataTensor::Dense(dense_tensor)
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TensorsOrScalars<T, S: TensorStructure> {
    Tensors(Vec<DataTensor<T, S>>),
    Scalars,
}

impl<T, S: TensorStructure> TensorsOrScalars<T, S> {
    pub fn push(&mut self, tensor: DataTensor<T, S>) {
        if tensor.is_scalar() {
            if let TensorsOrScalars::Tensors(_) = self {
                panic!("Trying to push a scalar to a list of tensors")
            }
        } else {
            match self {
                TensorsOrScalars::Tensors(t) => t.push(tensor),
                TensorsOrScalars::Scalars => {
                    panic!("Trying to push a tensor to a list of scalars")
                }
            }
        }
    }

    pub fn len(&self) -> usize {
        match self {
            TensorsOrScalars::Tensors(t) => t.len(),
            TensorsOrScalars::Scalars => 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            TensorsOrScalars::Tensors(t) => t.is_empty(),
            TensorsOrScalars::Scalars => true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalTensorSet<T, S: TensorStructure> {
    pub tensors: TensorsOrScalars<usize, S>,
    eval: T,
    size: usize, //
}

impl<S: TensorStructure, T> EvalTensorSet<T, S> {
    pub fn len(&self) -> usize {
        match &self.tensors {
            TensorsOrScalars::Tensors(t) => t.len(),
            TensorsOrScalars::Scalars => self.size,
        }
    }

    pub fn is_empty(&self) -> bool {
        match &self.tensors {
            TensorsOrScalars::Tensors(t) => t.is_empty(),
            TensorsOrScalars::Scalars => self.size == 0,
        }
    }
}

impl<T, S> TensorStructure for EvalTensor<T, S>
where
    S: TensorStructure,
{
    // type R = <T::Structure as TensorStructure>::R;
    type Indexed = EvalTensor<T, S::Indexed>;
    type Slot = S::Slot;

    fn reindex(self, indices: &[AbstractIndex]) -> Result<Self::Indexed, StructureError> {
        self.map_structure_result(|s| s.reindex(indices))
    }

    fn dual(self) -> Self {
        self.map_same_structure(|s| s.dual())
    }

    delegate! {
        to self.structure() {
            fn external_reps_iter(&self)-> impl Iterator<Item = Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn external_indices_iter(&self)-> impl Iterator<Item = AbstractIndex>;
            fn external_dims_iter(&self)-> impl Iterator<Item = Dimension>;
            fn external_structure_iter(&self)-> impl Iterator<Item = Self::Slot>;
            fn get_slot(&self, i: usize)-> Option<Self::Slot>;
            fn get_rep(&self, i: usize)-> Option<Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn get_dim(&self, i: usize)-> Option<Dimension>;
            fn get_aind(&self, i: usize)-> Option<AbstractIndex>;
            fn order(&self)-> usize;
        }
    }
}

impl<T, S: TensorStructure> HasStructure for EvalTensor<T, S> {
    type Scalar = T;
    type ScalarRef<'a>
        = &'a T
    where
        Self: 'a;
    type Structure = S;
    type Store<U>
        = EvalTensor<T, U>
    where
        U: TensorStructure;

    fn map_structure<O: TensorStructure>(self, f: impl Fn(Self::Structure) -> O) -> Self::Store<O> {
        EvalTensor {
            structure: f(self.structure),
            indexmap: self.indexmap,
            eval: self.eval,
        }
    }

    fn map_structure_result<O: TensorStructure, Er>(
        self,
        f: impl Fn(Self::Structure) -> Result<O, Er>,
    ) -> std::result::Result<Self::Store<O>, Er> {
        Ok(EvalTensor {
            structure: f(self.structure)?,
            indexmap: self.indexmap,
            eval: self.eval,
        })
    }

    fn structure(&self) -> &Self::Structure {
        &self.structure
    }

    fn mut_structure(&mut self) -> &mut Self::Structure {
        &mut self.structure
    }

    fn map_same_structure(self, f: impl FnOnce(Self::Structure) -> Self::Structure) -> Self {
        Self {
            eval: self.eval,
            indexmap: self.indexmap,
            structure: f(self.structure),
        }
    }

    fn scalar(self) -> Option<Self::Scalar> {
        if self.is_scalar() {
            Some(self.eval)
        } else {
            None
        }
    }

    fn scalar_ref(&self) -> Option<Self::ScalarRef<'_>> {
        if self.is_scalar() {
            Some(&self.eval)
        } else {
            None
        }
    }
}

impl<S: TensorStructure + Clone>
    EvalTensorSet<
        (
            ExpressionEvaluator<Rational>,
            Option<Vec<Expression<Rational>>>,
        ),
        S,
    >
{
    pub fn push_optimize(
        &mut self,
        mut tensor: EvalTreeTensor<Rational, S>,
        cpe_rounds: Option<usize>,
        iterations: usize,
        n_cores: usize,
        verbose: bool,
    ) {
        let usize_tensor = tensor.usize_tensor(self.size);
        trace!("adding a tensor to the list of {} tensors", self.len());
        self.size += usize_tensor.actual_size();
        self.tensors.push(usize_tensor);
        self.eval.1 =
            Some(tensor.optimize_horner_scheme(iterations, n_cores, self.eval.1.take(), verbose));
        tensor.common_subexpression_elimination();

        self.eval
            .0
            .merge(tensor.linearize(cpe_rounds).eval, cpe_rounds)
            .unwrap();
    }
}

pub type LinearizedEvalTensor<T, S> = EvalTensor<ExpressionEvaluator<T>, S>;

impl<T, S> EvalTensor<ExpressionEvaluator<T>, S> {
    pub fn map_coeff<T2, F: Fn(&T) -> T2>(self, f: &F) -> LinearizedEvalTensor<T2, S>
    where
        T: Clone + PartialEq + Default,
        S: Clone,
    {
        LinearizedEvalTensor {
            eval: self.eval.map_coeff(f),
            indexmap: self.indexmap,
            structure: self.structure,
        }
        // self.map_data_ref(|x| x.map_coeff(f))
    }
    pub fn export_cpp(
        &self,
        filename: &str,
        function_name: &str,
        include_header: bool,
        inline_asm: InlineASM,
    ) -> Result<EvalTensor<SerializableExportedCode, S>, std::io::Error>
    where
        T: Display,
        S: Clone,
    {
        Ok(EvalTensor {
            eval: SerializableExportedCode::export_cpp(
                &self.eval,
                filename,
                function_name,
                include_header,
                inline_asm,
            )?,
            indexmap: self.indexmap.clone(),
            structure: self.structure.clone(),
        })
    }

    pub fn evaluate(&mut self, params: &[T]) -> DataTensor<T, S>
    where
        T: Real,
        S: TensorStructure + Clone,
    {
        let zero = params[0].zero();
        if let Some(ref indexmap) = self.indexmap {
            let mut elements = vec![zero; indexmap.len()];
            self.eval.evaluate(params, &mut elements);
            let s = SparseTensor {
                elements: indexmap.iter().cloned().zip(elements.drain(0..)).collect(),
                structure: self.structure.clone(),
            };
            DataTensor::Sparse(s)
        } else {
            let mut out_data = DenseTensor::repeat(self.structure.clone(), zero.clone());
            self.eval.evaluate(params, &mut out_data.data);
            DataTensor::Dense(out_data)
        }
    }
}

#[derive(Debug, Clone)]
pub struct SerializableCompiledEvaluator {
    evaluator: CompiledEvaluator,
    library_filename: String,
    function_name: String,
}

impl Serialize for SerializableCompiledEvaluator {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("SerializableCompiledEvaluator", 2)?;
        state.serialize_field("library_filename", &self.library_filename)?;
        state.serialize_field("function_name", &self.function_name)?;
        state.end()
    }
}

impl<'d> Deserialize<'d> for SerializableCompiledEvaluator {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'d>,
    {
        #[derive(Deserialize)]
        struct Temp {
            library_filename: String,
            function_name: String,
        }

        let temp = Temp::deserialize(deserializer)?;

        let compiled = SerializableCompiledCode {
            library_filename: temp.library_filename,
            function_name: temp.function_name,
        };

        // Load the CompiledEvaluator during deserialization
        let evaluator = compiled.load().map_err(de::Error::custom)?;

        Ok(evaluator)
    }
}

impl SerializableCompiledEvaluator {
    /// Load a new function from the same library.
    pub fn load_new_function(&self, function_name: &str) -> Result<Self, String> {
        Ok(SerializableCompiledEvaluator {
            evaluator: self.evaluator.load_new_function(function_name)?,
            library_filename: self.library_filename.clone(),
            function_name: function_name.to_string(),
        })
    }

    /// Load a compiled evaluator from a shared library.
    pub fn load(file: &str, function_name: &str) -> Result<Self, String> {
        Ok(SerializableCompiledEvaluator {
            evaluator: CompiledEvaluator::load(file, function_name)?,
            library_filename: file.to_string(),
            function_name: function_name.to_string(),
        })
    }
    #[inline(always)]
    pub fn evaluate<T: CompiledEvaluatorFloat>(&mut self, args: &[T], out: &mut [T]) {
        self.evaluator.evaluate(args, out)
    }

    /// Evaluate the compiled code with double-precision floating point numbers.
    #[inline(always)]
    pub fn evaluate_double(&mut self, args: &[f64], out: &mut [f64]) {
        self.evaluator.evaluate_double(args, out)
    }

    /// Evaluate the compiled code with complex numbers.
    #[inline(always)]
    pub fn evaluate_complex(&mut self, args: &[SymComplex<f64>], out: &mut [SymComplex<f64>]) {
        self.evaluator.evaluate_complex(args, out)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableExportedCode {
    pub source_filename: String,
    pub function_name: String,
}
impl SerializableExportedCode {
    pub fn export_cpp<T: Display>(
        expr: &ExpressionEvaluator<T>,
        filename: &str,
        function_name: &str,
        include_header: bool,
        inline_asm: InlineASM,
    ) -> Result<Self, std::io::Error> {
        let mut filename = filename.to_string();
        if !filename.ends_with(".cpp") {
            filename += ".cpp";
        }

        let cpp = match inline_asm {
            InlineASM::X64 => expr.export_asm_str(function_name, include_header, inline_asm),
            InlineASM::None => expr.export_cpp_str(function_name, include_header),
            InlineASM::AArch64 => expr.export_asm_str(function_name, include_header, inline_asm),
        };

        std::fs::write(&filename, cpp)?;
        Ok(SerializableExportedCode {
            source_filename: filename,
            function_name: function_name.to_string(),
        })
    }
    /// Compile the code to a shared library.
    pub fn compile(
        &self,
        out: &str,
        options: CompileOptions,
    ) -> Result<SerializableCompiledCode, std::io::Error> {
        let mut builder = std::process::Command::new(&options.compiler);
        builder
            .arg("-shared")
            .arg("-fPIC")
            .arg(format!("-O{}", options.optimization_level));
        if options.fast_math {
            builder.arg("-ffast-math");
        }
        if options.unsafe_math {
            builder.arg("-funsafe-math-optimizations");
        }

        for c in &options.custom {
            builder.arg(c);
        }

        let r = builder
            .arg("-o")
            .arg(out)
            .arg(&self.source_filename)
            .output()?;

        if !r.status.success() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!(
                    "Could not compile code: {}",
                    String::from_utf8_lossy(&r.stderr)
                ),
            ));
        }

        Ok(SerializableCompiledCode {
            library_filename: out.to_string(),
            function_name: self.function_name.clone(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableCompiledCode {
    library_filename: String,
    function_name: String,
}

impl SerializableCompiledCode {
    pub fn load(&self) -> Result<SerializableCompiledEvaluator> {
        let eval = CompiledEvaluator::load(&self.library_filename, &self.function_name)
            .map_err(|s| anyhow!(s))?;
        Ok(SerializableCompiledEvaluator {
            evaluator: eval,
            library_filename: self.library_filename.clone(),
            function_name: self.function_name.clone(),
        })
    }
}
// impl Clone for SerializableCompiledEvaluator {
//     fn clone(&self) -> Self {
//         Self {
//             evaluator: self.evaluator.clone(),
//             library_filename: self.library_filename.clone(),
//             function_name: self.function_name.clone(),
//         }
//     }
// }

impl<S: TensorStructure> EvalTensor<SerializableExportedCode, S> {
    pub fn compile(
        &self,
        out: &str,
        options: CompileOptions,
    ) -> Result<EvalTensor<SerializableCompiledCode, S>, std::io::Error>
    where
        S: Clone,
    {
        Ok(EvalTensor {
            eval: self.eval.compile(out, options)?,
            indexmap: self.indexmap.clone(),
            structure: self.structure.clone(),
        })
    }
}

impl<S: TensorStructure> EvalTensor<ExportedCode, S> {
    pub fn compile(
        &self,
        out: &str,
        options: CompileOptions,
    ) -> Result<EvalTensor<CompiledCode, S>, std::io::Error>
    where
        S: Clone,
    {
        Ok(EvalTensor {
            eval: self.eval.compile(out, options)?,
            indexmap: self.indexmap.clone(),
            structure: self.structure.clone(),
        })
    }
}

impl<S: TensorStructure> EvalTensor<CompiledCode, S> {
    pub fn load(&self) -> Result<EvalTensor<CompiledEvaluator, S>, String>
    where
        S: Clone,
    {
        Ok(EvalTensor {
            eval: self.eval.load()?,
            indexmap: self.indexmap.clone(),
            structure: self.structure.clone(),
        })
    }
}

impl<S: TensorStructure> EvalTensor<SerializableCompiledCode, S> {
    pub fn load(&self) -> Result<CompiledEvalTensor<S>>
    where
        S: Clone,
    {
        Ok(EvalTensor {
            eval: self.eval.load()?,
            indexmap: self.indexmap.clone(),
            structure: self.structure.clone(),
        })
    }
}

pub type LinearizedEvalTensorSet<T, S> =
    EvalTensorSet<(ExpressionEvaluator<T>, Option<Vec<Expression<T>>>), S>;

impl<T, S: TensorStructure> LinearizedEvalTensorSet<T, S> {
    pub fn map_coeff<T2, F: Fn(&T) -> T2>(self, f: &F) -> LinearizedEvalTensorSet<T2, S>
    where
        T: Clone + PartialEq + Default,
        S: Clone,
    {
        LinearizedEvalTensorSet {
            eval: (self.eval.0.map_coeff(f), None),
            tensors: self.tensors,
            size: self.size,
        }
        // self.map_data_ref(|x| x.map_coeff(f))
    }

    pub fn export_cpp(
        &self,
        filename: &str,
        function_name: &str,
        include_header: bool,
        inline_asm: InlineASM,
    ) -> Result<EvalTensorSet<SerializableExportedCode, S>, std::io::Error>
    where
        T: Display,
        S: Clone,
    {
        Ok(EvalTensorSet {
            eval: SerializableExportedCode::export_cpp(
                &self.eval.0,
                filename,
                function_name,
                include_header,
                inline_asm,
            )?,
            tensors: self.tensors.clone(),
            size: self.size,
        })
    }

    pub fn evaluate(&mut self, params: &[T]) -> TensorSet<DataTensor<T, S>>
    where
        T: Real,
        S: TensorStructure + Clone,
    {
        let zero = params[0].zero();

        let mut elements = vec![zero; self.size];
        self.eval.0.evaluate(params, &mut elements);

        match &self.tensors {
            TensorsOrScalars::Scalars => TensorSet::Scalars(elements),
            TensorsOrScalars::Tensors(t) => {
                let mut out_tensors = Vec::with_capacity(t.len());
                trace!("Evaluating {} tensors", t.len());
                for t in t.iter() {
                    out_tensors.push(t.map_data_ref(|&i| elements[i].clone()));
                }
                TensorSet::Tensors(out_tensors)
            }
        }
    }
}

impl<S: TensorStructure> EvalTensorSet<SerializableExportedCode, S> {
    pub fn compile(
        &self,
        out: &str,
        options: CompileOptions,
    ) -> Result<EvalTensorSet<SerializableCompiledCode, S>, std::io::Error>
    where
        S: Clone,
    {
        Ok(EvalTensorSet {
            eval: self.eval.compile(out, options)?,
            tensors: self.tensors.clone(),
            size: self.size,
        })
    }
}

impl<S: TensorStructure> EvalTensorSet<ExportedCode, S> {
    pub fn compile(
        &self,
        out: &str,
        options: CompileOptions,
    ) -> Result<EvalTensorSet<CompiledCode, S>, std::io::Error>
    where
        S: Clone,
    {
        Ok(EvalTensorSet {
            eval: self.eval.compile(out, options)?,
            tensors: self.tensors.clone(),
            size: self.size,
        })
    }
}

impl<S: TensorStructure> EvalTensorSet<SerializableCompiledCode, S> {
    pub fn load(&self) -> Result<CompiledEvalTensorSet<S>>
    where
        S: Clone,
    {
        Ok(EvalTensorSet {
            eval: self.eval.load()?,
            tensors: self.tensors.clone(),
            size: self.size,
        })
    }
}

impl<S: TensorStructure> EvalTensorSet<CompiledCode, S> {
    pub fn load(&self) -> Result<EvalTensorSet<CompiledEvaluator, S>, String>
    where
        S: Clone,
    {
        Ok(EvalTensorSet {
            eval: self.eval.load()?,
            tensors: self.tensors.clone(),
            size: self.size,
        })
    }
}

pub type CompiledEvalTensor<S> = EvalTensor<SerializableCompiledEvaluator, S>;

impl<S> CompiledEvalTensor<S> {
    pub fn evaluate_float(&mut self, params: &[f64]) -> DataTensor<f64, S>
    where
        S: TensorStructure + Clone,
    {
        if let Some(ref indexmap) = self.indexmap {
            let mut elements = vec![0.; indexmap.len()];
            self.eval.evaluate(params, &mut elements);
            let s = SparseTensor {
                elements: indexmap.iter().cloned().zip(elements.drain(0..)).collect(),
                structure: self.structure.clone(),
            };
            DataTensor::Sparse(s)
        } else {
            let mut out_data = DenseTensor::repeat(self.structure.clone(), 0.);
            self.eval.evaluate(params, &mut out_data.data);
            DataTensor::Dense(out_data)
        }
    }

    pub fn evaluate_complex(&mut self, params: &[SymComplex<f64>]) -> DataTensor<SymComplex<f64>, S>
    where
        S: TensorStructure + Clone,
    {
        if let Some(ref indexmap) = self.indexmap {
            let mut elements: Vec<SymComplex<f64>> = vec![SymComplex::new_zero(); indexmap.len()];
            self.eval.evaluate_complex(params, &mut elements);
            let s = SparseTensor {
                elements: indexmap.iter().cloned().zip(elements.drain(0..)).collect(),
                structure: self.structure.clone(),
            };
            DataTensor::Sparse(s)
        } else {
            let mut out_data = DenseTensor::repeat(self.structure.clone(), SymComplex::new_zero());
            self.eval.evaluate_complex(params, &mut out_data.data);
            DataTensor::Dense(out_data)
        }
    }

    pub fn evaluate<T: CompiledEvaluatorFloat + Default + Clone>(
        &mut self,
        params: &[T],
    ) -> DataTensor<T, S>
    where
        S: TensorStructure + Clone,
    {
        if let Some(ref indexmap) = self.indexmap {
            let mut elements: Vec<T> = vec![T::default(); indexmap.len()];
            self.eval.evaluate(params, &mut elements);
            let s = SparseTensor {
                elements: indexmap.iter().cloned().zip(elements.drain(0..)).collect(),
                structure: self.structure.clone(),
            };
            DataTensor::Sparse(s)
        } else {
            let mut out_data = DenseTensor::repeat(self.structure.clone(), T::default());
            self.eval.evaluate(params, &mut out_data.data);
            DataTensor::Dense(out_data)
        }
    }
}

pub type CompiledEvalTensorSet<S> = EvalTensorSet<SerializableCompiledEvaluator, S>;

impl<S: TensorStructure> CompiledEvalTensorSet<S> {
    pub fn evaluate_float(&mut self, params: &[f64]) -> TensorSet<DataTensor<f64, S>>
    where
        S: TensorStructure + Clone,
    {
        let zero = params[0].zero();

        let mut elements = vec![zero; self.size];
        self.eval.evaluate_double(params, &mut elements);

        match &self.tensors {
            TensorsOrScalars::Scalars => TensorSet::Scalars(elements),
            TensorsOrScalars::Tensors(t) => {
                let mut out_tensors = Vec::with_capacity(t.len());
                for t in t.iter() {
                    out_tensors.push(t.map_data_ref(|&i| elements[i]));
                }
                TensorSet::Tensors(out_tensors)
            }
        }
    }

    pub fn evaluate_complex(
        &mut self,
        params: &[SymComplex<f64>],
    ) -> TensorSet<DataTensor<SymComplex<f64>, S>>
    where
        S: TensorStructure + Clone,
    {
        let zero = params[0].zero();

        let mut elements = vec![zero; self.size];
        self.eval.evaluate_complex(params, &mut elements);

        match &self.tensors {
            TensorsOrScalars::Scalars => TensorSet::Scalars(elements),
            TensorsOrScalars::Tensors(t) => {
                let mut out_tensors = Vec::with_capacity(t.len());
                for t in t.iter() {
                    out_tensors.push(t.map_data_ref(|&i| elements[i]));
                }
                TensorSet::Tensors(out_tensors)
            }
        }
    }

    pub fn evaluate<T: CompiledEvaluatorFloat + Default + Clone>(
        &mut self,
        params: &[T],
    ) -> TensorSet<DataTensor<T, S>>
    where
        S: TensorStructure + Clone,
    {
        let zero = T::default();

        let mut elements = vec![zero; self.size];
        self.eval.evaluate(params, &mut elements);

        match &self.tensors {
            TensorsOrScalars::Scalars => TensorSet::Scalars(elements),
            TensorsOrScalars::Tensors(t) => {
                let mut out_tensors = Vec::with_capacity(t.len());
                trace!("Evaluating {} tensors using compiled eval", t.len());
                for t in t.iter() {
                    out_tensors.push(t.map_data_ref(|&i| elements[i].clone()));
                }
                TensorSet::Tensors(out_tensors)
            }
        }
    }
}

// impl FallibleMul for SerializableAtom {
//     type Output = SerializableAtom;
//     fn mul_fallible(&self, rhs: &Self) -> Option<Self::Output> {
//         Some(SerializableAtom(&self.0 * &rhs.0))
//     }
// }

// impl FallibleAdd<SerializableAtom> for SerializableAtom {
//     type Output = SerializableAtom;
//     fn add_fallible(&self, rhs: &Self) -> Option<Self::Output> {
//         Some(SerializableAtom(&self.0 + &rhs.0))
//     }
// }
