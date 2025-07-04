#[cfg(feature = "shadowing")]
use anyhow::anyhow;
use serde::{Deserialize, Serialize};

use crate::{
    algebra::algebraic_traits::{One, Zero},
    structure::{permuted::PermuteTensor, PermutedStructure},
};
use std::{
    borrow::Cow,
    fmt::{Debug, Display},
};

#[cfg(feature = "shadowing")]
use symbolica::{
    atom::{Atom, AtomView},
    domains::{
        float::{Complex as SymComplex, Real, SingleFloat},
        rational::Rational,
        InternalOrdering,
    },
    evaluate::{
        CompileOptions, CompiledCode, CompiledEvaluator, EvalTree, ExportNumber, ExportedCode,
        ExpressionEvaluator, FunctionMap, InlineASM,
    },
};

#[cfg(feature = "shadowing")]
use crate::{
    tensors::data::{DataIterator, DenseTensor, SetTensorData, SparseTensor},
    tensors::parametric::ParamTensor,
};

#[cfg(feature = "shadowing")]
use super::store::TensorScalarStoreMapping;

use crate::{
    structure::{CastStructure, HasStructure, ScalarTensor, TensorStructure},
    tensors::data::DataTensor,
};

use super::{
    library::LibraryTensor,
    store::{NetworkStore, TensorScalarStore},
    ExecutionResult,
};
use super::{Library, Network, TensorNetworkError, TensorOrScalarOrKey};

#[derive(
    Debug,
    Clone,
    Serialize,
    Deserialize,
    bincode_trait_derive::Encode,
    bincode_trait_derive::Decode,
    // bincode_trait_derive::BorrowDecodeFromDecode,
)]
#[cfg_attr(
    feature = "shadowing",
    trait_decode(trait = symbolica::state::HasStateMap),
)]
pub struct TensorNetworkSet<S, K> {
    pub networks: Vec<Network<S, K>>,
}

impl<S, K> FromIterator<Network<S, K>> for TensorNetworkSet<S, K> {
    fn from_iter<T: IntoIterator<Item = Network<S, K>>>(iter: T) -> Self {
        Self {
            networks: Vec::from_iter(iter),
        }
    }
}

impl<S, K> TensorNetworkSet<S, K> {
    pub fn new() -> Self {
        TensorNetworkSet { networks: vec![] }
    }

    pub fn push(&mut self, network: Network<S, K>) {
        // self.scalars.push(network.scalar);
        self.networks.push(network);
    }
}

impl<S, K> Default for TensorNetworkSet<S, K> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "shadowing")]
pub type EvalTreeTensorNetworkSet<T, S, K, Str> = SharedTensorNetworkSet<EvalTree<T>, S, K, Str>;

#[cfg(feature = "shadowing")]
pub type EvalTensorNetworkSet<T, S, K, Str> =
    SharedTensorNetworkSet<ExpressionEvaluator<T>, S, K, Str>;

#[cfg(feature = "shadowing")]
pub type CompiledTensorNetworkSet<S, K, Str> = SharedTensorNetworkSet<CompiledEvaluator, S, K, Str>;

#[derive(Debug, Clone)]
pub struct SharedTensorNetworkSet<
    D,
    S: TensorStructure,
    K,
    Str: TensorScalarStore<Tensor = DataTensor<usize, S>, Scalar = usize> = NetworkStore<
        DataTensor<usize, S>,
        usize,
    >,
> {
    pub networks: Vec<Network<Str, K>>,
    pub shared_data: D,
    pub len: usize,
}

impl<T: TensorStructure, S, K: Display + Debug, Str: TensorScalarStore<Tensor = T, Scalar = S>>
    TensorNetworkSet<Str, K>
{
    pub fn result(
        &self,
    ) -> Result<
        Vec<ExecutionResult<TensorOrScalarOrKey<&T, &S, &PermutedStructure<K>>>>,
        TensorNetworkError<K>,
    > {
        self.networks.iter().map(|n| n.result()).collect()
    }

    pub fn result_tensor<'a, LT, L: Library<T::Structure, Key = K, Value = PermutedStructure<LT>>>(
        &'a self,
        lib: &L,
    ) -> Result<Vec<ExecutionResult<Cow<'a, T>>>, TensorNetworkError<K>>
    where
        S: 'a,
        T: Clone + ScalarTensor + HasStructure,
        T::Scalar: One + Zero,
        for<'b> &'b S: Into<T::Scalar>,
        LT: TensorStructure<Indexed = T> + Clone + LibraryTensor<WithIndices = T>,
        T: PermuteTensor<Permuted = T>,
    {
        self.networks.iter().map(|n| n.result_tensor(lib)).collect()
    }

    pub fn result_scalar<'a>(
        &'a self,
    ) -> Result<Vec<ExecutionResult<Cow<'a, S>>>, TensorNetworkError<K>>
    where
        T: Clone + ScalarTensor + 'a,
        T::Scalar: Into<S>,
        S: One + Zero + Clone,
    {
        self.networks.iter().map(|n| n.result_scalar()).collect()
    }

    pub fn cast<U>(self) -> TensorNetworkSet<Str::Store<U, S>, K>
    where
        K: Clone,
        T: CastStructure<U> + HasStructure,
        T::Structure: TensorStructure,
        U: HasStructure,
        U::Structure: From<T::Structure> + TensorStructure<Slot = T::Slot>,
    {
        self.networks.into_iter().map(|n| n.cast()).collect()
    }
}

#[cfg(feature = "shadowing")]
impl<
        Store: TensorScalarStore<Tensor = ParamTensor<S>, Scalar = Atom>,
        S: TensorStructure + Clone,
        K: Clone,
    > TensorNetworkSet<Store, K>
{
    pub fn eval_tree(
        self,
        fn_map: &FunctionMap,
        params: &[Atom],
    ) -> anyhow::Result<
        EvalTreeTensorNetworkSet<
            SymComplex<Rational>,
            S,
            K,
            Store::Store<DataTensor<usize, S>, usize>,
        >,
    >
    where
        Store::Store<DataTensor<usize, S>, usize>:
            TensorScalarStore<Tensor = DataTensor<usize, S>, Scalar = usize>,
        S: TensorStructure,
    {
        let mut networks = vec![];

        let mut atoms = vec![];
        let mut scalar_id = 0;
        let mut tensor_id = 0;

        for net in &self.networks {
            // let refnet = net.map_ref(|a|, |t|t);
            for s in net.iter_scalars() {
                atoms.push(s);
                tensor_id += 1;
            }

            let tensors = net.map_ref(
                |_| {
                    let oldid = scalar_id;
                    scalar_id += 1;
                    oldid
                },
                |t| {
                    let structure = t.structure().clone();
                    match &t.tensor {
                        DataTensor::Dense(d) => {
                            let oldid = tensor_id;
                            tensor_id += d.size().unwrap();
                            for (_, a) in d.flat_iter() {
                                atoms.push(a);
                            }
                            DataTensor::Dense(
                                DenseTensor::from_data(Vec::from_iter(oldid..tensor_id), structure)
                                    .expect("Failed to create DenseTensor"),
                            )
                        }
                        DataTensor::Sparse(s) => {
                            let mut t = SparseTensor::empty(structure);
                            for (i, a) in s.flat_iter() {
                                t.set_flat(i, tensor_id)
                                    .expect("Failed to set value in SparseTensor");
                                atoms.push(a);
                                tensor_id += 1;
                            }
                            DataTensor::Sparse(t)
                        }
                    }
                },
            );

            networks.push(tensors);
        }

        Ok(EvalTreeTensorNetworkSet {
            networks,
            shared_data: AtomView::to_eval_tree_multiple(&atoms, fn_map, params)
                .map_err(|s| anyhow!(s))?,
            len: atoms.len(),
        })
    }
}

#[cfg(feature = "shadowing")]
impl<
        S: Clone + TensorStructure,
        K,
        Store: TensorScalarStore<Tensor = DataTensor<usize, S>, Scalar = usize>,
    > EvalTreeTensorNetworkSet<SymComplex<Rational>, S, K, Store>
{
    pub fn horner_scheme(&mut self) {
        self.shared_data.horner_scheme();
    }
}

#[cfg(feature = "shadowing")]
impl<
        T,
        S: TensorStructure + Clone,
        K: Clone,
        Store: TensorScalarStore<Tensor = DataTensor<usize, S>, Scalar = usize> + Clone,
    > EvalTreeTensorNetworkSet<T, S, K, Store>
{
    pub fn map_coeff<T2, F: Fn(&T) -> T2>(&self, f: &F) -> EvalTreeTensorNetworkSet<T2, S, K, Store>
    where
        T: Clone + PartialEq,
    {
        EvalTreeTensorNetworkSet {
            networks: self.networks.clone(),
            shared_data: self.shared_data.map_coeff(f),
            len: self.len,
        }
        // self.map_data_ref(|x| x.map_coeff(f))
    }

    pub fn linearize(self, cpe_rounds: Option<usize>) -> EvalTensorNetworkSet<T, S, K, Store>
    where
        T: Clone + Default + PartialEq,
    {
        EvalTensorNetworkSet {
            networks: self.networks,
            shared_data: self.shared_data.linearize(cpe_rounds),
            len: self.len,
        }
    }

    pub fn common_subexpression_elimination(&mut self)
    where
        T: std::fmt::Debug + std::hash::Hash + Eq + InternalOrdering + Clone + Default,
    {
        self.shared_data.common_subexpression_elimination()
    }

    pub fn evaluate(
        &mut self,
        params: &[T],
    ) -> TensorNetworkSet<Store::Store<DataTensor<T, S>, T>, K>
    where
        T: Real + SingleFloat,
        S: TensorStructure + Clone,
    {
        let zero = params[0].zero();
        let mut data = vec![zero; self.len];

        let mut networks = vec![];

        self.shared_data.evaluate(params, &mut data);

        for net in self.networks.iter() {
            let data_net = net.map_ref(
                |s| data[*s].clone(),
                |p| {
                    let structure = p.structure().clone();
                    match &p {
                        DataTensor::Dense(d) => {
                            let mut t_data = vec![];
                            for (_, &a) in d.flat_iter() {
                                t_data.push(data[a].clone());
                            }
                            DataTensor::Dense(DenseTensor::from_data(t_data, structure).unwrap())
                        }
                        DataTensor::Sparse(s) => {
                            let mut t = SparseTensor::empty(structure);
                            for (i, &a) in s.flat_iter() {
                                t.set_flat(i, data[a].clone()).unwrap();
                            }
                            DataTensor::Sparse(t)
                        }
                    }
                },
            );

            networks.push(data_net);
        }

        TensorNetworkSet { networks }
    }
}

#[cfg(feature = "shadowing")]
impl<
        T,
        S: TensorStructure + Clone,
        K: Clone,
        Store: TensorScalarStore<Tensor = DataTensor<usize, S>, Scalar = usize> + Clone,
    > EvalTensorNetworkSet<T, S, K, Store>
{
    pub fn evaluate(
        &mut self,
        params: &[T],
    ) -> TensorNetworkSet<Store::Store<DataTensor<T, S>, T>, K>
    where
        T: Real + SingleFloat,
        S: TensorStructure + Clone,
    {
        let zero = params[0].zero();
        let mut data = vec![zero; self.len];

        let mut networks = vec![];

        self.shared_data.evaluate(params, &mut data);

        for net in self.networks.iter() {
            let data_net = net.map_ref(
                |s| data[*s].clone(),
                |p| {
                    let structure = p.structure().clone();
                    match &p {
                        DataTensor::Dense(d) => {
                            let mut t_data = vec![];
                            for (_, &a) in d.flat_iter() {
                                t_data.push(data[a].clone());
                            }
                            DataTensor::Dense(DenseTensor::from_data(t_data, structure).unwrap())
                        }
                        DataTensor::Sparse(s) => {
                            let mut t = SparseTensor::empty(structure);
                            for (i, &a) in s.flat_iter() {
                                t.set_flat(i, data[a].clone()).unwrap();
                            }
                            DataTensor::Sparse(t)
                        }
                    }
                },
            );

            networks.push(data_net);
        }

        TensorNetworkSet { networks }
    }

    /// Create a C++ code representation of the evaluation tree tensor.
    /// With `inline_asm` set to any value other than `None`,
    /// high-performance inline ASM code will be generated for most
    /// evaluation instructions. This often gives better performance than
    /// the `O3` optimization level and results in very fast compilation.
    pub fn export_cpp(
        &self,
        filename: &str,
        function_name: &str,
        include_header: bool,
        inline_asm: InlineASM,
    ) -> Result<SharedTensorNetworkSet<ExportedCode, S, K, Store>, std::io::Error>
    where
        T: ExportNumber + SingleFloat,
    {
        Ok(SharedTensorNetworkSet {
            networks: self.networks.clone(),
            shared_data: self.shared_data.export_cpp(
                filename,
                function_name,
                include_header,
                inline_asm,
            )?,
            len: self.len,
        })
    }
}

#[cfg(feature = "shadowing")]
impl<S: TensorStructure + Clone, K: Clone> SharedTensorNetworkSet<ExportedCode, S, K> {
    pub fn compile(
        &self,
        out: &str,
        options: CompileOptions,
    ) -> Result<SharedTensorNetworkSet<CompiledCode, S, K>, std::io::Error> {
        Ok(SharedTensorNetworkSet {
            networks: self.networks.clone(),
            shared_data: self.shared_data.compile(out, options)?,
            len: self.len,
        })
    }
}

#[cfg(feature = "shadowing")]
impl<S: TensorStructure + Clone, K: Clone> SharedTensorNetworkSet<CompiledCode, S, K> {
    pub fn load(&self) -> Result<SharedTensorNetworkSet<CompiledEvaluator, S, K>, String> {
        Ok(SharedTensorNetworkSet {
            networks: self.networks.clone(),
            shared_data: self.shared_data.load()?,
            len: self.len,
        })
    }
}

#[cfg(feature = "shadowing")]
impl<
        S: TensorStructure + Clone,
        K: Clone,
        Store: TensorScalarStore<Tensor = DataTensor<usize, S>, Scalar = usize> + Clone,
    > CompiledTensorNetworkSet<S, K, Store>
{
    pub fn evaluate<T: symbolica::evaluate::CompiledEvaluatorFloat + Default + Clone>(
        &mut self,
        params: &[T],
    ) -> TensorNetworkSet<Store::Store<DataTensor<T, S>, T>, K>
    where
        S: TensorStructure + Clone,
    {
        let zero = T::default();
        let mut data = vec![zero; self.len];

        let mut networks = vec![];

        self.shared_data.evaluate(params, &mut data);
        for net in self.networks.iter() {
            let data_net = net.map_ref(
                |s| data[*s].clone(),
                |p| {
                    let structure = p.structure().clone();
                    match &p {
                        DataTensor::Dense(d) => {
                            let mut t_data = vec![];
                            for (_, &a) in d.flat_iter() {
                                t_data.push(data[a].clone());
                            }
                            DataTensor::Dense(DenseTensor::from_data(t_data, structure).unwrap())
                        }
                        DataTensor::Sparse(s) => {
                            let mut t = SparseTensor::empty(structure);
                            for (i, &a) in s.flat_iter() {
                                t.set_flat(i, data[a].clone()).unwrap();
                            }
                            DataTensor::Sparse(t)
                        }
                    }
                },
            );

            networks.push(data_net);
        }
        TensorNetworkSet { networks }
    }
}
