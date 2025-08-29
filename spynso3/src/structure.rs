use std::ops::Deref;

use itertools::Itertools;

use pyo3::types::IntoPyDict;

use pyo3::{
    exceptions::{self, PyIndexError, PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    pybacked::PyBackedStr,
    types::{PyList, PyTuple},
};
use spenso::{
    network::{
        library::symbolic::{ETS, ExplicitKey},
        parsing::ShadowedStructure,
    },
    structure::{
        HasName, IndexLess, OrderedStructure, PermutedStructure, TensorStructure, ToSymbolic,
        abstract_index::AbstractIndex,
        dimension::Dimension,
        permuted::Perm,
        representation::{
            Euclidean, ExtendibleReps, LibraryRep, Minkowski, RepName, Representation,
        },
        slot::{IsAbstractSlot, Slot},
    },
    tensors::symbolic::SymbolicTensor,
};
use symbolica::{
    api::python::PythonTransformer,
    atom::{
        Atom, AtomView, DefaultNamespace, FunctionAttribute, FunctionBuilder, NamespacedSymbol,
        Symbol,
    },
    printer::PrintOptions,
    state::Workspace,
    symbol,
    transformer::{Transformer, TransformerState},
};

use symbolica::api::python::{ConvertibleToExpression, PythonExpression};

use thiserror::Error;

use idenso::{
    IndexTooling, color::CS, gamma::AGS, metric::PermuteWithMetric, representations::Bispinor,
};

use super::{ModuleInit, SliceOrIntOrExpanded};

#[cfg(feature = "python_stubgen")]
use pyo3_stub_gen::{PyStubType, derive::*, impl_stub_type};

pub struct ConvertibleToSpensoName(pub SpensoName);

impl<'py> FromPyObject<'py> for ConvertibleToSpensoName {
    fn extract_bound(structure: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(structure) = structure.extract::<SpensoName>() {
            Ok(ConvertibleToSpensoName(structure))
        } else if let Ok(s) = structure.extract::<String>() {
            Ok(ConvertibleToSpensoName(
                SpensoName::symbol_shorthand(s, None, None, None, None, None, None).unwrap(),
            ))
        } else if let Ok(s) = structure.extract::<PythonExpression>() {
            if let AtomView::Var(a) = s.as_view() {
                Ok(ConvertibleToSpensoName(SpensoName {
                    name: a.get_symbol(),
                }))
            } else {
                Err(PyTypeError::new_err(
                    "Tensor name cannot be built from non-variable expressions",
                ))
            }
        } else {
            Err(PyTypeError::new_err("Invalid input type for tensor name"))
        }
    }
}

#[cfg(feature = "python_stubgen")]
impl PyStubType for ConvertibleToSpensoName {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        SpensoIndices::type_output() | <Vec<SpensoSlot>>::type_output()
    }
}
pub enum SpensoSlotOrArgOrRep {
    Slot(SpensoSlot),
    Arg(PythonExpression),
    Rep(SpensoRepresentation),
}

impl<'py> FromPyObject<'py> for SpensoSlotOrArgOrRep {
    fn extract_bound(structure: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(structure) = structure.extract::<SpensoSlot>() {
            Ok(SpensoSlotOrArgOrRep::Slot(structure))
        } else if let Ok(s) = structure.extract::<SpensoRepresentation>() {
            Ok(SpensoSlotOrArgOrRep::Rep(s))
        } else if let Ok(s) = structure.extract::<ConvertibleToExpression>() {
            Ok(SpensoSlotOrArgOrRep::Arg(s.to_expression()))
        } else {
            Err(PyTypeError::new_err(
                "Invalid input type for tensor slot, representation, or argument",
            ))
        }
    }
}

#[cfg(feature = "python_stubgen")]
impl PyStubType for SpensoSlotOrArgOrRep {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        SpensoIndices::type_output() | <Vec<SpensoSlot>>::type_output()
    }
}

#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.community.spenso")
)]
#[pyclass(name = "TensorName", module = "symbolica.community.spenso")]
#[derive(Clone)]
/// The name of a tensor.
pub struct SpensoName {
    pub name: Symbol,
    // pub args: Vec<Atom>,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[pymethods]
impl SpensoName {
    #[new]
    #[pyo3(signature = (name,is_symmetric=None,is_antisymmetric=None,is_cyclesymmetric=None,is_linear=None,custom_normalization=None,custom_print=None))]
    /// Shorthand notation for :func:`Expression.symbol`.
    fn symbol_shorthand(
        name: String,
        is_symmetric: Option<bool>,
        is_antisymmetric: Option<bool>,
        is_cyclesymmetric: Option<bool>,
        is_linear: Option<bool>,
        custom_normalization: Option<PythonTransformer>,
        custom_print: Option<PyObject>,
    ) -> PyResult<Self> {
        let namespace = DefaultNamespace {
            namespace: "spenso_python".into(),
            data: "",
            file: "".into(),
            line: 0,
        };
        if is_symmetric.is_none()
            && is_antisymmetric.is_none()
            && is_cyclesymmetric.is_none()
            && is_linear.is_none()
            && custom_normalization.is_none()
            && custom_print.is_none()
        {
            let id = Symbol::new(namespace.attach_namespace(&name))
                .build()
                .map_err(|e| exceptions::PyTypeError::new_err(e.to_string()))?;

            return Ok(SpensoName {
                name: id,
                // args: vec![],
            });
        }

        let count = (is_symmetric == Some(true)) as u8
            + (is_antisymmetric == Some(true)) as u8
            + (is_cyclesymmetric == Some(true)) as u8;

        if count > 1 {
            Err(exceptions::PyValueError::new_err(
                "Function cannot be both symmetric, antisymmetric or cyclesymmetric",
            ))?;
        }

        let mut opts = vec![];

        if let Some(true) = is_symmetric {
            opts.push(FunctionAttribute::Symmetric);
        }

        if let Some(true) = is_antisymmetric {
            opts.push(FunctionAttribute::Antisymmetric);
        }

        if let Some(true) = is_cyclesymmetric {
            opts.push(FunctionAttribute::Cyclesymmetric);
        }

        if let Some(true) = is_linear {
            opts.push(FunctionAttribute::Linear);
        }

        let name = namespace.attach_namespace(&name);

        let mut symbol = Symbol::new(name).with_attributes(opts);

        if let Some(f) = custom_normalization {
            symbol = symbol.with_normalization_function(Box::new(
                move |input: AtomView<'_>, out: &mut Atom| {
                    let _ = Workspace::get_local()
                        .with(|ws| {
                            Transformer::execute_chain(
                                input,
                                &f.chain,
                                ws,
                                &TransformerState::default(),
                                out,
                            )
                        })
                        .unwrap();
                    true
                },
            ))
        }

        if let Some(f) = custom_print {
            symbol = symbol.with_print_function(Box::new(
                move |input: AtomView<'_>, opts: &PrintOptions| {
                    Python::with_gil(|py| {
                        let kwargs = opts.into_py_dict(py).unwrap();
                        f.call(
                            py,
                            (PythonExpression::from(input.to_owned()),),
                            Some(&kwargs),
                        )
                        .unwrap()
                        .extract::<Option<String>>(py)
                        .unwrap()
                    })
                },
            ))
        }

        let symbol = symbol
            .build()
            .map_err(|e| exceptions::PyTypeError::new_err(e.to_string()))?;

        Ok(SpensoName {
            name: symbol,
            // args: vec![],
        })
    }

    #[pyo3(signature = (*args))]
    fn __call__(&self, args: &Bound<'_, PyTuple>) -> PyResult<PossiblyIndexed> {
        let mut add_args: Vec<Atom> = Vec::new();
        let mut slots: Vec<_> = Vec::new();
        let mut reps: Vec<_> = Vec::new();

        for arg_bound in args.iter() {
            let convertible = arg_bound.extract::<SpensoSlotOrArgOrRep>()?;

            match convertible {
                SpensoSlotOrArgOrRep::Arg(expr) => add_args.push(expr.expr),
                SpensoSlotOrArgOrRep::Slot(slot) => slots.push(slot.slot),
                SpensoSlotOrArgOrRep::Rep(rep) => reps.push(rep.representation),
            }
        }

        let add_args = if add_args.is_empty() {
            None
        } else {
            Some(add_args)
        };

        if slots.is_empty() && reps.is_empty() {
            Err(exceptions::PyValueError::new_err(
                "No slots or representations provided",
            ))
        } else if reps.is_empty() {
            Ok(PossiblyIndexed::Indexed(SpensoIndices {
                structure: ShadowedStructure::<AbstractIndex>::from_iter(
                    slots, self.name, add_args,
                ),
            }))
        } else if slots.is_empty() {
            Ok(PossiblyIndexed::Unindexed(SpensoStructure {
                structure: ExplicitKey::from_iter(reps, self.name, add_args),
            }))
        } else {
            Err(exceptions::PyValueError::new_err(
                "Cannot generate structure with both slots and representations",
            ))
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.name)
    }

    fn __str__(&self) -> String {
        format!("{}", self.name)
    }

    fn to_expression(&self) -> PythonExpression {
        PythonExpression::from(Atom::var(self.name))
    }

    #[classattr]
    fn g() -> SpensoName {
        SpensoName { name: ETS.metric }
    }
    #[classattr]
    fn flat() -> SpensoName {
        SpensoName { name: ETS.flat }
    }
    #[classattr]
    fn gamma() -> SpensoName {
        SpensoName { name: AGS.gamma }
    }
    #[classattr]
    fn gamma5() -> SpensoName {
        SpensoName { name: AGS.gamma5 }
    }
    #[classattr]
    fn projm() -> SpensoName {
        SpensoName { name: AGS.projm }
    }
    #[classattr]
    fn projp() -> SpensoName {
        SpensoName { name: AGS.projp }
    }
    #[classattr]
    fn sigma() -> SpensoName {
        SpensoName { name: AGS.sigma }
    }
    #[classattr]
    fn f() -> SpensoName {
        SpensoName { name: CS.f }
    }
    #[classattr]
    fn t() -> SpensoName {
        SpensoName { name: CS.t }
    }
}

#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.community.spenso")
)]
#[pyclass(name = "TensorIndices", module = "symbolica.community.spenso")]
#[derive(Clone)]
/// A structure that can be used to represent the "shape" of a tensor, along with a list of abstract indices.
/// This has an optional name, and accompanying symbolica expressions that are considered as additional non-indexed arguments.
/// The structure is essentially a list of `Slots` that are used to define the structure of the tensor.
pub struct SpensoIndices {
    pub structure: PermutedStructure<ShadowedStructure<AbstractIndex>>,
}

impl Deref for SpensoIndices {
    type Target = ShadowedStructure<AbstractIndex>;

    fn deref(&self) -> &Self::Target {
        &self.structure.structure
    }
}

impl From<ShadowedStructure<AbstractIndex>> for SpensoIndices {
    fn from(value: ShadowedStructure<AbstractIndex>) -> Self {
        SpensoIndices {
            structure: PermutedStructure::identity(value),
        }
    }
}

impl ModuleInit for SpensoIndices {
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<SpensoIndices>()?;
        m.add_class::<SpensoName>()?;
        m.add_class::<SpensoSlot>()?;
        m.add_class::<SpensoStructure>()?;
        m.add_class::<SpensoRepresentation>()?;
        Ok(())
    }
}

pub enum ArithmeticStructure {
    Convertible(ConvertibleToExpression),
    Structure(SpensoIndices),
    Expression(PythonExpression),
}

#[cfg(feature = "python_stubgen")]
impl PyStubType for ArithmeticStructure {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        ConvertibleToExpression::type_output()
            | SpensoIndices::type_output()
            | PythonExpression::type_output()
    }
}

impl ArithmeticStructure {
    pub fn to_expression(self) -> PyResult<PythonExpression> {
        match self {
            ArithmeticStructure::Convertible(expr) => Ok(expr.to_expression()),
            ArithmeticStructure::Structure(indices) => indices.to_expression(),
            ArithmeticStructure::Expression(expr) => Ok(expr),
        }
    }
}

impl<'a> FromPyObject<'a> for ArithmeticStructure {
    fn extract_bound(ob: &Bound<'a, PyAny>) -> PyResult<Self> {
        if let Ok(ob) = ob.extract::<ConvertibleToExpression>() {
            Ok(ArithmeticStructure::Convertible(ob))
        } else if let Ok(ob) = ob.extract::<SpensoIndices>() {
            Ok(ArithmeticStructure::Structure(ob))
        } else {
            Err(exceptions::PyTypeError::new_err(
                "Only convertible expressions and spenso indices can be used",
            ))
        }
    }
}

pub struct ConvertibleToStructure(pub SpensoIndices);

impl<'py> FromPyObject<'py> for ConvertibleToStructure {
    fn extract_bound(structure: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(structure) = structure.extract::<SpensoIndices>() {
            Ok(ConvertibleToStructure(structure))
        } else if let Ok(s) = structure.extract::<Vec<SpensoSlot>>() {
            Ok(ConvertibleToStructure(SpensoIndices {
                structure: PermutedStructure::<OrderedStructure>::from_iter(
                    s.into_iter().map(|s| s.slot),
                )
                .map_structure(Into::into),
            }))
        } else {
            Err(PyTypeError::new_err(
                "Internal tensor structure can only be build from TensorIndices or lists of Slots",
            ))
        }
    }
}

#[cfg(feature = "python_stubgen")]
impl PyStubType for ConvertibleToStructure {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        SpensoIndices::type_output() | <Vec<SpensoSlot>>::type_output()
    }
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[pymethods]
impl SpensoIndices {
    #[new]
    #[pyo3(signature =
           (
           *additional_args,name=None))]
    pub fn from_list(
        additional_args: &Bound<'_, PyTuple>,
        name: Option<ConvertibleToSpensoName>,
    ) -> PyResult<Self> {
        let mut args = Vec::new();
        let mut slots = Vec::new();
        for a in additional_args {
            if let Ok(s) = a.extract::<SpensoSlot>() {
                slots.push(s.slot);
            } else if let Ok(arg) = a.extract::<PythonExpression>() {
                args.push(arg.expr);
            } else {
                return Err(exceptions::PyTypeError::new_err(
                    "Only slots and expressions can be used",
                ));
            }
        }

        let args = if args.is_empty() { None } else { Some(args) };
        let mut a: PermutedStructure<ShadowedStructure<AbstractIndex>> =
            PermutedStructure::<OrderedStructure>::from_iter(slots).map_structure(Into::into);
        if let Some(name) = name {
            a.structure.set_name(name.0.name);
        };
        a.structure.additional_args = args;

        Ok(SpensoIndices { structure: a })
    }

    fn set_name(&mut self, name: ConvertibleToSpensoName) {
        self.structure.structure.set_name(name.0.name);
    }

    fn get_name(&self) -> Option<SpensoName> {
        self.structure
            .structure
            .name()
            .map(|a| SpensoName { name: a })
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.structure)
    }

    fn __str__(&self) -> String {
        if let Some(structure) = SymbolicTensor::from_named(&self.structure.structure) {
            let atom = PermutedStructure {
                index_permutation: self.structure.index_permutation.clone(),
                rep_permutation: self.structure.rep_permutation.clone(),
                structure,
            }
            .permute_inds()
            .expression;

            format!("{}", atom)
        } else {
            assert!(self.structure.index_permutation.is_identity());
            assert!(self.structure.rep_permutation.is_identity());
            let args = self
                .structure
                .structure
                .external_structure_iter()
                .map(|r| r.to_atom())
                .join(",");

            format!("({})", args.trim_end())
        }
    }

    fn to_expression(&self) -> PyResult<PythonExpression> {
        if self.structure.structure.name().is_none() {
            return Err(PyRuntimeError::new_err("No name"));
        }

        let atom = PermutedStructure {
            index_permutation: self.structure.index_permutation.clone(),
            rep_permutation: self.structure.rep_permutation.clone(),
            structure: self.structure.structure.clone(),
        }
        .permute_with_metric();

        Ok(atom.into())
    }

    fn __len__(&self) -> usize {
        self.structure.structure.size().unwrap()
    }

    fn __getitem__(&self, item: SliceOrIntOrExpanded) -> PyResult<Py<PyAny>> {
        match item {
            SliceOrIntOrExpanded::Int(i) => {
                let out: Vec<_> = self
                    .expanded_index(i.into())
                    .map_err(|s| PyIndexError::new_err(s.to_string()))?
                    .into();

                Ok(Python::with_gil(|py| out.into_pyobject(py).map(|a| a.unbind()))?.into_any())
            }
            SliceOrIntOrExpanded::Expanded(idxs) => {
                let out: usize = self
                    .flat_index(&idxs)
                    .map_err(|s| PyIndexError::new_err(s.to_string()))?
                    .into();

                Ok(Python::with_gil(|py| out.into_pyobject(py).map(|a| a.unbind()))?.into_any())
            }
            SliceOrIntOrExpanded::Slice(s) => {
                let r = s.indices(self.size().unwrap() as isize)?;

                let start = if r.start < 0 {
                    (r.slicelength as isize + r.start) as usize
                } else {
                    r.start as usize
                };

                let end = if r.stop < 0 {
                    (r.slicelength as isize + r.stop) as usize
                } else {
                    r.stop as usize
                };

                let (range, step) = if r.step < 0 {
                    (end..start, -r.step as usize)
                } else {
                    (start..end, r.step as usize)
                };

                let slice: Result<Vec<Vec<usize>>, _> = range
                    .step_by(step)
                    .map(|i| self.expanded_index(i.into()).map(Vec::<usize>::from))
                    .collect();

                match slice {
                    Ok(slice) => {
                        Ok(
                            Python::with_gil(|py| slice.into_pyobject(py).map(|a| a.unbind()))?
                                .into_any(),
                        )
                    }
                    Err(e) => Err(PyIndexError::new_err(e.to_string())),
                }
            }
        }
    }

    /// Add this expression to `other`, returning the result.
    pub fn __add__(&self, rhs: ArithmeticStructure) -> PyResult<PythonExpression> {
        let rhs = rhs.to_expression()?;
        Ok((self.to_expression()?.expr.as_ref() + rhs.expr.as_ref()).into())
    }

    /// Add this expression to `other`, returning the result.
    pub fn __radd__(&self, rhs: ArithmeticStructure) -> PyResult<PythonExpression> {
        self.__add__(rhs)
    }

    /// Subtract `other` from this expression, returning the result.
    pub fn __sub__(&self, rhs: ArithmeticStructure) -> PyResult<PythonExpression> {
        let rhs = rhs.to_expression()?.__neg__()?;
        self.__add__(ArithmeticStructure::Expression(rhs))
    }

    /// Subtract this expression from `other`, returning the result.
    pub fn __rsub__(&self, rhs: ArithmeticStructure) -> PyResult<PythonExpression> {
        let s = self.to_expression()?.__neg__()?.expr;

        let r = rhs.to_expression()?.expr;
        Ok((r + s).into())
    }

    /// Add this expression to `other`, returning the result.
    pub fn __mul__(&self, rhs: ArithmeticStructure) -> PyResult<PythonExpression> {
        let rhs = rhs.to_expression()?;
        Ok((self.to_expression()?.expr.as_ref() * rhs.expr.as_ref()).into())
    }

    /// Add this expression to `other`, returning the result.
    pub fn __rmul__(&self, rhs: ArithmeticStructure) -> PyResult<PythonExpression> {
        self.__mul__(rhs)
    }
}

#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.community.spenso")
)]
#[pyclass(name = "TensorStructure", module = "symbolica.community.spenso")]
#[derive(Clone)]
/// A structure that can be used to represent the "shape" of a tensor.
/// This has an optional name, and accompanying symbolica expressions that are considered as additional non-indexed arguments.
/// The structure is essentially a list of `Representation` that are used to define the structure of the tensor.
pub struct SpensoStructure {
    pub structure: PermutedStructure<ExplicitKey<AbstractIndex>>,
}

impl Deref for SpensoStructure {
    type Target = ExplicitKey<AbstractIndex>;

    fn deref(&self) -> &Self::Target {
        &self.structure.structure
    }
}

pub struct ConvertibleToIndexLess(pub SpensoStructure);

impl From<ExplicitKey<AbstractIndex>> for SpensoStructure {
    fn from(value: ExplicitKey<AbstractIndex>) -> Self {
        SpensoStructure {
            structure: PermutedStructure::identity(value),
        }
    }
}

impl<'py> FromPyObject<'py> for ConvertibleToIndexLess {
    fn extract_bound(structure: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(structure) = structure.extract::<SpensoStructure>() {
            Ok(ConvertibleToIndexLess(structure))
        } else if let Ok(s) = structure.extract::<Vec<SpensoRepresentation>>() {
            Ok(ConvertibleToIndexLess(SpensoStructure {
                structure: PermutedStructure::<IndexLess>::from_iter(
                    s.into_iter().map(|s| s.representation),
                )
                .map_structure(Into::into),
            }))
        } else if let Ok(s) = structure.extract::<Vec<usize>>() {
            Ok(ConvertibleToIndexLess(SpensoStructure {
                structure: PermutedStructure::<IndexLess>::from_iter(
                    s.into_iter().map(|s| ExtendibleReps::EUCLIDEAN.new_rep(s)),
                )
                .map_structure(Into::into),
            }))
        } else {
            Err(PyTypeError::new_err(
                "Internal tensor structure can only be build from TensorStructure or lists of Representations or Integers",
            ))
        }
    }
}

#[cfg(feature = "python_stubgen")]
impl PyStubType for ConvertibleToIndexLess {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        SpensoStructure::type_output()
            | <Vec<SpensoRepresentation>>::type_output()
            | <Vec<usize>>::type_output()
    }
}
#[derive(Error, Debug)]
pub enum SpensoError {
    #[error("Must have a name to register")]
    NoName,
}

#[pymethods]
#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
impl SpensoStructure {
    #[new]
    #[pyo3(signature =
           (
           *additional_args,name=None))]
    pub fn from_list(
        additional_args: &Bound<'_, PyTuple>,
        name: Option<ConvertibleToSpensoName>,
    ) -> PyResult<Self> {
        let mut args = Vec::new();
        let mut slots = Vec::new();
        for a in additional_args {
            if let Ok(s) = a.extract::<SpensoRepresentation>() {
                slots.push(s.representation);
            } else if let Ok(arg) = a.extract::<PythonExpression>() {
                args.push(arg.expr);
            } else {
                return Err(exceptions::PyTypeError::new_err(
                    "Only slots and expressions can be used",
                ));
            }
        }

        let args = if args.is_empty() { None } else { Some(args) };

        let mut a: PermutedStructure<ExplicitKey<AbstractIndex>> =
            PermutedStructure::<IndexLess>::from_iter(slots).map_structure(Into::into);
        if let Some(name) = name {
            a.structure.set_name(name.0.name);
        };
        a.structure.additional_args = args;

        Ok(SpensoStructure { structure: a })
    }

    fn set_name(&mut self, name: ConvertibleToSpensoName) {
        self.structure.structure.set_name(name.0.name);
    }

    fn get_name(&self) -> Option<SpensoName> {
        self.structure
            .structure
            .name()
            .map(|a| SpensoName { name: a })
    }

    fn __repr__(&self) -> String {
        format!(
            "{}",
            self.to_symbolic(Some(self.structure.rep_permutation.clone()))
                .unwrap()
        )
    }

    fn __str__(&self) -> String {
        let slot = self
            .external_reps()
            .into_iter()
            .map(|r| r.to_symbolic([]))
            .join(",");

        match (self.name(), self.args()) {
            (Some(name), Some(args)) => {
                let args = args.iter().join(",");
                format!("{}({})[{}]", name, args, slot)
            }
            (Some(name), None) => {
                format!("{}[{}]", name, slot)
            }
            (None, Some(args)) => {
                let args = args.iter().join(",");
                format!("({})[{}]", args, slot)
            }
            (None, None) => {
                format!("[{}]", slot)
            }
        }
    }

    fn __len__(&self) -> usize {
        self.size().unwrap()
    }

    fn __getitem__(&self, item: SliceOrIntOrExpanded) -> PyResult<Py<PyAny>> {
        match item {
            SliceOrIntOrExpanded::Int(i) => {
                let out: Vec<_> = self
                    .expanded_index(i.into())
                    .map_err(|s| PyIndexError::new_err(s.to_string()))?
                    .into();

                Ok(Python::with_gil(|py| out.into_pyobject(py).map(|a| a.unbind()))?.into_any())
            }
            SliceOrIntOrExpanded::Expanded(idxs) => {
                let out: usize = self
                    .flat_index(&idxs)
                    .map_err(|s| PyIndexError::new_err(s.to_string()))?
                    .into();

                Ok(Python::with_gil(|py| out.into_pyobject(py).map(|a| a.unbind()))?.into_any())
            }
            SliceOrIntOrExpanded::Slice(s) => {
                let r = s.indices(self.size().unwrap() as isize)?;

                let start = if r.start < 0 {
                    (r.slicelength as isize + r.start) as usize
                } else {
                    r.start as usize
                };

                let end = if r.stop < 0 {
                    (r.slicelength as isize + r.stop) as usize
                } else {
                    r.stop as usize
                };

                let (range, step) = if r.step < 0 {
                    (end..start, -r.step as usize)
                } else {
                    (start..end, r.step as usize)
                };

                let slice: Result<Vec<Vec<usize>>, _> = range
                    .step_by(step)
                    .map(|i| self.expanded_index(i.into()).map(Vec::<usize>::from))
                    .collect();

                match slice {
                    Ok(slice) => {
                        Ok(
                            Python::with_gil(|py| slice.into_pyobject(py).map(|a| a.unbind()))?
                                .into_any(),
                        )
                    }
                    Err(e) => Err(PyIndexError::new_err(e.to_string())),
                }
            }
        }
    }

    #[pyo3(signature = (*args, extra_args=None))]
    /// Convenience method. Calls `symbolic(*args, extra_args=extra_args)`.
    ///
    /// Creates a symbolic `Expression` representing this tensor structure. See the
    /// `symbolic` method documentation for details on argument handling.
    fn __call__(
        &self,
        args: &Bound<'_, PyTuple>,
        extra_args: Option<&Bound<'_, PyList>>,
    ) -> PyResult<PythonExpression> {
        // Directly delegate to symbolic, passing relevant arguments through
        self.symbolic(args, extra_args)
    }

    #[pyo3(signature = (*args, extra_args=None))]
    /// Creates a symbolic `Expression` representing this tensor structure with the given arguments.
    ///
    /// # Args:
    ///     *args (int | str | Symbol | Expression | ';'): Positional arguments. Can include
    ///         indices, expressions, or the semicolon separator.
    ///     extra_args (list[Expression], optional): Explicit list of additional non-tensorial args.
    ///
    /// Interprets positional arguments (`*args`) as potential indices. Arguments
    /// before a semicolon separator (`;`) and arguments provided via the `extra_args`
    /// keyword argument are combined and treated as additional non-tensorial arguments.
    /// Arguments after the semicolon (or all positional arguments if no separator is used)
    /// are treated as the symbolic tensor indices.
    ///
    ///
    /// # Returns:
    ///     Expression: A symbolic expression representing the tensor.
    ///
    /// # Raises:
    ///     ValueError: If index count mismatches or separator is misused.
    ///     TypeError: If arguments have unexpected types.
    ///     RuntimeError: If the structure does not have a name.
    fn symbolic(
        &self,
        args: &Bound<'_, PyTuple>,
        extra_args: Option<&Bound<'_, PyList>>,
    ) -> PyResult<PythonExpression> {
        // Use helper to parse arguments
        let (final_additional_args, potential_indices) =
            self.parse_args_for_indexing(args, extra_args)?;

        // --- Generate Symbolic Expression ---
        let name = self.name().ok_or_else(|| {
            PyRuntimeError::new_err("Cannot create symbolic atom: structure has no name")
        })?;

        let index_atoms: Vec<Atom> = potential_indices
            .iter()
            .map(|item| {
                match item {
                    // potential_indices now only contains Aind or Atom
                    ConvertibleToAbstractIndex::Aind(idx) => (*idx).into(),
                    ConvertibleToAbstractIndex::Atom(expr) => expr.expr.clone(),
                    ConvertibleToAbstractIndex::Separator => unreachable!(), // Helper ensures this
                }
            })
            .collect();

        if self.order() != index_atoms.len() {
            return Err(PyValueError::new_err(format!(
                "Number of index atoms {} does not match structure order {}",
                index_atoms.len(),
                self.order()
            )));
        }

        let slots_atoms = self
            .external_reps_iter()
            .zip(index_atoms)
            .map(|(rep, ind_atom)| rep.to_symbolic([ind_atom]))
            .collect::<Vec<_>>();

        let value_builder = FunctionBuilder::new(name);
        let final_expr = value_builder
            .add_args(&final_additional_args)
            .add_args(&slots_atoms)
            .finish();

        Ok(PythonExpression::from(final_expr))
    }

    #[pyo3(signature = (*args, extra_args=None, cook_indices=false))]
    /// Creates an indexed tensor instance (`TensorIndices`) from this structure.
    ///
    /// Interprets positional arguments (`*args`) as potential indices. Arguments
    /// before a semicolon separator (`;`) and arguments provided via the `extra_args`
    /// keyword argument are combined and treated as additional non-tensorial arguments.
    /// Arguments after the semicolon (or all positional arguments if no separator is used)
    /// are treated as the tensor indices.
    ///
    /// # Args:
    ///     *args: Positional arguments. Can include indices (int, str, Symbol, Expression),
    ///            or a single semicolon string (`;`).
    ///     extra_args (list[Expression], optional): An explicit list of additional non-tensorial
    ///         arguments. Defaults to None.
    ///     cook_indices (bool, optional): If True, attempt to "cook" non-index arguments
    ///         intended as tensor indices into valid `AbstractIndex` representations.
    ///         Defaults to False.
    ///
    /// # Returns:
    ///     TensorIndices: An object representing the tensor structure with concrete indices assigned.
    ///
    /// # Raises:
    ///     ValueError: If index resolution fails, counts mismatch, or separator is misused.
    ///     TypeError: If arguments have unexpected types.
    fn index(
        &self,
        args: &Bound<'_, PyTuple>,
        extra_args: Option<&Bound<'_, PyList>>,
        cook_indices: bool,
    ) -> PyResult<SpensoIndices> {
        // Use helper to parse arguments
        let (final_additional_args, potential_indices) =
            self.parse_args_for_indexing(args, extra_args)?;

        // --- Resolve Indices (No change in this logic) ---
        let mut resolved_indices: Vec<AbstractIndex> = Vec::new();
        for item in potential_indices {
            // potential_indices now only contains Aind or Atom
            match item {
                ConvertibleToAbstractIndex::Aind(idx) => {
                    resolved_indices.push(idx);
                }
                ConvertibleToAbstractIndex::Atom(expr) => {
                    let converted_atom: Result<AbstractIndex, _> = if cook_indices {
                        expr.expr.cook_indices().as_view().try_into()
                    } else {
                        expr.expr.as_view().try_into()
                    };
                    match converted_atom {
                        Ok(idx) => resolved_indices.push(idx),
                        Err(e) => {
                            let cook_msg = if cook_indices {
                                ""
                            } else {
                                " Try setting cook_indices=True."
                            };
                            return Err(exceptions::PyValueError::new_err(format!(
                                "Cannot convert argument '{}' to an AbstractIndex: {}. Ensure it's a valid index type or cookable.{}",
                                expr.expr, e, cook_msg
                            )));
                        }
                    }
                }
                ConvertibleToAbstractIndex::Separator => unreachable!(), // Helper ensures this
            }
        }

        let mut structure_clone = self.structure.structure.clone();
        structure_clone.additional_args = if final_additional_args.is_empty() {
            None
        } else {
            Some(final_additional_args)
        };
        match structure_clone.reindex(&resolved_indices) {
            Ok(indexed_structure) => Ok(SpensoIndices {
                structure: indexed_structure,
            }),
            Err(e) => Err(PyValueError::new_err(format!(
                "Failed to create TensorIndices: {}",
                e
            ))),
        }
    }
}

impl SpensoStructure {
    fn parse_args_for_indexing(
        &self,
        args: &Bound<'_, PyTuple>,
        extra_args_opt: Option<&Bound<'_, PyList>>,
    ) -> PyResult<(Vec<Atom>, Vec<ConvertibleToAbstractIndex>)> {
        let mut pre_separator_args: Vec<ConvertibleToAbstractIndex> = Vec::new();
        let mut post_separator_args: Vec<ConvertibleToAbstractIndex> = Vec::new();
        let mut separator_found = false;

        for arg_bound in args.iter() {
            let convertible = arg_bound.extract::<ConvertibleToAbstractIndex>()?;

            match convertible {
                ConvertibleToAbstractIndex::Separator => {
                    if separator_found {
                        return Err(exceptions::PyValueError::new_err(
                            "Separator token ';' used more than once.",
                        ));
                    }

                    separator_found = true;
                    pre_separator_args.append(&mut post_separator_args);
                }
                item => {
                    post_separator_args.push(item);
                }
            }
        }

        let mut final_additional_args = self.args().unwrap_or_default();
        for item in pre_separator_args {
            match item {
                ConvertibleToAbstractIndex::Aind(idx) => final_additional_args.push(idx.into()),
                ConvertibleToAbstractIndex::Atom(expr) => {
                    final_additional_args.push(expr.expr.clone())
                }
                ConvertibleToAbstractIndex::Separator => unreachable!(),
            }
        }
        if let Some(extra_args_list) = extra_args_opt {
            for item_bound in extra_args_list.iter() {
                let expr = item_bound.extract::<PythonExpression>()?;
                final_additional_args.push(expr.expr);
            }
        }

        Ok((final_additional_args, post_separator_args))
    }
}
#[derive(IntoPyObject)]
pub enum PossiblyIndexed {
    Unindexed(SpensoStructure),
    Indexed(SpensoIndices),
}

#[cfg(feature = "python_stubgen")]
impl PyStubType for PossiblyIndexed {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        SpensoStructure::type_output() | SpensoIndices::type_output()
    }
}

#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.community.spenso")
)]
#[pyclass(name = "Representation", module = "symbolica.community.spenso")]
#[derive(Clone)]
/// A representation class in the sense of representation theory. This class is used to represent the representation of a tensor. It is essentially a pair of a name and a dimension.
/// New representations are registered when constructing.
/// Some representations are dualizable, meaning that they have a dual representation.
/// Indices will only ever match across dual representations.
/// There are some already registered representations, such as:
///  EUCLIDEAN: Rep = Rep::SelfDual(0);
///  BISPINOR: Rep = Rep::SelfDual(1);
///  COLORADJ: Rep = Rep::SelfDual(2);
///  MINKOWSKI: Rep = Rep::SelfDual(3);
///
///  LORENTZ_UP: Rep = Rep::Dualizable(1);
///  LORENTZ_DOWN: Rep = Rep::Dualizable(-1);
///  SPINFUND: Rep = Rep::Dualizable(2);
///  SPINANTIFUND: Rep = Rep::Dualizable(-2);
///  COLORFUND: Rep = Rep::Dualizable(3);
///  COLORANTIFUND: Rep = Rep::Dualizable(-3);
///  COLORSEXT: Rep = Rep::Dualizable(4);
///  COLORANTISEXT: Rep = Rep::Dualizable(-4);
///
pub struct SpensoRepresentation {
    pub representation: Representation<LibraryRep>,
}

pub enum ConvertibleToAbstractIndex {
    Aind(AbstractIndex),
    Atom(PythonExpression),
    Separator,
}

impl<'py> FromPyObject<'py> for ConvertibleToAbstractIndex {
    fn extract_bound(aind: &Bound<'py, PyAny>) -> PyResult<Self> {
        let aind = if let Ok(i) = aind.extract::<char>() {
            if i == ';' {
                ConvertibleToAbstractIndex::Separator
            } else {
                let mut tmp = [0u8; 4];
                let name = i.encode_utf8(&mut tmp);
                ConvertibleToAbstractIndex::Aind(AbstractIndex::Symbol(symbol!(&name).into()))
            }
        } else if let Ok(i) = aind.extract::<isize>() {
            ConvertibleToAbstractIndex::Aind(i.into())
        } else if let Ok(expr) = aind.extract::<PythonExpression>() {
            match expr.expr.as_view() {
                AtomView::Var(v) => {
                    ConvertibleToAbstractIndex::Aind(AbstractIndex::Symbol(v.get_symbol().into()))
                }
                _ => ConvertibleToAbstractIndex::Atom(expr),
            }
        } else if let Ok(s) = aind.extract::<PyBackedStr>() {
            let id = symbol!(&s);
            ConvertibleToAbstractIndex::Aind(AbstractIndex::Symbol(id.into()))
        } else {
            return Err(PyTypeError::new_err(
                "Argument must be convertible to an index (int, str, Symbol), an Expression,, or the separator ';'",
            ));
        };

        Ok(aind)
    }
}

#[cfg(feature = "python_stubgen")]
impl_stub_type!(ConvertibleToAbstractIndex = isize | Symbol | PyBackedStr);

pub struct ConvertibleToDimension(Dimension);

impl<'py> FromPyObject<'py> for ConvertibleToDimension {
    fn extract_bound(dimension: &Bound<'py, PyAny>) -> PyResult<Self> {
        let dim = if let Ok(i) = dimension.extract::<usize>() {
            Dimension::from(i)
        } else if let Ok(expr) = dimension.extract::<PythonExpression>() {
            let id = match expr.expr.as_view() {
                AtomView::Var(v) => v.get_symbol(),
                _ => {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only symbols can be abstract indices",
                    ));
                }
            };
            Dimension::from(id)
        } else if let Ok(s) = dimension.extract::<PyBackedStr>() {
            let ns = "spenso_python";
            let id = Symbol::new(NamespacedSymbol {
                symbol: format!("{}::{}", ns, s).into(),
                namespace: ns.into(),
                file: file!().into(),
                line: line!() as usize,
            })
            .build()
            .unwrap();

            Dimension::from(id)
        } else {
            return Err(PyTypeError::new_err(
                "dimension must be an non-zero integer or a symbol",
            ));
        };
        Ok(ConvertibleToDimension(dim))
    }
}

#[cfg(feature = "python_stubgen")]
impl_stub_type!(ConvertibleToDimension = usize | PythonExpression | PyBackedStr);

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[pymethods]
impl SpensoRepresentation {
    #[new]
    #[pyo3(signature =
           (
           name,dimension,is_self_dual=true))]
    /// Register a new representation with the given name and dimension. If dual is true, the representation will be dualizable, else it will be self-dual.
    pub fn register_new(
        name: Bound<'_, PyAny>,
        dimension: ConvertibleToDimension,
        is_self_dual: bool,
    ) -> PyResult<Self> {
        let name = name.extract::<PyBackedStr>()?;

        let dim = dimension.0;

        let rep = if is_self_dual {
            LibraryRep::new_self_dual(&name).unwrap().new_rep(dim)
        } else {
            LibraryRep::new_dual(&name).unwrap().new_rep(dim)
        };
        Ok(SpensoRepresentation {
            representation: rep,
        })
    }

    /// Generate a new slot with the given index, from this representation
    fn __call__(&self, py: Python<'_>, aind: ConvertibleToAbstractIndex) -> PyResult<Py<PyAny>> {
        match aind {
            ConvertibleToAbstractIndex::Separator => {
                Err(PyValueError::new_err("separator cannot be an index"))
            }
            ConvertibleToAbstractIndex::Aind(aind) => Ok(SpensoSlot {
                slot: self.representation.slot(aind),
            }
            .into_pyobject(py)
            .map(|a| a.unbind())?
            .into_any()),
            ConvertibleToAbstractIndex::Atom(a) => {
                let a: PythonExpression = self.representation.to_symbolic([a.expr]).into();

                Ok(a.into_pyobject(py).map(|a| a.unbind())?.into_any())
            }
        }
    }

    fn g(
        &self,
        i: ConvertibleToAbstractIndex,
        j: ConvertibleToAbstractIndex,
    ) -> PyResult<SpensoIndices> {
        match (i, j) {
            (ConvertibleToAbstractIndex::Aind(i), ConvertibleToAbstractIndex::Aind(j)) => {
                let structure = ShadowedStructure::<AbstractIndex>::from_iter(
                    [self.representation.slot(i), self.representation.slot(j)],
                    ETS.metric,
                    None,
                );

                Ok(SpensoIndices { structure })
            }
            _ => Err(PyValueError::new_err("indices must be abstract indices")),
        }
    }

    fn flat(
        &self,
        i: ConvertibleToAbstractIndex,
        j: ConvertibleToAbstractIndex,
    ) -> PyResult<SpensoIndices> {
        match (i, j) {
            (ConvertibleToAbstractIndex::Aind(i), ConvertibleToAbstractIndex::Aind(j)) => {
                let structure = ShadowedStructure::<AbstractIndex>::from_iter(
                    [self.representation.slot(i), self.representation.slot(j)],
                    ETS.flat,
                    None,
                );

                Ok(SpensoIndices { structure })
            }
            _ => Err(PyValueError::new_err("indices must be abstract indices")),
        }
    }

    fn id(
        &self,
        i: ConvertibleToAbstractIndex,
        j: ConvertibleToAbstractIndex,
    ) -> PyResult<SpensoIndices> {
        match (i, j) {
            (ConvertibleToAbstractIndex::Aind(i), ConvertibleToAbstractIndex::Aind(j)) => {
                let structure = ShadowedStructure::<AbstractIndex>::from_iter(
                    [self.representation.slot(i), self.representation.slot(j)],
                    ETS.metric,
                    None,
                );

                Ok(SpensoIndices { structure })
            }
            _ => Err(PyValueError::new_err("indices must be abstract indices")),
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.representation)
    }

    fn __str__(&self) -> String {
        format!("{}", self.representation.to_symbolic([]))
    }

    fn to_expression(&self) -> PythonExpression {
        PythonExpression::from(self.representation.to_symbolic([]))
    }

    #[staticmethod]
    fn bis(dimension: ConvertibleToDimension) -> Self {
        let dim = dimension.0;
        let rep = Bispinor {}.new_rep(dim).cast();
        Self {
            representation: rep,
        }
    }

    #[staticmethod]
    fn euc(dimension: ConvertibleToDimension) -> Self {
        let dim = dimension.0;
        let rep = Euclidean {}.new_rep(dim).cast();
        Self {
            representation: rep,
        }
    }

    #[staticmethod]
    fn mink(dimension: ConvertibleToDimension) -> Self {
        let dim = dimension.0;
        let rep = Minkowski {}.new_rep(dim).cast();
        Self {
            representation: rep,
        }
    }
}

/// An abstract index slot for a tensor.
/// This is essentially a tuple of a `Representation` and an abstract index id.
///
/// The abstract index id can be either an integer or a symbol.
/// This is the building block for creating tensor structures that can be contracted.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.community.spenso")
)]
#[pyclass(name = "Slot", module = "symbolica.community.spenso")]
#[derive(Clone)]
pub struct SpensoSlot {
    pub slot: Slot<LibraryRep>,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[pymethods]
impl SpensoSlot {
    fn __repr__(&self) -> String {
        format!("{:?}", self.slot)
    }

    fn __str__(&self) -> String {
        format!("{}", self.slot.to_atom())
    }

    #[new]
    #[pyo3(signature =
           (
           name,dimension,aind,dual=false))]
    /// Create a new slot from a name of a representation, a dimension and an abstract index.
    ///  If dual is true, the representation will be dualizable, else it will be self-dual.
    pub fn register_new(
        name: Bound<'_, PyAny>,
        dimension: usize,
        aind: Bound<'_, PyAny>,
        dual: bool,
    ) -> PyResult<Self> {
        let name = name.extract::<PyBackedStr>()?;
        let rep = if dual {
            LibraryRep::new_dual(&name).unwrap().new_rep(dimension)
        } else {
            LibraryRep::new_self_dual(&name).unwrap().new_rep(dimension)
        };
        if let Ok(i) = aind.extract::<isize>() {
            Ok(SpensoSlot { slot: rep.slot(i) })
        } else if let Ok(expr) = aind.extract::<PythonExpression>() {
            let id = match expr.expr.as_view() {
                AtomView::Var(v) => v.get_symbol(),
                _ => {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only symbols can be abstract indices",
                    ));
                }
            };

            let aind = AbstractIndex::Symbol(id.into());
            Ok(SpensoSlot {
                slot: rep.slot(aind),
            })
        } else if let Ok(s) = aind.extract::<PyBackedStr>() {
            let id = symbol!(&s);

            Ok(SpensoSlot {
                slot: rep.slot(AbstractIndex::Symbol(id.into())),
            })
        } else {
            Err(PyTypeError::new_err("aind must be an integer or a symbol"))
        }
    }

    fn to_expression(&self) -> PythonExpression {
        PythonExpression::from(self.slot.to_atom())
    }
}

#[cfg(feature = "python_stubgen")]
pyo3_stub_gen::define_stub_info_gatherer!(stub_info);
