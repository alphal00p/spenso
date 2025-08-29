use std::{collections::HashMap, ops::Deref};

use pyo3::{
    exceptions::{self, PyRuntimeError},
    prelude::*,
};

use spenso::{
    network::{
        ContractScalars, ExecutionResult, Network, Sequential, SingleSmallestDegree,
        SmallestDegree, Steps,
        library::symbolic::ExplicitKey,
        parsing::ShadowedStructure,
        store::{NetworkStore, TensorScalarStoreMapping},
    },
    structure::abstract_index::AbstractIndex,
    tensors::parametric::{MixedTensor, ParamOrConcrete, atomcore::TensorAtomMaps},
};
use spenso_hep_lib::HEP_LIB;
use symbolica::{
    api::python::{ConvertibleToPatternRestriction, ConvertibleToReplaceWith, PythonExpression},
    atom::{Atom, AtomCore, AtomView},
    evaluate::EvaluationFn,
    id::{MatchSettings, ReplaceWith},
    poly::Variable,
};

use symbolica::api::python::ConvertibleToExpression;

use super::{Spensor, library::SpensorLibrary, structure::ArithmeticStructure};

use super::ModuleInit;

#[cfg(feature = "python_stubgen")]
use pyo3_stub_gen::{PyStubType, derive::*};

#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.community.spenso")
)]
#[pyclass(name = "TensorNetwork")]
#[derive(Clone)]
#[allow(clippy::type_complexity)]
/// A tensor network.
///
/// This class is a wrapper around the `TensorNetwork` class from the `spenso` crate.
/// Such a network is a graph representing the arithmetic operations between tensors.
/// In the most basic case, edges represent the contraction of indices.
pub struct SpensoNet {
    pub network: Network<
        NetworkStore<MixedTensor<f64, ShadowedStructure<AbstractIndex>>, Atom>,
        ExplicitKey<AbstractIndex>,
    >,
}

#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass_enum(module = "symbolica.community.spenso")
)]
#[pyclass(name = "ExecutionMode")]
#[derive(Clone)]
pub enum ExecutionMode {
    Single,
    Scalar,
    All,
}

impl ModuleInit for ExecutionMode {}

impl ModuleInit for SpensoNet {
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<SpensoNet>()
    }
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pyfunction)]
#[pyfunction(name = "to_net")]
pub fn python_to_tensor_network(
    a: ArithmeticStructure,
    library: Option<&SpensorLibrary>,
) -> anyhow::Result<SpensoNet> {
    SpensoNet::from_expression(a, library)
}

pub type ParsingNet = Network<
    NetworkStore<MixedTensor<f64, ShadowedStructure<AbstractIndex>>, Atom>,
    ExplicitKey<AbstractIndex>,
>;

impl From<ParsingNet> for SpensoNet {
    fn from(network: ParsingNet) -> Self {
        SpensoNet { network }
    }
}

pub struct ConvertibleToSpensoNet(SpensoNet);

impl ConvertibleToSpensoNet {
    pub fn to_net(self) -> SpensoNet {
        self.0
    }
}

impl<'a> FromPyObject<'a> for ConvertibleToSpensoNet {
    fn extract_bound(ob: &Bound<'a, pyo3::PyAny>) -> PyResult<Self> {
        if let Ok(a) = ob.extract::<SpensoNet>() {
            Ok(ConvertibleToSpensoNet(a))
        } else if let Ok(num) = ob.extract::<Spensor>() {
            Ok(ConvertibleToSpensoNet(SpensoNet {
                network: Network::from_tensor(num.tensor.structure),
            }))
        } else if let Ok(a) = ob.extract::<ConvertibleToExpression>() {
            Ok(ConvertibleToSpensoNet(SpensoNet {
                network: ParsingNet::try_from_view(
                    a.to_expression().expr.as_view(),
                    &SpensorLibrary::constuct().library,
                )
                .map_err(|a| PyRuntimeError::new_err(a.to_string()))?,
            }))
        } else {
            Err(exceptions::PyTypeError::new_err(
                "Cannot convert to expression",
            ))
        }
    }
}

#[cfg(feature = "python_stubgen")]
impl PyStubType for ConvertibleToSpensoNet {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        ArithmeticStructure::type_output() | SpensoNet::type_output() | Spensor::type_output()
    }
}

// #[gen_stub_pymethods]

#[pymethods]
impl SpensoNet {
    #[new]
    /// Parses an expression into a network
    #[pyo3(signature = (expr, library=None))]
    pub fn from_expression(
        expr: ArithmeticStructure,
        library: Option<&SpensorLibrary>,
    ) -> anyhow::Result<SpensoNet> {
        let lib = library.map(|l| &l.library).unwrap_or(HEP_LIB.deref());

        Ok(SpensoNet {
            network: ParsingNet::try_from_view(expr.to_expression()?.as_view(), lib)?,
        })
    }

    #[staticmethod]
    pub fn one() -> SpensoNet {
        SpensoNet {
            network: Network::one(),
        }
    }

    #[staticmethod]
    pub fn zero() -> SpensoNet {
        SpensoNet {
            network: Network::zero(),
        }
    }

    #[pyo3(signature = (pattern, rhs, _cond = None, non_greedy_wildcards = None, level_range = None, level_is_tree_depth = None, allow_new_wildcards_on_rhs = None, rhs_cache_size = None, repeat = None))]
    #[allow(clippy::too_many_arguments)]
    pub fn replace(
        &self,
        pattern: ConvertibleToExpression,
        rhs: ConvertibleToReplaceWith,
        _cond: Option<ConvertibleToPatternRestriction>,
        non_greedy_wildcards: Option<Vec<PythonExpression>>,
        level_range: Option<(usize, Option<usize>)>,
        level_is_tree_depth: Option<bool>,
        allow_new_wildcards_on_rhs: Option<bool>,
        rhs_cache_size: Option<usize>,
        repeat: Option<bool>,
    ) -> PyResult<SpensoNet> {
        let pattern = pattern.to_expression().expr.to_pattern();
        let ReplaceWith::Pattern(rhs) = &rhs.to_replace_with()? else {
            return Err(exceptions::PyTypeError::new_err(
                "Only normal patterns supported",
            ));
        };

        let mut settings = MatchSettings::cached();

        if let Some(ngw) = non_greedy_wildcards {
            settings.non_greedy_wildcards = ngw
                .iter()
                .map(|x| match x.expr.as_view() {
                    AtomView::Var(v) => {
                        let name = v.get_symbol();
                        if v.get_wildcard_level() == 0 {
                            return Err(exceptions::PyTypeError::new_err(
                                "Only wildcards can be restricted.",
                            ));
                        }
                        Ok(name)
                    }
                    _ => Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    )),
                })
                .collect::<Result<_, _>>()?;
        }
        if let Some(level_range) = level_range {
            settings.level_range = level_range;
        }
        if let Some(level_is_tree_depth) = level_is_tree_depth {
            settings.level_is_tree_depth = level_is_tree_depth;
        }
        if let Some(allow_new_wildcards_on_rhs) = allow_new_wildcards_on_rhs {
            settings.allow_new_wildcards_on_rhs = allow_new_wildcards_on_rhs;
        }
        if let Some(rhs_cache_size) = rhs_cache_size {
            settings.rhs_cache_size = rhs_cache_size;
        }

        let cond = None;

        Ok(SpensoNet {
            network: self.network.map_ref(
                |s| {
                    let r = s.replace(&pattern);
                    let r = if let Some(cond) = cond.as_ref() {
                        r.when(cond)
                    } else {
                        r
                    }
                    .non_greedy_wildcards(settings.non_greedy_wildcards.clone())
                    .level_range(settings.level_range)
                    .level_is_tree_depth(settings.level_is_tree_depth)
                    .allow_new_wildcards_on_rhs(settings.allow_new_wildcards_on_rhs)
                    .rhs_cache_size(settings.rhs_cache_size);

                    let r = if let Some(true) = repeat {
                        r.repeat()
                    } else {
                        r
                    };

                    r.with(rhs.borrow())
                },
                |t| match t {
                    ParamOrConcrete::Param(p) => {
                        let r = p.replace(&pattern);
                        let r = if let Some(cond) = cond.as_ref() {
                            r.when(cond)
                        } else {
                            r
                        }
                        .non_greedy_wildcards(settings.non_greedy_wildcards.clone())
                        .level_range(settings.level_range)
                        .level_is_tree_depth(settings.level_is_tree_depth)
                        .allow_new_wildcards_on_rhs(settings.allow_new_wildcards_on_rhs)
                        .rhs_cache_size(settings.rhs_cache_size);

                        let r = if let Some(true) = repeat {
                            r.repeat()
                        } else {
                            r
                        };

                        ParamOrConcrete::Param(r.with(rhs.borrow()))
                    }
                    _ => t.clone(),
                },
            ),
        })
    }

    pub fn evaluate(
        &self,
        constants: HashMap<PythonExpression, f64>,
        functions: HashMap<Variable, PyObject>,
    ) -> PyResult<Self> {
        let constants = constants
            .iter()
            .map(|(k, v)| (k.expr.as_view(), *v))
            .collect();

        let functions = functions
            .into_iter()
            .map(|(k, v)| {
                let id = if let Variable::Symbol(v) = k {
                    v
                } else {
                    Err(exceptions::PyValueError::new_err(format!(
                        "Expected function name instead of {:?}",
                        k
                    )))?
                };

                Ok((
                    id,
                    EvaluationFn::new(Box::new(move |args, _, _, _| {
                        Python::with_gil(|py| {
                            v.call(py, (args.to_vec(),), None)
                                .expect("Bad callback function")
                                .extract::<f64>(py)
                                .expect("Function does not return a float")
                        })
                    })),
                ))
            })
            .collect::<PyResult<_>>()?;

        let mut network = self.network.clone();
        network.evaluate_real(|x| x.into(), &constants, &functions);
        Ok(SpensoNet { network })
    }

    #[pyo3(signature = (library=None, n_steps=None, mode=ExecutionMode::All))]
    fn execute(
        &mut self,
        library: Option<&SpensorLibrary>,
        n_steps: Option<usize>,
        mode: ExecutionMode,
    ) -> PyResult<()> {
        let lib = library.map(|l| &l.library).unwrap_or(HEP_LIB.deref());

        if let Some(n) = n_steps {
            for _ in 0..n {
                match mode {
                    ExecutionMode::All => {
                        self.network
                            .execute::<Steps<1>, SmallestDegree, _, _>(lib)
                            .map_err(|a| PyRuntimeError::new_err(a.to_string()))?;
                    }
                    ExecutionMode::Scalar => {
                        self.network
                            .execute::<Steps<1>, ContractScalars, _, _>(lib)
                            .map_err(|a| PyRuntimeError::new_err(a.to_string()))?;
                    }
                    ExecutionMode::Single => {
                        self.network
                            .execute::<Steps<1>, SingleSmallestDegree<false>, _, _>(lib)
                            .map_err(|a| PyRuntimeError::new_err(a.to_string()))?;
                    }
                }
            }
        } else {
            match mode {
                ExecutionMode::All => {
                    self.network
                        .execute::<Sequential, SmallestDegree, _, _>(lib)
                        .map_err(|a| PyRuntimeError::new_err(a.to_string()))?;
                }
                ExecutionMode::Scalar => {
                    self.network
                        .execute::<Sequential, ContractScalars, _, _>(lib)
                        .map_err(|a| PyRuntimeError::new_err(a.to_string()))?;
                }
                ExecutionMode::Single => {
                    self.network
                        .execute::<Sequential, SingleSmallestDegree<false>, _, _>(lib)
                        .map_err(|a| PyRuntimeError::new_err(a.to_string()))?;
                }
            }
        }
        Ok(())
    }
    #[pyo3(signature = (library=None))]
    fn result_tensor(&self, library: Option<&SpensorLibrary>) -> PyResult<Spensor> {
        let lib = library.map(|l| &l.library).unwrap_or(HEP_LIB.deref());

        Ok(
            match self
                .network
                .result_tensor(lib)
                .map_err(|s| PyRuntimeError::new_err(s.to_string()))?
            {
                ExecutionResult::One => Spensor::one(),
                ExecutionResult::Zero => Spensor::zero(),
                ExecutionResult::Val(v) => v.into_owned().into(),
            },
        )
    }

    fn result_scalar(&self) -> PyResult<PythonExpression> {
        Ok(
            match self
                .network
                .result_scalar()
                .map_err(|s| PyRuntimeError::new_err(s.to_string()))?
            {
                ExecutionResult::One => Atom::num(1).into(),
                ExecutionResult::Zero => Atom::Zero.into(),
                ExecutionResult::Val(v) => v.into_owned().into(),
            },
        )
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(self.network.dot_pretty())
    }

    /// Add this expression to `other`, returning the result.
    pub fn __add__(&self, rhs: ConvertibleToSpensoNet) -> PyResult<SpensoNet> {
        let rhs = rhs.to_net();
        Ok((self.network.clone() + rhs.network).into())
    }

    /// Add this expression to `other`, returning the result.
    pub fn __radd__(&self, rhs: ConvertibleToSpensoNet) -> PyResult<SpensoNet> {
        self.__add__(rhs)
    }

    /// Subtract `other` from this expression, returning the result.
    pub fn __sub__(&self, rhs: ConvertibleToSpensoNet) -> PyResult<SpensoNet> {
        let rhs = rhs.to_net();
        Ok((self.network.clone() - rhs.network).into())
    }

    /// Subtract this expression from `other`, returning the result.
    pub fn __rsub__(&self, rhs: ConvertibleToSpensoNet) -> PyResult<SpensoNet> {
        let rhs = rhs.to_net();
        Ok((rhs.network - self.network.clone()).into())
    }

    /// Add this expression to `other`, returning the result.
    pub fn __mul__(&self, rhs: ConvertibleToSpensoNet) -> PyResult<SpensoNet> {
        let rhs = rhs.to_net();
        Ok((rhs.network * self.network.clone()).into())
    }

    /// Add this expression to `other`, returning the result.
    pub fn __rmul__(&self, rhs: ConvertibleToSpensoNet) -> PyResult<SpensoNet> {
        let rhs = rhs.to_net();
        Ok((rhs.network * self.network.clone()).into())
    }

    // pub fn __pow__(&self, rhs: usize, number: Option<i64>) -> PyResult<PythonExpression> {
    //     if number.is_some() {
    //         return Err(exceptions::PyValueError::new_err(
    //             "Optional number argument not supported",
    //         ));
    //     }

    //     // let rhs = rhs.to_net();
    //     Ok(self.network.pow(&rhs).into())
    // }
}
