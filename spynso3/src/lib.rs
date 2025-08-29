use std::{collections::HashMap, ops::Deref};

use anyhow::anyhow;
use library::SpensorLibrary;
use network::SpensoNet;

use pyo3::{
    PyClass,
    exceptions::{self, PyIndexError, PyOverflowError, PyRuntimeError, PyTypeError},
    prelude::*,
    types::{PyComplex, PyFloat, PySlice, PyType},
};

use spenso::{
    algebra::complex::{Complex, RealOrComplex, symbolica_traits::CompiledComplexEvaluatorSpenso},
    tensors::{
        data::{DenseTensor, GetTensorData, SetTensorData, SparseOrDense, SparseTensor},
        parametric::{
            ConcreteOrParam, EvalTensor, ParamOrConcrete, ParamTensor, atomcore::TensorAtomOps,
        },
    },
};

use spenso::{
    network::parsing::ShadowedStructure,
    structure::{
        HasStructure, PermutedStructure, ScalarTensor, TensorStructure,
        abstract_index::AbstractIndex, permuted::Perm,
    },
    tensors::{
        complex::RealOrComplexTensor,
        data::{DataTensor, StorageTensor},
        parametric::{LinearizedEvalTensor, MixedTensor},
    },
};
use structure::{ConvertibleToStructure, SpensoIndices};
use symbolica::{
    api::python::SymbolicaCommunityModule,
    atom::Atom,
    domains::{float::Complex as SymComplex, rational::Rational},
    evaluate::{CompileOptions, ExportSettings, FunctionMap, InlineASM, OptimizationSettings},
    poly::Variable,
};

use symbolica::api::python::PythonExpression;

#[cfg(feature = "python_stubgen")]
use pyo3_stub_gen::{PyStubType, TypeInfo, define_stub_info_gatherer, derive::*};

pub mod library;
pub mod library_tensor;
pub mod network;
pub mod structure;

trait ModuleInit: PyClass {
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<Self>()
    }
}

pub struct SpensoModule;

impl SymbolicaCommunityModule for SpensoModule {
    fn get_name() -> String {
        "spenso".to_string()
    }

    fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
        initialize_spenso(m)
    }
}

pub(crate) fn initialize_spenso(m: &Bound<'_, PyModule>) -> PyResult<()> {
    use library_tensor::LibrarySpensor;
    use network::ExecutionMode;

    SpensoNet::init(m)?;

    ExecutionMode::init(m)?;
    SpensoNet::init(m)?;
    Spensor::init(m)?;
    LibrarySpensor::init(m)?;
    SpensoIndices::init(m)?;
    SpensorLibrary::init(m)?;
    Ok(())
}

/// A tensor class that can be either dense or sparse.
/// The data is either float or complex or a symbolica expression
/// It can be instantiated with data using the `sparse_empty` or `dense` module functions.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.community.spenso")
)]
#[pyclass(name = "Tensor")]
#[derive(Clone)]
pub struct Spensor {
    tensor: PermutedStructure<MixedTensor<f64, ShadowedStructure<AbstractIndex>>>,
}

impl Deref for Spensor {
    type Target = MixedTensor<f64, ShadowedStructure<AbstractIndex>>;

    fn deref(&self) -> &Self::Target {
        &self.tensor.structure
    }
}

impl ModuleInit for Spensor {}

// #[gen_stub_pyclass_enum]

#[derive(FromPyObject)]
pub enum SliceOrIntOrExpanded<'a> {
    Slice(Bound<'a, PySlice>),
    Int(usize),
    Expanded(Vec<usize>),
}

#[cfg(feature = "python_stubgen")]
impl<'a> PyStubType for SliceOrIntOrExpanded<'a> {
    fn type_input() -> pyo3_stub_gen::TypeInfo {
        TypeInfo::builtin("slice") | usize::type_input() | TypeInfo::list_of::<usize>()
    }

    fn type_output() -> pyo3_stub_gen::TypeInfo {
        TypeInfo::builtin("slice") | usize::type_input() | TypeInfo::list_of::<usize>()
    }
}

#[derive(IntoPyObject)]
pub enum TensorElements {
    Real(Py<PyFloat>),
    Complex(Py<PyComplex>),
    Symbolica(PythonExpression),
}

impl From<ConcreteOrParam<RealOrComplex<f64>>> for TensorElements {
    fn from(value: ConcreteOrParam<RealOrComplex<f64>>) -> Self {
        match value {
            ConcreteOrParam::Concrete(RealOrComplex::Real(f)) => {
                TensorElements::Real(Python::with_gil(|py| {
                    PyFloat::new(py, f).as_unbound().to_owned()
                }))
            }
            ConcreteOrParam::Concrete(RealOrComplex::Complex(c)) => {
                TensorElements::Complex(Python::with_gil(|py| {
                    PyComplex::from_doubles(py, c.re, c.im)
                        .as_unbound()
                        .to_owned()
                }))
            }
            ConcreteOrParam::Param(p) => TensorElements::Symbolica(PythonExpression::from(p)),
        }
    }
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[pymethods]
impl Spensor {
    pub fn structure(&self) -> SpensoIndices {
        SpensoIndices {
            structure: PermutedStructure {
                structure: self.tensor.structure.structure().clone(),
                rep_permutation: self.tensor.rep_permutation.clone(),
                index_permutation: self.tensor.index_permutation.clone(),
            },
        }
    }

    #[staticmethod]
    /// Create a new sparse empty tensor with the given structure and type.
    /// The type is either a float or a symbolica expression.
    ///
    pub fn sparse(
        structure: ConvertibleToStructure,
        type_info: Bound<'_, PyType>,
    ) -> PyResult<Spensor> {
        if type_info.is_subclass_of::<PyFloat>()? {
            Ok(Spensor {
                tensor: structure
                    .0
                    .structure
                    .map_structure(|s| SparseTensor::<f64, _>::empty(s).into()),
            })
        } else if type_info.is_subclass_of::<PythonExpression>()? {
            Ok(Spensor {
                tensor: structure.0.structure.map_structure(|s| {
                    ParamOrConcrete::Param(ParamTensor::from(SparseTensor::<Atom, _>::empty(s)))
                }),
            })
        } else {
            Err(PyTypeError::new_err("Only float type supported"))
        }
    }

    #[staticmethod]
    /// Create a new dense tensor with the given structure and data.
    /// The structure can be a list of integers, a list of representations, or a list of slots.
    /// In the first two cases, no "indices" are assumed, and thus the tensor is indexless (i.e.) it has a shape but no proper way to contract it.
    /// The structure can also be a proper `TensorIndices` object or `TensorStructure` object.
    ///
    /// The data is either a list of floats or a list of symbolica expressions, of length equal to the number of elements in the structure, in row-major order.
    ///
    pub fn dense(structure: ConvertibleToStructure, data: Bound<'_, PyAny>) -> PyResult<Spensor> {
        let dense = if let Ok(d) = data.extract::<Vec<f64>>() {
            DenseTensor::<f64, _>::from_data(d, structure.0.structure.structure)
                .map_err(|e| PyOverflowError::new_err(e.to_string()))?
                .into()
        } else if let Ok(d) = data.extract::<Vec<PythonExpression>>() {
            let data = d.into_iter().map(|e| e.expr).collect();
            ParamOrConcrete::Param(ParamTensor::from(
                DenseTensor::<Atom, _>::from_data(data, structure.0.structure.structure)
                    .map_err(|e| PyOverflowError::new_err(e.to_string()))?,
            ))
        } else {
            return Err(PyTypeError::new_err("Only float type supported"));
        };

        let dense = PermutedStructure {
            structure: dense,
            rep_permutation: structure.0.structure.rep_permutation,
            index_permutation: structure.0.structure.index_permutation,
        };

        Ok(Spensor {
            tensor: dense.permute_inds_wrapped(),
        })
    }
    #[staticmethod]
    pub fn one() -> Spensor {
        Spensor {
            tensor: PermutedStructure::identity(ParamOrConcrete::new_scalar(
                ConcreteOrParam::Concrete(RealOrComplex::Real(1.)),
            )),
        }
    }

    #[staticmethod]
    pub fn zero() -> Spensor {
        Spensor {
            tensor: PermutedStructure::identity(ParamOrConcrete::new_scalar(
                ConcreteOrParam::Concrete(RealOrComplex::Real(2.)),
            )),
        }
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_dense(&mut self) {
        self.tensor.structure = self.tensor.structure.clone().to_dense();
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_sparse(&mut self) {
        self.tensor.structure = self.tensor.structure.clone().to_sparse();
    }

    fn __repr__(&self) -> String {
        format!("Spensor(\n{})", self.tensor)
    }

    fn __str__(&self) -> String {
        format!("{}", self.tensor.structure)
    }

    fn __len__(&self) -> usize {
        self.size().unwrap()
    }

    fn __getitem__(&self, item: SliceOrIntOrExpanded) -> PyResult<Py<PyAny>> {
        let out = match item {
            SliceOrIntOrExpanded::Int(i) => self
                .get_owned_linear(i.into())
                .ok_or(PyIndexError::new_err("flat index out of bounds"))?,
            SliceOrIntOrExpanded::Expanded(idxs) => self
                .get_owned(&idxs)
                .map_err(|s| PyIndexError::new_err(s.to_string()))?,
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

                let slice: Option<Vec<TensorElements>> = range
                    .step_by(step)
                    .map(|i| self.get_owned_linear(i.into()).map(TensorElements::from))
                    .collect();

                if let Some(slice) = slice {
                    return Ok(
                        Python::with_gil(|py| slice.into_pyobject(py).map(|a| a.unbind()))?
                            .into_any(),
                    );
                } else {
                    return Err(PyIndexError::new_err("slice out of bounds"));
                }
            }
        };

        Python::with_gil(|py| {
            TensorElements::from(out)
                .into_pyobject(py)
                .map(|a| a.unbind())
        })
    }

    fn __setitem__<'py>(
        &mut self,
        item: Bound<'py, PyAny>,
        value: Bound<'py, PyAny>,
    ) -> anyhow::Result<()> {
        let value = if let Ok(v) = value.extract::<PythonExpression>() {
            ConcreteOrParam::Param(v.expr)
        } else if let Ok(v) = value.extract::<f64>() {
            ConcreteOrParam::Concrete(RealOrComplex::Real(v))
        } else {
            return Err(anyhow!("Value must be a PythonExpression or a float"));
        };

        if let Ok(flat_index) = item.extract::<usize>() {
            self.tensor.structure.set_flat(flat_index.into(), value)
        } else if let Ok(expanded_idxs) = item.extract::<Vec<usize>>() {
            self.tensor.structure.set(&expanded_idxs, value)
        } else {
            Err(anyhow!("Index must be an integer"))
        }
    }

    #[pyo3(signature =
           (constants,
           funs,
           params,
           iterations = 100,
           n_cores = 4,
           verbose = false),
           )]
    pub fn evaluator(
        &self,
        constants: HashMap<PythonExpression, PythonExpression>,
        funs: HashMap<(Variable, String, Vec<Variable>), PythonExpression>,
        params: Vec<PythonExpression>,
        iterations: usize,
        n_cores: usize,
        verbose: bool,
    ) -> PyResult<SpensoExpressionEvaluator> {
        let mut fn_map = FunctionMap::new();

        for (k, v) in &constants {
            if let Ok(r) = v.expr.clone().try_into() {
                fn_map.add_constant(k.expr.clone(), r);
            } else {
                Err(exceptions::PyValueError::new_err(
                    "Constants must be rationals. If this is not possible, pass the value as a parameter",
                ))?
            }
        }

        for ((symbol, rename, args), body) in &funs {
            let symbol = symbol
                .to_id()
                .ok_or(exceptions::PyValueError::new_err(format!(
                    "Bad function name {}",
                    symbol
                )))?;
            let args: Vec<_> = args
                .iter()
                .map(|x| {
                    x.to_id().ok_or(exceptions::PyValueError::new_err(format!(
                        "Bad function name {}",
                        symbol
                    )))
                })
                .collect::<Result<_, _>>()?;

            fn_map
                .add_function(symbol, rename.clone(), args, body.expr.clone())
                .map_err(|e| {
                    exceptions::PyValueError::new_err(format!("Could not add function: {}", e))
                })?;
        }

        let settings = OptimizationSettings {
            horner_iterations: iterations,
            n_cores,
            verbose,
            ..OptimizationSettings::default()
        };

        let params: Vec<_> = params.iter().map(|x| x.expr.clone()).collect();

        let mut evaltensor = match &self.tensor.structure {
            ParamOrConcrete::Param(s) => s.to_evaluation_tree(&fn_map, &params).map_err(|e| {
                exceptions::PyValueError::new_err(format!("Could not create evaluator: {}", e))
            })?,
            ParamOrConcrete::Concrete(_) => return Err(PyRuntimeError::new_err("not atom")),
        };

        evaltensor.optimize_horner_scheme(&settings);

        evaltensor.common_subexpression_elimination();
        let linear = evaltensor.linearize(None, false);
        Ok(SpensoExpressionEvaluator {
            eval: None,
            eval_complex: linear
                .clone()
                .map_coeff(&|x| Complex::new(x.re.to_f64(), x.im.to_f64())),
            eval_rat: linear,
        })
    }

    fn scalar(&self) -> PyResult<PythonExpression> {
        self.tensor
            .structure
            .clone()
            .scalar()
            .map(|r| PythonExpression { expr: r.into() })
            .ok_or_else(|| PyRuntimeError::new_err("No scalar found"))
    }
}

impl From<DataTensor<f64, ShadowedStructure<AbstractIndex>>> for Spensor {
    fn from(value: DataTensor<f64, ShadowedStructure<AbstractIndex>>) -> Self {
        Spensor {
            tensor: PermutedStructure::identity(MixedTensor::Concrete(RealOrComplexTensor::Real(
                value,
            ))),
        }
    }
}

impl From<DataTensor<Complex<f64>, ShadowedStructure<AbstractIndex>>> for Spensor {
    fn from(value: DataTensor<Complex<f64>, ShadowedStructure<AbstractIndex>>) -> Self {
        Spensor {
            tensor: PermutedStructure::identity(MixedTensor::Concrete(
                RealOrComplexTensor::Complex(value.map_data(|c| c)),
            )),
        }
    }
}
impl From<MixedTensor<f64, ShadowedStructure<AbstractIndex>>> for Spensor {
    fn from(value: MixedTensor<f64, ShadowedStructure<AbstractIndex>>) -> Self {
        Spensor {
            tensor: PermutedStructure::identity(value),
        }
    }
}

/// An optimized evaluator for tensors.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.community.spenso")
)]
#[pyclass(name = "TensorEvaluator")]
#[derive(Clone)]
pub struct SpensoExpressionEvaluator {
    pub eval_rat: LinearizedEvalTensor<SymComplex<Rational>, ShadowedStructure<AbstractIndex>>,
    pub eval: Option<LinearizedEvalTensor<f64, ShadowedStructure<AbstractIndex>>>,
    pub eval_complex: LinearizedEvalTensor<Complex<f64>, ShadowedStructure<AbstractIndex>>,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[pymethods]
impl SpensoExpressionEvaluator {
    /// Evaluate the expression for multiple inputs and return the results.
    fn evaluate(&mut self, inputs: Vec<Vec<f64>>) -> PyResult<Vec<Spensor>> {
        let eval = self.eval.as_mut().ok_or(exceptions::PyValueError::new_err(
            "Evaluator contains complex coefficients. Use evaluate_complex instead.",
        ))?;

        Ok(inputs.iter().map(|s| eval.evaluate(s).into()).collect())
    }

    /// Evaluate the expression for multiple inputs and return the results.
    fn evaluate_complex(&mut self, inputs: Vec<Vec<Complex<f64>>>) -> Vec<Spensor> {
        let eval = &mut self.eval_complex;

        inputs.iter().map(|s| eval.evaluate(s).into()).collect()
    }

    /// Compile the evaluator to a shared library using C++ and optionally inline assembly and load it.
    #[pyo3(signature =
        (function_name,
        filename,
        library_name,
        // number_type,
        inline_asm = "default",
        optimization_level = 3,
        compiler_path = None,
        // compiler_flags = None,
        custom_header = None,
        // cuda_number_of_evaluations = 1,
        // cuda_block_size = 512
    ))]
    #[allow(clippy::too_many_arguments)]
    fn compile(
        &self,
        function_name: &str,
        filename: &str,
        library_name: &str,
        // number_type: &str,
        inline_asm: &str,
        optimization_level: u8,
        compiler_path: Option<&str>,
        // compiler_flags: Option<Vec<String>>,
        custom_header: Option<String>,
        // cuda_number_of_evaluations: usize,
        // cuda_block_size: usize,
        // py: Python<'_>,
    ) -> PyResult<SpensoCompiledExpressionEvaluator> {
        let mut options = CompileOptions {
            optimization_level: optimization_level as usize,
            ..Default::default()
        };

        if let Some(compiler_path) = compiler_path {
            options.compiler = compiler_path.to_string();
        }
        let inline_asm = match inline_asm.to_lowercase().as_str() {
            "default" => InlineASM::default(),
            "x64" => InlineASM::X64,
            "aarch64" => InlineASM::AArch64,
            "none" => InlineASM::None,
            _ => {
                return Err(exceptions::PyValueError::new_err(
                    "Invalid inline assembly type specified.",
                ));
            }
        };

        Ok(SpensoCompiledExpressionEvaluator {
            eval: self
                .eval_complex
                .export_cpp::<Complex<f64>>(
                    filename,
                    function_name,
                    ExportSettings {
                        include_header: true,
                        inline_asm,
                        custom_header,
                        // ..Default::default()
                    },
                )
                .map_err(|e| exceptions::PyValueError::new_err(format!("Export error: {}", e)))?
                .compile(library_name, options)
                .map_err(|e| {
                    exceptions::PyValueError::new_err(format!("Compilation error: {}", e))
                })?
                .load()
                .map_err(|e| {
                    exceptions::PyValueError::new_err(format!("Library loading error: {}", e))
                })?,
        })
    }
}

/// A compiled and optimized evaluator for tensors.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.community.spenso")
)]
#[pyclass(name = "CompiledTensorEvaluator")]
#[derive(Clone)]
pub struct SpensoCompiledExpressionEvaluator {
    pub eval: EvalTensor<CompiledComplexEvaluatorSpenso, ShadowedStructure<AbstractIndex>>,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[pymethods]
impl SpensoCompiledExpressionEvaluator {
    /// Evaluate the expression for multiple inputs and return the results.
    fn evaluate_complex(&mut self, inputs: Vec<Vec<Complex<f64>>>) -> Vec<Spensor> {
        inputs
            .iter()
            .map(|s| self.eval.evaluate(s).into())
            .collect()
    }
}

#[cfg(feature = "python_stubgen")]
define_stub_info_gatherer!(stub_info);
