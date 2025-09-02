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
    Spensor::init(m)?;
    LibrarySpensor::init(m)?;
    SpensoIndices::init(m)?;
    SpensorLibrary::init(m)?;
    Ok(())
}

/// A tensor class that can be either dense or sparse with flexible data types.
///
/// The tensor can store data as floats, complex numbers, or symbolic expressions (Symbolica atoms).
/// Tensors have an associated structure that defines their shape and index properties.
///
/// # Examples
/// ```python
/// from symbolica.community.spenso import (
///     Tensor,
///     TensorIndices,
///     Representation,
/// )
/// # Create a structure for a 2x2 matrix
/// structure = TensorIndices(Representation.euc(4)(1))  # 4 elements total
/// # Create a dense tensor with float data
/// data = [1.0, 2.0, 3.0, 4.0]
/// tensor = Tensor.dense(structure, data)
/// # Create a sparse tensor
/// sparse_tensor = Tensor.sparse(structure, float)
/// ```
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.community.spenso")
)]
#[pyclass(name = "Tensor", module = "symbolica.community.spenso")]
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
impl PyStubType for SliceOrIntOrExpanded<'_> {
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
    /// Create a new sparse empty tensor with the given structure and data type.
    ///
    /// # Args:
    ///     structure: The tensor structure (TensorIndices or list of Slots)
    ///     type_info: The data type - either `float` or `Expression` class
    ///
    /// # Returns:
    ///     A new sparse tensor with all elements initially zero
    ///
    /// # Examples:
    /// ```python
    /// from symbolica.community.spenso import Tensor, TensorIndices,Representation as R
    ///
    /// # Create structure
    /// structure = TensorIndices(R.euc(3)(1), R.euc(3)(2))  # 3x3 tensor
    ///
    /// # Create sparse float tensor
    /// sparse_float = Tensor.sparse(structure, float)
    ///
    /// # Create sparse symbolic tensor
    /// sparse_sym = Tensor.sparse(structure, symbolica.Expression)
    /// ```
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
    ///
    /// # Args:
    ///     structure: The tensor structure (TensorIndices or list of Slots)
    ///     data: The tensor data in row-major order:
    ///         - List of floats for numerical tensors
    ///         - List of Expressions for symbolic tensors
    ///
    /// # Returns:
    ///     A new dense tensor with the specified data
    ///
    /// # Examples:
    /// ```python
    /// import symbolica as sp
    /// from symbolica import S
    /// from symbolica.community.spenso import (
    ///     Tensor,
    ///     TensorIndices,
    ///     Representation as R,
    /// )
    /// # Create a 2x2 matrix
    /// structure = TensorIndices(R.euc(2)(1), R.euc(2)(2))
    /// data = [1.0, 2.0, 3.0, 4.0]  # [[1,2], [3,4]]
    /// tensor = Tensor.dense(structure, data)
    /// # Create symbolic tensor
    /// x, y = S("x", "y")
    /// sym_data = [x, y, x * y, x + y]
    /// sym_tensor = Tensor.dense(structure, sym_data)
    /// print(tensor)
    /// print(sym_tensor)
    /// ```
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
    /// Create a scalar tensor with value 1.0.
    ///
    /// # Returns:
    ///     A scalar tensor containing the value 1.0
    ///
    /// # Examples:
    /// ```python
    /// from symbolica.community.spenso import Tensor
    ///
    /// one = Tensor.one()
    /// print(one)  # Scalar tensor with value 1.0
    /// ```
    pub fn one() -> Spensor {
        Spensor {
            tensor: PermutedStructure::identity(ParamOrConcrete::new_scalar(
                ConcreteOrParam::Concrete(RealOrComplex::Real(1.)),
            )),
        }
    }

    #[staticmethod]
    /// Create a scalar tensor with value 0.0.
    ///
    /// # Returns:
    ///     A scalar tensor containing the value 0.0
    ///
    /// # Examples:
    /// ```python
    /// from symbolica.community.spenso import Tensor
    ///
    /// zero = Tensor.zero()
    /// print(zero)  # Scalar tensor with value 0.0
    /// ```
    pub fn zero() -> Spensor {
        Spensor {
            tensor: PermutedStructure::identity(ParamOrConcrete::new_scalar(
                ConcreteOrParam::Concrete(RealOrComplex::Real(2.)),
            )),
        }
    }

    #[allow(clippy::wrong_self_convention)]
    /// Convert this tensor to dense storage format.
    ///
    /// Converts sparse tensors to dense format in-place. Dense tensors are unchanged.
    /// This allocates memory for all tensor elements.
    ///
    /// # Examples:
    /// ```python
    /// from symbolica.community.spenso import Tensor, TensorIndices,Representation as R
    ///
    /// structure = TensorIndices(R.euc(4)(2))
    /// tensor = Tensor.sparse(structure, float)
    /// tensor[0] = 1.0
    /// tensor.to_dense()  # Now stores all 4 elements explicitly
    /// ```
    fn to_dense(&mut self) {
        self.tensor.structure = self.tensor.structure.clone().to_dense();
    }

    #[allow(clippy::wrong_self_convention)]
    /// Convert this tensor to sparse storage format.
    ///
    /// Converts dense tensors to sparse format in-place, only storing non-zero elements.
    /// This can save memory for tensors with many zero elements.
    ///
    /// # Examples:
    /// ```python
    /// from symbolica.community.spenso import Tensor, TensorIndices,Representation as R
    ///
    /// structure = TensorIndices(R.euc(2)(2),R.euc(2)(1))
    /// data = [1.0, 0.0, 0.0, 2.0]
    /// tensor = Tensor.dense(structure, data)
    /// tensor.to_sparse()  # Now only stores 2 non-zero elements
    /// print(tensor)
    /// ```
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

    /// Set tensor element(s) at the specified index or indices.
    ///
    /// # Args:
    ///     item: Index specification:
    ///         - int: Flat index into the tensor
    ///         - List[int]: Multi-dimensional index coordinates
    ///     value: The value to set:
    ///         - float: Numerical value
    ///         - Expression: Symbolic expression
    ///
    /// # Examples:
    /// ```python
    /// from symbolica.community.spenso import (
    ///     Tensor,
    ///     TensorIndices,
    ///     Representation as R,
    /// )
    /// structure = TensorIndices(R.euc(2)(2), R.euc(2)(1))
    /// data = [1.0, 0.0, 0.0, 2.0]
    /// tensor = Tensor.dense(structure, data)
    /// tensor.to_sparse()  # Now only stores 2 non-zero elements
    /// print(tensor)
    /// tensor = Tensor.sparse(structure, float)
    /// # Set using flat index
    /// tensor[0] = 4.0
    /// # Set using coordinates
    /// tensor[1, 1] = 1.0
    /// print(tensor)
    /// ```
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
    /// Create an optimized evaluator for symbolic tensor expressions.
    ///
    /// Compiles the symbolic expressions in this tensor into an optimized evaluation tree
    /// that can efficiently compute numerical values for different parameter inputs.
    ///
    /// # Args:
    ///     constants: Dict mapping symbolic expressions to their constant numerical values
    ///     funs: Dict mapping function signatures to their symbolic definitions
    ///     params: List of symbolic parameters that will be varied during evaluation
    ///     iterations: Number of optimization iterations for Horner scheme (default: 100)
    ///     n_cores: Number of CPU cores to use for optimization (default: 4)
    ///     verbose: Whether to print optimization progress (default: False)
    ///
    /// # Returns:
    ///     A TensorEvaluator object for efficient numerical evaluation
    ///
    /// # Examples:
    /// ```python
    /// from symbolica import S
    /// from symbolica.community.spenso import (
    ///     Tensor,
    ///     TensorIndices,
    ///     Representation as R,
    /// )
    /// # Create symbolic tensor
    /// x, y = S("x", "y")
    /// structure = TensorIndices(R.euc(2)(1))
    /// tensor = Tensor.dense(structure, [x * y, x + y])
    /// # Create evaluator
    /// evaluator = tensor.evaluator(
    ///     constants={}, funs={}, params=[x, y], iterations=50
    /// )
    /// # Use evaluator
    /// results = evaluator.evaluate_complex([[1.0, 2.0], [3.0, 4.0]])
    /// print(results)
    /// ```
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

    /// Extract the scalar value from a rank-0 (scalar) tensor.
    ///
    /// # Returns:
    ///     The scalar expression contained in this tensor
    ///
    /// # Raises:
    ///     RuntimeError: If the tensor is not a scalar
    ///
    /// # Examples:
    /// ```python
    /// from symbolica.community.spenso import Tensor
    ///
    /// scalar_tensor = Tensor.one()
    /// value = scalar_tensor.scalar()  # Returns expression "1"
    /// ```
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

/// An optimized evaluator for symbolic tensor expressions.
///
/// This class provides efficient numerical evaluation of symbolic tensor expressions
/// after optimization. It supports both real and complex-valued evaluations.
///
/// Create instances using the `Tensor.evaluator()` method rather than directly.
///
/// # Examples:
/// ```python
/// # Created from a symbolic tensor
/// evaluator = my_tensor.evaluator(constants={}, funs={}, params=[x, y])
///
/// # Evaluate for multiple parameter sets
/// results = evaluator.evaluate([[1.0, 2.0], [3.0, 4.0]])
/// ```
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.community.spenso")
)]
#[pyclass(name = "TensorEvaluator", module = "symbolica.community.spenso")]
#[derive(Clone)]
pub struct SpensoExpressionEvaluator {
    pub eval_rat: LinearizedEvalTensor<SymComplex<Rational>, ShadowedStructure<AbstractIndex>>,
    pub eval: Option<LinearizedEvalTensor<f64, ShadowedStructure<AbstractIndex>>>,
    pub eval_complex: LinearizedEvalTensor<Complex<f64>, ShadowedStructure<AbstractIndex>>,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[pymethods]
impl SpensoExpressionEvaluator {
    /// Evaluate the tensor expression for multiple real-valued parameter inputs.
    ///
    /// # Args:
    ///     inputs: List of parameter value lists, where each inner list contains
    ///             numerical values for all parameters in the same order as specified
    ///             when creating the evaluator
    ///
    /// # Returns:
    ///     List of evaluated tensors, one for each input parameter set
    ///
    /// # Raises:
    ///     ValueError: If the evaluator contains complex coefficients
    ///
    /// # Examples:
    /// ```python
    /// # Evaluate for two different parameter sets
    /// results = evaluator.evaluate([
    ///     [1.0, 2.0],  # x=1.0, y=2.0
    ///     [3.0, 4.0]   # x=3.0, y=4.0
    /// ])
    /// ```
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

    /// Compile the evaluator to a shared library using C++ for maximum performance.
    ///
    /// Generates optimized C++ code with optional inline assembly and compiles it
    /// into a shared library that can be loaded for extremely fast evaluation.
    ///
    /// # Args:
    ///     function_name: Name for the generated C++ function
    ///     filename: Path for the generated C++ source file
    ///     library_name: Name for the compiled shared library
    ///     inline_asm: Type of inline assembly optimization:
    ///                  - "default": Platform-appropriate assembly
    ///                  - "x64": x86-64 specific optimizations
    ///                  - "aarch64": ARM64 specific optimizations
    ///                  - "none": No inline assembly
    ///     optimization_level: Compiler optimization level 0-3 (default: 3)
    ///     compiler_path: Path to specific C++ compiler (default: system default)
    ///     custom_header: Additional C++ header code to include
    ///
    /// # Returns:
    ///     A CompiledTensorEvaluator for maximum performance evaluation
    ///
    /// # Examples:
    /// ```python
    /// from symbolica import S
    /// from symbolica.community.spenso import (
    ///     Tensor,
    ///     TensorIndices,
    ///     Representation as R,
    /// )
    /// # Create symbolic tensor
    /// x, y = S("x", "y")
    /// structure = TensorIndices(R.euc(2)(1))
    /// tensor = Tensor.dense(structure, [x * y, x + y])
    /// # Create evaluator
    /// evaluator = tensor.evaluator(
    ///     constants={}, funs={}, params=[x, y], iterations=50
    /// )
    /// inputs = [[1.0, 2.0], [3.0, 4.0]]
    /// # Use evaluator
    /// results = evaluator.evaluate_complex(inputs)
    /// print(results)
    /// compiled = evaluator.compile(
    ///     function_name="fast_eval",
    ///     filename="tensor_eval.cpp",
    ///     library_name="tensor_lib",
    ///     optimization_level=3,
    /// )
    /// # Use compiled evaluator
    /// results = compiled.evaluate_complex(inputs)
    /// ```
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

/// A compiled and optimized evaluator for maximum performance tensor evaluation.
///
/// This class wraps a compiled C++ shared library for extremely fast numerical
/// evaluation of tensor expressions. It only supports complex-valued evaluation
/// as this is the most general case.
///
/// Create instances using the `TensorEvaluator.compile()` method.
///
/// # Examples:
/// ```python
/// # Created from a TensorEvaluator
/// compiled = evaluator.compile("eval_func", "code.cpp", "lib")
///
/// # Use for high-performance evaluation
/// results = compiled.evaluate_complex(large_input_batch)
/// ```
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.community.spenso")
)]
#[pyclass(
    name = "CompiledTensorEvaluator",
    module = "symbolica.community.spenso"
)]
#[derive(Clone)]
pub struct SpensoCompiledExpressionEvaluator {
    pub eval: EvalTensor<CompiledComplexEvaluatorSpenso, ShadowedStructure<AbstractIndex>>,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[pymethods]
impl SpensoCompiledExpressionEvaluator {
    /// Evaluate the tensor expression for multiple complex-valued parameter inputs.
    ///
    /// Uses the compiled C++ code for maximum performance evaluation with complex numbers.
    ///
    /// # Args:
    ///     inputs: List of parameter value lists, where each inner list contains
    ///             complex values for all parameters in the same order as specified
    ///             when creating the original evaluator
    ///
    /// # Returns:
    ///     List of evaluated tensors, one for each input parameter set
    ///
    /// # Examples:
    /// ```python
    /// import symbolica as sp
    /// from symbolica.community.spenso import Tensor
    ///
    /// # Use compiled evaluator for complex inputs
    /// complex_inputs = [
    ///     [1.0+2.0j, 3.0+0.0j],  # x=1+2i, y=3
    ///     [0.0+1.0j, 2.0+1.0j]   # x=i, y=2+i
    /// ]
    /// results = compiled_evaluator.evaluate_complex(complex_inputs)
    /// ```
    fn evaluate_complex(&mut self, inputs: Vec<Vec<Complex<f64>>>) -> Vec<Spensor> {
        inputs
            .iter()
            .map(|s| self.eval.evaluate(s).into())
            .collect()
    }
}

#[cfg(feature = "python_stubgen")]
define_stub_info_gatherer!(stub_info);
