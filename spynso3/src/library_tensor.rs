use std::ops::Deref;

use anyhow::anyhow;

use pyo3::{
    exceptions::{PyIndexError, PyOverflowError, PyRuntimeError, PyTypeError},
    prelude::*,
    types::{PyFloat, PyType},
};

#[cfg(feature = "python_stubgen")]
use pyo3_stub_gen::{
    PyStubType, TypeInfo,
    generate::MethodType,
    impl_stub_type,
    inventory::submit,
    type_info::{ArgInfo, MethodInfo, PyMethodsInfo},
};
use spenso::{
    algebra::complex::{Complex, RealOrComplex},
    tensors::{
        data::{DenseTensor, GetTensorData, SetTensorData, SparseOrDense, SparseTensor},
        parametric::{ConcreteOrParam, ParamOrConcrete, ParamTensor},
    },
};

use crate::SliceOrIntOrExpanded;
use spenso::{
    network::library::symbolic::ExplicitKey,
    structure::{
        HasStructure, PermutedStructure, ScalarTensor, TensorStructure,
        abstract_index::AbstractIndex, permuted::Perm,
    },
    tensors::{
        complex::RealOrComplexTensor,
        data::{DataTensor, StorageTensor},
        parametric::MixedTensor,
    },
};
use symbolica::atom::Atom;

use symbolica::api::python::PythonExpression;

#[cfg(feature = "python_stubgen")]
use pyo3_stub_gen::{define_stub_info_gatherer, derive::*};

use super::{
    ModuleInit, TensorElements,
    structure::{ConvertibleToIndexLess, SpensoStructure},
};

/// A library tensor class optimized for use in tensor libraries and networks.
///
/// Library tensors are similar to regular tensors but use explicit keys for efficient
/// lookup and storage in tensor libraries. They can be either dense or sparse and
/// store data as floats, complex numbers, or symbolic expressions.
///
/// LibraryTensors are designed for:
/// - Registration in TensorLibrary instances
/// - Use in tensor networks where structure reuse is important
/// - Efficient symbolic manipulation and pattern matching
///
/// # Examples:
/// ```python
/// from symbolica.community.spenso import LibraryTensor, TensorStructure, Representation
///
/// # Create a structure for a color matrix
/// rep = Representation.euc(3)  # 3x3 color fundamental
/// structure = TensorStructure(rep, rep, name="T")
///
/// # Create dense library tensor
/// data = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # Identity matrix
/// tensor = LibraryTensor.dense(structure, data)
///
/// # Create sparse library tensor
/// sparse_tensor = LibraryTensor.sparse(structure, float)
/// ```
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.community.spenso")
)]
#[pyclass(name = "LibraryTensor", module = "symbolica.community.spenso")]
#[derive(Clone)]
pub struct LibrarySpensor {
    pub tensor: PermutedStructure<MixedTensor<f64, ExplicitKey<AbstractIndex>>>,
}

impl Deref for LibrarySpensor {
    type Target = MixedTensor<f64, ExplicitKey<AbstractIndex>>;

    fn deref(&self) -> &Self::Target {
        &self.tensor.structure
    }
}

impl ModuleInit for LibrarySpensor {}

pub enum AtomsOrFloats {
    Atoms(Vec<Atom>),
    Floats(Vec<f64>),
    Complex(Vec<Complex<f64>>),
}

impl<'py> FromPyObject<'py> for AtomsOrFloats {
    fn extract_bound(aind: &Bound<'py, PyAny>) -> PyResult<Self> {
        let aind = if let Ok(i) = aind.extract::<Vec<f64>>() {
            AtomsOrFloats::Floats(i)
        } else if let Ok(i) = aind.extract::<Vec<Complex<f64>>>() {
            AtomsOrFloats::Complex(i)
        } else if let Ok(i) = aind.extract::<Vec<PythonExpression>>() {
            AtomsOrFloats::Atoms(i.into_iter().map(|e| e.expr).collect())
        } else {
            return Err(PyTypeError::new_err(
                "Argument must be a list of floats, complex numbers, or Atoms",
            ));
        };
        Ok(aind)
    }
}

#[cfg(feature = "python_stubgen")]
impl_stub_type!(AtomsOrFloats = Vec<PythonExpression> | Vec<f64> | Vec<Complex<f64>>);

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[pymethods]
impl LibrarySpensor {
    pub fn structure(&self) -> SpensoStructure {
        SpensoStructure {
            structure: PermutedStructure {
                structure: self.tensor.structure.structure().clone(),
                rep_permutation: self.tensor.rep_permutation.clone(),
                index_permutation: self.tensor.index_permutation.clone(),
            },
        }
    }

    #[staticmethod]
    /// Create a new sparse empty library tensor with the given structure and data type.
    ///
    /// Creates a sparse tensor that initially contains no non-zero elements.
    /// Elements can be set individually using indexing operations.
    ///
    /// # Parameters:
    /// - structure: The tensor structure (TensorStructure, list of Representations, or list of integers)
    /// - type_info: The data type - either `float` or `Expression` class
    ///
    /// # Examples:
    /// ```python
    /// import symbolica as sp
    /// from symbolica.community.spenso import LibraryTensor, TensorStructure, Representation
    ///
    /// # Create structure from representations
    /// rep = Representation.euc(3)
    /// structure = TensorStructure(rep, rep)
    ///
    /// # Create sparse float tensor
    /// sparse_float = LibraryTensor.sparse(structure, float)
    ///
    /// # Create sparse symbolic tensor
    /// sparse_sym = LibraryTensor.sparse(structure, sp.Expression)
    ///
    /// # Set individual elements
    /// sparse_float[0, 0] = 1.0
    /// sparse_float[1, 1] = 2.0
    /// ```
    pub fn sparse(
        structure: ConvertibleToIndexLess,
        type_info: Bound<'_, PyType>,
    ) -> PyResult<Self> {
        if type_info.is_subclass_of::<PyFloat>()? {
            Ok(Self {
                tensor: structure
                    .0
                    .structure
                    .map_structure(|s| SparseTensor::<f64, _>::empty(s).into()),
            })
        } else if type_info.is_subclass_of::<PythonExpression>()? {
            Ok(Self {
                tensor: structure.0.structure.map_structure(|s| {
                    ParamOrConcrete::Param(ParamTensor::from(SparseTensor::<Atom, _>::empty(s)))
                }),
            })
        } else {
            Err(PyTypeError::new_err("Only float type supported"))
        }
    }

    #[staticmethod]
    /// Create a new dense library tensor with the given structure and data.
    ///
    /// Dense tensors store all elements explicitly in row-major order. The structure
    /// defines the tensor's shape and indexing properties.
    ///
    /// # Parameters:
    /// - structure: The tensor structure (TensorStructure, list of Representations, or list of integers)
    /// - data: The tensor data in row-major order (list of floats, complex numbers, or Expressions)
    ///
    /// # Examples:
    /// ```python
    /// from symbolica import S
    /// from symbolica.community.spenso import LibraryTensor, TensorStructure, Representation
    ///
    /// # Create a 2x2 color matrix
    /// rep = Representation.euc(2)
    /// sigma = S("sigma")
    /// structure = TensorStructure(rep, rep, name=sigma)
    /// data = [0.0, 1.0, 1.0, 0.0]  # Pauli matrix σ_x
    /// tensor = LibraryTensor.dense(structure, data)
    ///
    /// # Create symbolic tensor
    /// x, y = S("x", "y")
    /// sym_data = [x, y, -y, x]  # 2x2 matrix with symbolic entries
    /// sym_tensor = LibraryTensor.dense(structure, sym_data)
    /// ```
    pub fn dense(structure: ConvertibleToIndexLess, data: AtomsOrFloats) -> PyResult<Self> {
        let dense = match data {
            AtomsOrFloats::Floats(f) => {
                DenseTensor::<f64, _>::from_data(f, structure.0.structure.structure)
                    .map_err(|e| PyOverflowError::new_err(e.to_string()))?
                    .into()
            }
            AtomsOrFloats::Atoms(a) => ParamOrConcrete::Param(ParamTensor::from(
                DenseTensor::<Atom, _>::from_data(a, structure.0.structure.structure)
                    .map_err(|e| PyOverflowError::new_err(e.to_string()))?,
            )),
            AtomsOrFloats::Complex(c) => {
                MixedTensor::Concrete(RealOrComplexTensor::Complex(DataTensor::Dense(
                    DenseTensor::<Complex<f64>, _>::from_data(c, structure.0.structure.structure)
                        .map_err(|e| PyOverflowError::new_err(e.to_string()))?,
                )))
            }
        };

        let dense = PermutedStructure {
            structure: dense,
            rep_permutation: structure.0.structure.rep_permutation,
            index_permutation: structure.0.structure.index_permutation,
        };

        Ok(Self {
            tensor: dense.permute_inds_wrapped(),
        })
    }
    #[staticmethod]
    /// Create a scalar library tensor with value 1.0.
    ///
    /// # Examples:
    /// ```python
    /// from symbolica.community.spenso import LibraryTensor
    ///
    /// one = LibraryTensor.one()
    /// ```
    pub fn one() -> Self {
        Self {
            tensor: PermutedStructure::identity(ParamOrConcrete::new_scalar(
                ConcreteOrParam::Concrete(RealOrComplex::Real(1.)),
            )),
        }
    }

    #[staticmethod]
    /// Create a scalar library tensor with value 0.0.
    ///
    /// # Examples:
    /// ```python
    /// from symbolica.community.spenso import LibraryTensor
    ///
    /// zero = LibraryTensor.zero()
    /// ```
    pub fn zero() -> Self {
        Self {
            tensor: PermutedStructure::identity(ParamOrConcrete::new_scalar(
                ConcreteOrParam::Concrete(RealOrComplex::Real(2.)),
            )),
        }
    }

    #[allow(clippy::wrong_self_convention)]
    /// Convert this library tensor to dense storage format.
    ///
    /// Converts sparse tensors to dense format in-place. Dense tensors are unchanged.
    /// This allocates memory for all tensor elements.
    ///
    /// # Examples:
    /// ```python
    /// from symbolica.community.spenso import LibraryTensor, TensorStructure, Representation
    ///
    /// rep = Representation.cof(2)
    /// structure = TensorStructure([rep, rep])
    /// tensor = LibraryTensor.sparse(structure, float)
    /// tensor[0, 0] = 1.0
    /// tensor.to_dense()  # Now stores all 4 elements explicitly
    /// ```
    fn to_dense(&mut self) {
        self.tensor.structure = self.tensor.structure.clone().to_dense();
    }

    #[allow(clippy::wrong_self_convention)]
    /// Convert this library tensor to sparse storage format.
    ///
    /// Converts dense tensors to sparse format in-place, only storing non-zero elements.
    /// This can save memory for tensors with many zero elements.
    ///
    /// # Examples:
    /// ```python
    /// from symbolica.community.spenso import LibraryTensor, TensorStructure, Representation
    ///
    /// rep = Representation.euc(2)
    /// structure = TensorStructure(rep, rep)
    /// data = [1.0, 0.0, 0.0, 2.0]
    /// tensor = LibraryTensor.dense(structure, data)
    /// tensor.to_sparse()  # Now only stores 2 non-zero elements
    /// ```
    fn to_sparse(&mut self) {
        self.tensor.structure = self.tensor.structure.clone().to_sparse();
    }

    fn __repr__(&self) -> String {
        format!("Spensor(\n{})", self.tensor)
    }

    fn __str__(&self) -> String {
        format!("{}", self.tensor)
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

    /// Set library tensor element(s) at the specified index or indices.
    ///
    /// # Parameters:
    /// - item: Index specification (int for flat index, List[int] for coordinates)
    /// - value: The value to set (float or Expression)
    ///
    /// # Examples:
    /// ```python
    /// from symbolica.community.spenso import LibraryTensor, TensorStructure, Representation
    ///
    /// rep = Representation.euc(2)
    /// structure = TensorStructure(rep, rep)
    /// tensor = LibraryTensor.sparse(structure, float)
    ///
    /// # Set using flat index
    /// tensor[0] = 1.0
    ///
    /// # Set using coordinates
    /// tensor[1, 1] = 2.0
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

    /// Extract the scalar value from a rank-0 (scalar) library tensor.
    ///
    /// # Examples:
    /// ```python
    /// from symbolica.community.spenso import LibraryTensor
    ///
    /// scalar_tensor = LibraryTensor.one()
    /// value = scalar_tensor.scalar()
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

impl From<DataTensor<f64, ExplicitKey<AbstractIndex>>> for LibrarySpensor {
    fn from(value: DataTensor<f64, ExplicitKey<AbstractIndex>>) -> Self {
        LibrarySpensor {
            tensor: PermutedStructure::identity(MixedTensor::Concrete(RealOrComplexTensor::Real(
                value,
            ))),
        }
    }
}

impl From<DataTensor<Complex<f64>, ExplicitKey<AbstractIndex>>> for LibrarySpensor {
    fn from(value: DataTensor<Complex<f64>, ExplicitKey<AbstractIndex>>) -> Self {
        LibrarySpensor {
            tensor: PermutedStructure::identity(MixedTensor::Concrete(
                RealOrComplexTensor::Complex(value.map_data(|c| c)),
            )),
        }
    }
}

#[cfg(feature = "python_stubgen")]
submit! {
    PyMethodsInfo {
        struct_id: std::any::TypeId::of::<LibrarySpensor>,
        attrs: &[],
        getters: &[],
        setters: &[],
        methods: &[
            MethodInfo {
                name: "__getitem__",
                args: &[
                    ArgInfo {
                        name: "item",
                        signature: None,
                        r#type: || TypeInfo::builtin("slice"),
                    },
                ],
                r#type: MethodType::Instance,
                r#return: Vec::<TensorElements>::type_output,
                doc:r##"Get library tensor elements at the specified range of indices.

# Parameters:
- item: Slice object defining the range of indices

Returns list of complex numbers, floats, or Expressions.
"##,
                is_async: false,
                deprecated: None,
                type_ignored: None,
            },
            MethodInfo {
                name: "__getitem__",
                args: &[
                    ArgInfo {
                        name: "item",
                        signature: None,
                        r#type: || Vec::<usize>::type_input()|usize::type_input()
                    },
                ],
                r#type: MethodType::Instance,
                r#return: TensorElements::type_output,
                doc:r##"Get library tensor element at the specified index or indices.

# Parameters:
- item: Index specification (int for flat index, List[int] for coordinates)

Returns complex number, float, or Expression.
"##,
                is_async: false,
                deprecated: None,
                type_ignored: None,
            },
            MethodInfo {
                name: "__setitem__",
                args: &[
                    ArgInfo {
                        name: "item",
                        signature: None,
                        r#type: ||Vec::<usize>::type_input()|usize::type_input()
                    },
                    ArgInfo {
                        name: "value",
                        signature: None,
                        r#type: ||TensorElements::type_input()
                    },
                ],
                r#type: MethodType::Instance,
                r#return: TypeInfo::none,
                doc:r##"Set library tensor element(s) at the specified index or indices.

# Parameters:
- item: Index specification (int for flat index, List[int] for coordinates)
- value: The value to set (float or Expression)

# Examples:
```python
from symbolica.community.spenso import LibraryTensor, TensorStructure, Representation

rep = Representation.euc(2)
structure = TensorStructure(rep, rep)
tensor = LibraryTensor.sparse(structure, float)

# Set using flat index
tensor[0] = 1.0

# Set using coordinates
tensor[1, 1] = 2.0
```
"##,
                is_async: false,
                deprecated: None,
                type_ignored: None,
            },
        ]
    }
}

#[cfg(feature = "python_stubgen")]
define_stub_info_gatherer!(stub_info);
