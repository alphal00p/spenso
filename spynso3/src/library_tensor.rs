use std::ops::Deref;

use anyhow::anyhow;

use pyo3::{
    exceptions::{PyIndexError, PyOverflowError, PyRuntimeError, PyTypeError},
    prelude::*,
    types::{PyFloat, PyType},
};

use spenso::{
    algebra::complex::RealOrComplex,
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
use symbolica::{atom::Atom, domains::float::Complex};

use symbolica::api::python::PythonExpression;

#[cfg(feature = "python_stubgen")]
use pyo3_stub_gen::{define_stub_info_gatherer, derive::*};

use super::{
    ModuleInit, TensorElements,
    structure::{ConvertibleToIndexLess, SpensoStructure},
};

/// A tensor class that can be either dense or sparse.
/// The data is either float or complex or a symbolica expression
/// It can be instantiated with data using the `sparse_empty` or `dense` module functions.
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
    /// Create a new sparse empty tensor with the given structure and type.
    /// The type is either a float or a symbolica expression.
    ///
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
    /// Create a new dense tensor with the given structure and data.
    /// The structure can be a list of integers, a list of representations, or a list of slots.
    /// In the first two cases, no "indices" are assumed, and thus the tensor is indexless (i.e.) it has a shape but no proper way to contract it.
    /// The structure can also be a proper `TensorIndices` object or `TensorStructure` object.
    ///
    /// The data is either a list of floats or a list of symbolica expressions, of length equal to the number of elements in the structure, in row-major order.
    ///
    pub fn dense(structure: ConvertibleToIndexLess, data: Bound<'_, PyAny>) -> PyResult<Self> {
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

        Ok(Self {
            tensor: dense.permute_inds_wrapped(),
        })
    }
    #[staticmethod]
    pub fn one() -> Self {
        Self {
            tensor: PermutedStructure::identity(ParamOrConcrete::new_scalar(
                ConcreteOrParam::Concrete(RealOrComplex::Real(1.)),
            )),
        }
    }

    #[staticmethod]
    pub fn zero() -> Self {
        Self {
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
                RealOrComplexTensor::Complex(value.map_data(|c| c.into())),
            )),
        }
    }
}

#[cfg(feature = "python_stubgen")]
define_stub_info_gatherer!(stub_info);
