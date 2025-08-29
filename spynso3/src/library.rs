use anyhow::anyhow;
use pyo3::{Bound, FromPyObject, exceptions, types::PyAnyMethods};

use pyo3::{PyResult, pyclass, pymethods};

#[cfg(feature = "python_stubgen")]
use pyo3_stub_gen::{
    PyStubType,
    derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods},
};
use spenso::{
    network::library::symbolic::{ExplicitKey, TensorLibrary},
    structure::{HasStructure, PermutedStructure, abstract_index::AbstractIndex},
    tensors::parametric::MixedTensor,
};
use symbolica::{
    api::python::PythonExpression,
    atom::{AtomView, Symbol},
    try_symbol,
};

use super::{Spensor, library_tensor::LibrarySpensor, structure::SpensoStructure};

use super::ModuleInit;

#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.community.spenso")
)]
#[pyclass(name = "TensorLibrary", module = "symbolica.community.spenso")]
pub struct SpensorLibrary {
    pub(crate) library: TensorLibrary<MixedTensor<f64, ExplicitKey<AbstractIndex>>, AbstractIndex>,
}

impl ModuleInit for SpensorLibrary {}

pub enum ConvertibleToSymbol {
    Name(String),
    Symbol(PythonExpression),
}

impl ConvertibleToSymbol {
    fn symbol(&self) -> anyhow::Result<Symbol> {
        match self {
            ConvertibleToSymbol::Name(name) => Ok(try_symbol!(name).map_err(|e| anyhow!(e))?),
            ConvertibleToSymbol::Symbol(symbol) => {
                if let AtomView::Var(a) = symbol.as_view() {
                    Ok(a.get_symbol())
                } else {
                    Err(anyhow::anyhow!("Symbol is not a variable"))
                }
            }
        }
    }
}

impl<'a> FromPyObject<'a> for ConvertibleToSymbol {
    fn extract_bound(ob: &Bound<'a, pyo3::PyAny>) -> PyResult<Self> {
        if let Ok(a) = ob.extract::<String>() {
            Ok(ConvertibleToSymbol::Name(a))
        } else if let Ok(num) = ob.extract::<PythonExpression>() {
            Ok(ConvertibleToSymbol::Symbol(num))
        } else {
            Err(exceptions::PyTypeError::new_err(
                "Cannot convert to expression",
            ))
        }
    }
}

#[cfg(feature = "python_stubgen")]
impl PyStubType for ConvertibleToSymbol {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        symbolica::api::python::ConvertibleToExpression::type_output() | String::type_input()
    }
}

pub struct ConvertibleToLibraryTensor(LibrarySpensor);

impl<'a> FromPyObject<'a> for ConvertibleToLibraryTensor {
    fn extract_bound(ob: &Bound<'a, pyo3::PyAny>) -> PyResult<Self> {
        if let Ok(a) = ob.extract::<LibrarySpensor>() {
            Ok(ConvertibleToLibraryTensor(a))
        } else if let Ok(num) = ob.extract::<Spensor>() {
            Ok(ConvertibleToLibraryTensor(LibrarySpensor {
                tensor: num.tensor.map_structure(|a| a.map_structure(Into::into)),
            }))
        } else {
            Err(exceptions::PyTypeError::new_err(
                "Cannot convert to library tensor",
            ))
        }
    }
}

#[cfg(feature = "python_stubgen")]
impl PyStubType for ConvertibleToLibraryTensor {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        Spensor::type_output() | LibrarySpensor::type_input()
    }
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[pymethods]
impl SpensorLibrary {
    #[new]
    pub fn constuct() -> Self {
        let mut a = Self {
            library: TensorLibrary::new(),
        };
        a.library.update_ids();
        a
    }

    pub fn register(&mut self, tensor: ConvertibleToLibraryTensor) -> PyResult<()> {
        self.library.insert_explicit(tensor.0.tensor);
        Ok(())
    }

    pub fn __getitem__(&self, key: ConvertibleToSymbol) -> anyhow::Result<SpensoStructure> {
        let symbol = key.symbol()?;
        let key = self.library.get_key_from_name(symbol)?;

        Ok(SpensoStructure {
            structure: PermutedStructure::identity(key),
        })
    }

    #[staticmethod]
    pub fn hep_lib() -> Self {
        Self {
            library: spenso_hep_lib::hep_lib(1., 0.),
        }
    }
}

#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass_enum(module = "symbolica.community.spenso")
)]
#[pyclass(eq, eq_int, module = "symbolica.community.spenso")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorNamespace {
    Weyl,
    Algebra,
}

#[cfg(feature = "python_stubgen")]
pyo3_stub_gen::define_stub_info_gatherer!(stub_info);
