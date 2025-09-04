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
use symbolica::atom::Atom;
use symbolica::{
    api::python::PythonExpression,
    atom::{AtomView, Symbol},
    try_symbol,
};

use crate::structure::SpensoName;

use super::{Spensor, library_tensor::LibrarySpensor, structure::SpensoStructure};

use super::ModuleInit;

/// A library for registering and managing tensor templates and structures.
///
/// The TensorLibrary provides a centralized registry for tensor definitions that can be
/// reused across tensor networks and expressions. It manages tensor structures with their
/// associated names and can resolve symbolic references to registered tensors.
///
/// ```python
/// import symbolica
/// from symbolica.community.spenso import TensorLibrary, LibraryTensor, TensorStructure, Representation
///
/// lib = TensorLibrary()
/// rep = Representation.euc(3)
/// name = symbolica.S("my_tensor")
/// structure = TensorStructure(rep, rep, name=name)
/// tensor = LibraryTensor.dense(structure, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
/// lib.register(tensor)
/// tensor_ref = lib[name]
/// ```
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
        } else if let Ok(num) = ob.extract::<SpensoName>() {
            Ok(ConvertibleToSymbol::Symbol(PythonExpression {
                expr: Atom::var(num.name),
            }))
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

#[allow(clippy::new_without_default)]
#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[pymethods]
impl SpensorLibrary {
    #[new]
    /// Create a new empty tensor library.
    ///
    /// Initializes an empty library ready for registering tensor structures.
    /// The library automatically manages internal tensor IDs.
    ///
    /// Returns
    /// -------
    /// TensorLibrary
    ///     A new empty tensor library
    ///
    /// Examples
    /// --------
    /// >>> from symbolica.community.spenso import TensorLibrary
    /// >>> lib = TensorLibrary()
    pub fn new() -> Self {
        let mut a = Self {
            library: TensorLibrary::new(),
        };
        a.library.update_ids();
        a
    }

    #[staticmethod]
    /// Create a new empty tensor library (static method).
    ///
    /// Initializes an empty library ready for registering tensor structures.
    /// The library automatically manages internal tensor IDs.
    ///
    /// Returns
    /// -------
    /// TensorLibrary
    ///     A new empty tensor library
    ///
    /// Examples
    /// --------
    /// >>> from symbolica.community.spenso import TensorLibrary
    /// >>> lib = TensorLibrary.construct()
    pub fn construct() -> Self {
        Self::new()
    }

    /// Register a tensor in the library.
    ///
    /// Adds a tensor template to the library that can be referenced by name
    /// in tensor networks and symbolic expressions. The tensor must have a name
    /// set in its structure.
    ///
    /// Parameters
    /// ----------
    /// tensor : LibraryTensor or Tensor
    ///     The tensor to register - can be a LibraryTensor or regular Tensor
    ///
    /// Examples
    /// --------
    /// >>> import symbolica
    /// >>> from symbolica.community.spenso import TensorLibrary, LibraryTensor, TensorStructure, Representation
    /// >>> lib = TensorLibrary()
    /// >>> rep = Representation.euc(3)
    /// >>> name = symbolica.S("my_tensor")
    /// >>> structure = TensorStructure(rep, rep, name=name)
    /// >>> tensor = LibraryTensor.dense(structure, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    /// >>> lib.register(tensor)
    /// >>> tensor_ref = lib[name]
    pub fn register(&mut self, tensor: ConvertibleToLibraryTensor) -> PyResult<()> {
        self.library.insert_explicit(tensor.0.tensor);
        Ok(())
    }

    /// Retrieve a registered tensor structure by name.
    ///
    /// Looks up a previously registered tensor by its name and returns
    /// a reference structure that can be used to create new tensor instances.
    ///
    /// Parameters
    /// ----------
    /// key : str or Expression
    ///     The tensor name - can be a string or symbolic expression
    ///
    /// Returns
    /// -------
    /// TensorStructure
    ///     A TensorStructure representing the registered tensor template
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If the tensor name is not found in the library
    ///
    /// Examples
    /// --------
    /// >>> structure = lib["T"]
    pub fn __getitem__(&self, key: ConvertibleToSymbol) -> anyhow::Result<SpensoStructure> {
        let symbol = key.symbol()?;
        let key = self.library.get_key_from_name(symbol)?;

        Ok(SpensoStructure {
            structure: PermutedStructure::identity(key),
        })
    }

    #[staticmethod]
    /// Create a library pre-loaded with High Energy Physics tensor definitions.
    ///
    /// Returns a library containing standard HEP tensors such as gamma matrices,
    /// color generators, metric tensors, and other commonly used structures in
    /// particle physics calculations.
    ///
    /// Returns
    /// -------
    /// TensorLibrary
    ///     A TensorLibrary pre-populated with HEP tensor definitions
    ///
    /// Examples
    /// --------
    /// >>> import symbolica
    /// >>> from symbolica.community.spenso import TensorLibrary, TensorName
    /// >>> hep_lib = TensorLibrary.hep_lib()
    /// >>> gamma_structure = hep_lib[symbolica.S("spenso::gamma")]
    pub fn hep_lib() -> Self {
        Self {
            library: spenso_hep_lib::hep_lib(1., 0.),
        }
    }
}

/// Enumeration for different tensor namespaces in physics.
///
/// Provides categorization for different types of tensor operations and structures
/// commonly used in theoretical physics calculations.
///
/// # Variants:
/// - Weyl: Tensors related to Weyl spinors and chiral representations
/// - Algebra: Tensors related to algebraic structures and Lie algebras
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
