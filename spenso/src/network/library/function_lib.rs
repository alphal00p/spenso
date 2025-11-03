use std::collections::HashMap;

use anyhow::anyhow;
use symbolica::{atom::Symbol, function};

use crate::{
    network::library::{FunctionLibrary, FunctionLibraryError},
    structure::{HasStructure, TensorStructure},
    tensors::{
        data::StorageTensor,
        parametric::{to_param::ToParam, ParamOrConcrete, ParamTensor},
    },
};

pub struct SymbolLib<T, Missing> {
    pub functions: HashMap<Symbol, Box<dyn Fn(T) -> T + Send + Sync>>,
    pub _missing: Missing,
}

impl<T, Missing> SymbolLib<T, Missing> {
    pub fn insert<F>(&mut self, key: Symbol, func: F)
    where
        F: Fn(T) -> T + Send + Sync + 'static,
    {
        self.functions.insert(key, Box::new(func));
    }
}

pub struct Panic;
impl Panic {
    pub fn new_lib<T>() -> SymbolLib<T, Self> {
        SymbolLib {
            functions: HashMap::new(),
            _missing: Self,
        }
    }
}
pub struct Wrap;
impl Wrap {
    pub fn new_lib<T>() -> SymbolLib<T, Self> {
        SymbolLib {
            functions: HashMap::new(),
            _missing: Self,
        }
    }
}

impl<S: TensorStructure> FunctionLibrary<ParamTensor<S>> for SymbolLib<ParamTensor<S>, Panic> {
    type Key = Symbol;

    fn apply(
        &self,
        key: &Self::Key,
        tensor: ParamTensor<S>,
    ) -> Result<ParamTensor<S>, FunctionLibraryError<Symbol>> {
        if let Some(func) = self.functions.get(key) {
            Ok(func(tensor))
        } else {
            Err(FunctionLibraryError::NotFound(*key))
        }
    }
}

impl<S: TensorStructure + Clone> FunctionLibrary<ParamTensor<S>>
    for SymbolLib<ParamTensor<S>, Wrap>
{
    type Key = Symbol;

    fn apply(
        &self,
        key: &Self::Key,
        tensor: ParamTensor<S>,
    ) -> Result<ParamTensor<S>, FunctionLibraryError<Symbol>> {
        Ok(if let Some(func) = self.functions.get(key) {
            func(tensor)
        } else {
            tensor.map_data_self(|a| function!(*key, a))
        })
    }
}

impl<S: TensorStructure + Clone, C: ToParam + HasStructure<Structure = S>>
    FunctionLibrary<ParamOrConcrete<C, S>> for SymbolLib<C, Wrap>
{
    type Key = Symbol;

    fn apply(
        &self,
        key: &Self::Key,
        tensor: ParamOrConcrete<C, S>,
    ) -> Result<ParamOrConcrete<C, S>, FunctionLibraryError<Symbol>> {
        Ok(match tensor {
            ParamOrConcrete::Concrete(c) => {
                if let Some(func) = self.functions.get(key) {
                    ParamOrConcrete::Concrete(func(c))
                } else {
                    ParamOrConcrete::Param(c.to_param().map_data_self(|a| function!(*key, a)))
                }
            }
            ParamOrConcrete::Param(p) => {
                ParamOrConcrete::Param(p.map_data_self(|a| function!(*key, a)))
            }
        })
    }
}

pub struct PanicMissingConcrete;
impl PanicMissingConcrete {
    pub fn new_lib<T>() -> SymbolLib<T, Self> {
        SymbolLib {
            functions: HashMap::new(),
            _missing: Self,
        }
    }
}

impl<S: TensorStructure + Clone, C: ToParam + HasStructure<Structure = S>>
    FunctionLibrary<ParamOrConcrete<C, S>> for SymbolLib<C, PanicMissingConcrete>
{
    type Key = Symbol;

    fn apply(
        &self,
        key: &Self::Key,
        tensor: ParamOrConcrete<C, S>,
    ) -> Result<ParamOrConcrete<C, S>, FunctionLibraryError<Symbol>> {
        match tensor {
            ParamOrConcrete::Concrete(c) => {
                if let Some(func) = self.functions.get(key) {
                    Ok(ParamOrConcrete::Concrete(func(c)))
                } else {
                    Err(FunctionLibraryError::NotFound(*key))
                }
            }
            ParamOrConcrete::Param(p) => Ok(ParamOrConcrete::Param(
                p.map_data_self(|a| function!(*key, a)),
            )),
        }
    }
}

impl<S: TensorStructure + Clone, C: ToParam + HasStructure<Structure = S>>
    FunctionLibrary<ParamOrConcrete<C, S>> for SymbolLib<C, Panic>
{
    type Key = Symbol;

    fn apply(
        &self,
        key: &Self::Key,
        tensor: ParamOrConcrete<C, S>,
    ) -> Result<ParamOrConcrete<C, S>, FunctionLibraryError<Symbol>> {
        if let Some(func) = self.functions.get(key) {
            if let ParamOrConcrete::Concrete(c) = tensor {
                Ok(ParamOrConcrete::Concrete(func(c)))
            } else {
                Err(FunctionLibraryError::Other(anyhow!(
                    "Cannot map parametric tensor"
                )))
            }
        } else {
            Err(FunctionLibraryError::NotFound(*key))
        }
    }
}

impl<S: TensorStructure + Clone, C: ToParam + HasStructure<Structure = S>>
    FunctionLibrary<ParamOrConcrete<C, S>> for SymbolLib<ParamOrConcrete<C, S>, Wrap>
{
    type Key = Symbol;

    fn apply(
        &self,
        key: &Self::Key,
        tensor: ParamOrConcrete<C, S>,
    ) -> Result<ParamOrConcrete<C, S>, FunctionLibraryError<Symbol>> {
        Ok(if let Some(func) = self.functions.get(key) {
            func(tensor)
        } else {
            ParamOrConcrete::Param(
                match tensor {
                    ParamOrConcrete::Concrete(c) => c.to_param(),
                    ParamOrConcrete::Param(p) => p,
                }
                .map_data_self(|a| function!(*key, a)),
            )
        })
    }
}

impl<S: TensorStructure + Clone, C> FunctionLibrary<ParamOrConcrete<C, S>>
    for SymbolLib<ParamOrConcrete<C, S>, PanicMissingConcrete>
{
    type Key = Symbol;

    fn apply(
        &self,
        key: &Self::Key,
        tensor: ParamOrConcrete<C, S>,
    ) -> Result<ParamOrConcrete<C, S>, FunctionLibraryError<Symbol>> {
        if let Some(func) = self.functions.get(key) {
            Ok(func(tensor))
        } else if let ParamOrConcrete::Param(p) = tensor {
            Ok(ParamOrConcrete::Param(
                p.map_data_self(|a| function!(*key, a)),
            ))
        } else {
            Err(FunctionLibraryError::NotFound(*key))
        }
    }
}

impl<S: TensorStructure + Clone, C: ToParam + HasStructure<Structure = S>>
    FunctionLibrary<ParamOrConcrete<C, S>> for SymbolLib<ParamOrConcrete<C, S>, Panic>
{
    type Key = Symbol;
    fn apply(
        &self,
        key: &Self::Key,
        tensor: ParamOrConcrete<C, S>,
    ) -> Result<ParamOrConcrete<C, S>, FunctionLibraryError<Symbol>> {
        if let Some(func) = self.functions.get(key) {
            Ok(func(tensor))
        } else {
            Err(FunctionLibraryError::NotFound(*key))
        }
    }
}
