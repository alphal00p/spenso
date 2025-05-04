use bincode::{Decode, Encode};

use super::*;
use linnet::half_edge::NodeIndex;
use ref_ops::{RefAdd, RefMul, RefNeg};
use serde::{Deserialize, Serialize};

use tensor_library::{Library, LibraryError};

use crate::algebraic_traits::{One, Zero};
use crate::arithmetic::ScalarMul;
use crate::contraction::Contract;
use crate::network::tensor_library::LibraryTensor;
// use crate::shadowing::Concretize;
use crate::structure::representation::LibrarySlot;
use crate::structure::{StructureError, TensorShell};
use std::borrow::Cow;
use std::fmt::Display;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg};
use store::{NetworkStore, TensorScalarStore, TensorScalarStoreMapping};
use thiserror::Error;
// use log::trace;

use symbolica::atom::{representation::FunView, AddView, Atom, AtomView, MulView, PowView};

use crate::{
    contraction::ContractionError,
    structure::{CastStructure, HasStructure, ScalarTensor, TensorStructure},
};

use crate::{
    parametric::ParamTensor,
    shadowing::Concretize,
    structure::representation::LibraryRep,
    structure::slot::IsAbstractSlot,
    structure::HasName,
    symbolica_utils::{IntoArgs, IntoSymbol},
};

impl<
        'a,
        Sc,
        T: HasStructure + TensorStructure,
        K: Clone + Display + Debug,
        Str: TensorScalarStore<Tensor = T, Scalar = Sc> + Clone,
    > Network<Str, K>
where
    Sc: for<'r> TryFrom<AtomView<'r>> + Clone,
    TensorNetworkError<K>: for<'r> From<<Sc as TryFrom<AtomView<'r>>>::Error>,
{
    pub fn try_from_view<S, Lib: Library<S, Key = K, Value: LibraryTensor<WithIndices = T>>>(
        value: AtomView<'a>,
        library: &Lib,
    ) -> Result<Self, TensorNetworkError<K>>
    where
        S: TryFrom<FunView<'a>> + TensorStructure + Clone + HasName,
        TensorShell<S>: Concretize<T>,
    {
        match value {
            AtomView::Mul(m) => Self::try_from_mul(m, library),
            AtomView::Fun(f) => Self::try_from_fun(f, library),
            AtomView::Add(a) => Self::try_from_add(a, library),
            AtomView::Pow(p) => Self::try_from_pow(p, library),
            a => Ok(Network::scalar(a.try_into()?)),
        }
    }

    fn try_from_mul<S, Lib: Library<S, Key = K, Value: LibraryTensor<WithIndices = T>>>(
        value: MulView<'a>,
        library: &Lib,
    ) -> Result<Self, TensorNetworkError<K>>
    where
        S: TryFrom<FunView<'a>> + TensorStructure + Clone + HasName,
        TensorShell<S>: Concretize<T>,
    {
        let mut iter = value.iter();

        let first = Self::try_from_view(iter.next().unwrap(), library)?;

        let rest: Result<Vec<_>, _> = iter.map(|a| Self::try_from_view(a, library)).collect();

        Ok(first.n_mul(rest?))
    }

    fn try_from_fun<S, Lib: Library<S, Key = K, Value: LibraryTensor<WithIndices = T>>>(
        value: FunView<'a>,
        library: &Lib,
    ) -> Result<Self, TensorNetworkError<K>>
    where
        S: TryFrom<FunView<'a>> + TensorStructure + Clone + HasName,
        TensorShell<S>: Concretize<T>,
    {
        let s: Result<S, _> = value.try_into();

        if let Ok(s) = s {
            let s = s.clone();
            match library.key_for_structure(s) {
                Ok(key) => {
                    let t = library.get(&key).unwrap();
                    Ok(Self::library_tensor(t.as_ref(), key))
                }
                Err(shell) => Ok(Self::local_tensor(shell.to_shell().concretize())),
            }
        } else {
            Ok(Self::scalar(
                value.as_view().try_into().map_err(Into::into)?,
            ))
        }
    }

    fn try_from_pow<S, Lib: Library<S, Key = K, Value: LibraryTensor<WithIndices = T>>>(
        value: PowView<'a>,
        library: &Lib,
    ) -> std::result::Result<Self, TensorNetworkError<K>>
    where
        S: TryFrom<FunView<'a>> + TensorStructure + Clone + HasName,
        TensorShell<S>: Concretize<T>,
    {
        let (base, exp) = value.get_base_exp();

        if let Ok(n) = i64::try_from(exp) {
            if n < 0 {
                return Ok(Self::scalar(value.as_view().try_into()?));
            }
            if n == 0 {
                let one = Atom::new_num(1);
                return Ok(Self::scalar(one.as_view().try_into()?));
            } else if n == 1 {
                return Self::try_from_view(base, library);
            }
            let net = Self::try_from_view(base, library)?;
            let cloned_net = net.clone();

            Ok(net.n_mul((0..n).map(|_| cloned_net.clone())))
        } else {
            Ok(Self::scalar(value.as_view().try_into()?))
        }
    }

    fn try_from_add<S, Lib: Library<S, Key = K, Value: LibraryTensor<WithIndices = T>>>(
        value: AddView<'a>,
        library: &Lib,
    ) -> Result<Self, TensorNetworkError<K>>
    where
        S: TryFrom<FunView<'a>> + TensorStructure + Clone + HasName,
        TensorShell<S>: Concretize<T>,
    {
        let mut iter = value.iter();

        let first = Self::try_from_view(iter.next().unwrap(), library)?;

        let rest: Result<Vec<_>, _> = iter.map(|a| Self::try_from_view(a, library)).collect();

        Ok(first.n_add(rest?))
    }
}

#[cfg(test)]
pub mod test {
    use crate::{data::DenseTensor, symbolic::SymbolicTensor};

    use super::*;
    use symbolica::parse;
    use tensor_library::{DummyKey, DummyLibrary, DummyLibraryTensor};

    #[test]
    fn parse() {
        let expr = parse!("1").unwrap();
        let lib = DummyLibrary::<_>::new();
        let net =
            Network::<NetworkStore<SymbolicTensor, Atom>, _>::try_from_view(expr.as_view(), &lib);
    }
}
