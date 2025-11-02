use std::{fmt::Display, marker::PhantomData};

use crate::network::library::{DummyKey, FunctionLibrary, FunctionLibraryError};

pub struct PanicingLibrary<K = DummyKey> {
    key: PhantomData<K>,
}

impl<K> PanicingLibrary<K> {
    pub fn new() -> Self {
        Self { key: PhantomData }
    }
}

impl<T, K: Display + Clone> FunctionLibrary<T> for PanicingLibrary<K> {
    type Key = K;

    fn apply(&self, key: &Self::Key, tensor: T) -> Result<T, FunctionLibraryError<K>> {
        Err(FunctionLibraryError::NotFound(key.clone()))
    }
}
