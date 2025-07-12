use super::TensorNetworkError;

#[test]
fn display() {
    let a = TensorNetworkError::<i8>::Infallible;

    println!("{a}")
}
