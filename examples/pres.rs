// use std::ops::Neg;

// use spenso::{
//     ufo::{euclidean_four_vector_sym, gammasym, mink_four_vector_sym},
//     AbstractIndex, Complex, Contract, Dimension, FallibleMul, HistoryStructure, MixedTensor,
//     NamedStructure, NumTensor, Representation, Shadowable, Slot, SymbolicTensor, TensorNetwork,
//     TensorStructure,
// };

// use num::ToPrimitive;

// use symbolica::atom::Symbol;

// fn gamma_net_sym(
//     minkindices: &[i32],
//     vbar: [Complex<f64>; 4],
//     u: [Complex<f64>; 4],
// ) -> TensorNetwork<NumTensor<HistoryStructure<Symbol>>, Complex<f64>> {
//     let mut i = 0;
//     let mut contracting_index = 0.into();
//     let mut result: Vec<NumTensor<HistoryStructure<Symbol>>> =
//         vec![euclidean_four_vector_sym(contracting_index, &vbar).into()];
//     for m in minkindices {
//         let ui = contracting_index;
//         contracting_index += 1.into();
//         let uj = contracting_index;
//         if *m > 0 {
//             let p = [
//                 Complex::<f64>::new(1.0 + 0.01 * i.to_f64().unwrap(), 0.0),
//                 Complex::<f64>::new(1.1 + 0.01 * i.to_f64().unwrap(), 0.0),
//                 Complex::<f64>::new(1.2 + 0.01 * i.to_f64().unwrap(), 0.0),
//                 Complex::<f64>::new(1.3 + 0.01 * i.to_f64().unwrap(), 0.0),
//             ];
//             i += 1;
//             result.push(mink_four_vector_sym(usize::try_from(*m).unwrap().into(), &p).into());
//             result.push(gammasym(usize::try_from(*m).unwrap().into(), (ui, uj)).into());
//         } else {
//             result.push(
//                 gammasym(
//                     AbstractIndex::from(usize::try_from(m.neg()).unwrap() + 10000),
//                     (ui, uj),
//                 )
//                 .into(),
//             );
//         }
//     }
//     result.push(euclidean_four_vector_sym(contracting_index, &u).into());
//     TensorNetwork::from(result)
// }

// fn indices(n: i32, m: i32) -> Vec<i32> {
//     let spacings: [i32; 2] = [n, m];
//     let mut start = 1;
//     let mut ranges = Vec::new();

//     for &spacing in spacings.iter() {
//         ranges.push((start..start + spacing).chain(std::iter::once(-1)));
//         start += spacing;
//     }

//     ranges.into_iter().flatten().collect()
// }

// fn main() {
//     let one = Complex::<f64>::new(1.0, 0.0);
//     let _zero = Complex::<f64>::new(0.0, 0.0);

//     let vbar = [
//         one.mul_fallible(&3.0).unwrap(),
//         one.mul_fallible(&3.1).unwrap(),
//         one.mul_fallible(&3.2).unwrap(),
//         one.mul_fallible(&3.3).unwrap(),
//     ];
//     let u = [
//         one.mul_fallible(&4.0).unwrap(),
//         one.mul_fallible(&4.1).unwrap(),
//         one.mul_fallible(&4.2).unwrap(),
//         one.mul_fallible(&4.3).unwrap(),
//     ];
//     let minkindices = indices(20, 24);

//     let mut netsym = gamma_net_sym(&minkindices, vbar, u);

//     println!("one:{}", netsym.dot());
//     netsym.contract_algo(|s| TensorNetwork::edge_to_min_degree_node_with_depth(&s, 5));
//     println!("two:{}", netsym.dot());
//     netsym.contract_algo(|s| TensorNetwork::edge_to_min_degree_node_with_depth(&s, 10));
//     println!("three:{}", netsym.dot());

//     let mink = Representation::Lorentz(Dimension(4));
//     let mu = Slot::from((AbstractIndex(0), mink));
//     let spin = Representation::SpinFundamental(Dimension(4));
//     let spina = Representation::SpinAntiFundamental(Dimension(4));

//     let i = Slot::from((AbstractIndex(1), spin));
//     let j = Slot::from((AbstractIndex(2), spina));
//     let k = Slot::from((9.into(), spin));

//     let structure: NamedStructure<&str, ()> = NamedStructure::from_iter(vec![mu, i, j], "γ", None);
//     let p_struct: NamedStructure<&str, ()> = NamedStructure::from_iter(vec![mu], "p", None);
//     let t_struct: NamedStructure<&str, ()> = NamedStructure::from_iter(vec![i, j, k], "T", None);

//     let gamma_sym = SymbolicTensor::from_named(&structure.to_shell()).unwrap();
//     let p_sym = SymbolicTensor::from_named(&p_struct.to_shell()).unwrap();
//     let t_sym = SymbolicTensor::from_named(&t_struct.to_shell()).unwrap();

//     let f = gamma_sym
//         .contract(&p_sym)
//         .unwrap()
//         .contract(&t_sym)
//         .unwrap();

//     println!("{}", *f.get_atom());

//     let _a = f.to_network().unwrap();

//     // println!("{}", a.dot());

//     // let γ1: MixedTensor<_> = gamma(1.into(), (1.into(), 2.into())).into();

//     // let γ2: MixedTensor<_> = gamma(10.into(), (2.into(), 3.into())).into();

//     // let _p1 = MixedTensor::param(param_mink_four_vector(1.into(), "p1", None).into());

//     let u: MixedTensor<f64, NamedStructure<_, ()>> = MixedTensor::param(
//         NamedStructure::from_iter(
//             [(1.into(), Representation::SpinFundamental(4.into()))],
//             "u",
//             None,
//         )
//         .to_shell()
//         .expanded_shadow()
//         .unwrap()
//         .into(),
//     );

//     println!("{}", u.try_into_parametric().unwrap());
// }
fn main() {}
