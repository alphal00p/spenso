use std::collections::HashSet;

use ahash::{AHashMap, AHashSet, HashMap};
use spenso::{
    complex::Complex,
    data::HasTensorData,
    network::TensorNetwork,
    parametric::{FlatCoefficent, MixedTensor},
    structure::{
        abstract_index::AbstractIndex,
        representation::{Lorentz, PhysReps, RepName},
        ContractionCountStructure, ToSymbolic,
    },
    symbolica_utils::SerializableAtom,
    ufo::{euclidean_four_vector, gamma},
    upgrading_arithmetic::FallibleMul,
};

use rand::{distributions::Uniform, Rng, SeedableRng};
use rand_xoshiro::Xoroshiro64Star;
use symbolica::{
    atom::{Atom, AtomView},
    state::State,
};
fn gamma_net_param(
    minkindices: &[i32],
    vbar: [Complex<f64>; 4],
    u: [Complex<f64>; 4],
) -> TensorNetwork<MixedTensor<f64, ContractionCountStructure>, SerializableAtom> {
    let mut i: i32 = 0;
    let mut contracting_index = 0.into();
    let mut result: Vec<MixedTensor<f64, ContractionCountStructure>> =
        vec![euclidean_four_vector(contracting_index, vbar).into()];
    let lor_fouru = PhysReps::new_rep(&Lorentz {}.into(), 4);
    let lor_fourd = lor_fouru.dual();
    let p = State::get_symbol("p");
    let mut seen = HashSet::new();
    for m in minkindices {
        let ui = contracting_index;
        contracting_index += 1.into();
        let uj = contracting_index;
        if *m > 0 {
            let ps = ContractionCountStructure::from_iter([
                lor_fourd.new_slot(usize::try_from(*m).unwrap())
            ]);
            // let ps: ContractionCountStructure = vec![
            //     lor_fourd.new_slot() Slot::from((
            //     usize::try_from(*m).unwrap().into(),
            //     Lorentz(4.into()),
            // ))]
            // .into_iter()
            // .collect();
            i += 1;
            let id = Atom::new_num(i);

            result.push(MixedTensor::param(
                ps.to_dense_labeled(|_, index| FlatCoefficent {
                    name: Some(p),
                    index,
                    args: Some([id.clone()]),
                })
                .unwrap()
                .into(),
            ));

            result.push(gamma(AbstractIndex::from(-*m), [ui, uj]).into());
        } else {
            let mu = if seen.insert(m) {
                AbstractIndex::from(-*m + 10000)
            } else {
                AbstractIndex::from(*m - 10000)
            };
            result.push(gamma(mu, [ui, uj]).into());
        }
    }
    result.push(euclidean_four_vector(contracting_index, u).into());
    TensorNetwork::from(result)
}

fn const_map_gen<'a, 'b>(
    params: &'a AHashSet<Atom>,
    const_map: &mut HashMap<AtomView<'b>, symbolica::domains::float::Complex<f64>>,
) where
    'a: 'b,
{
    let mut rng: Xoroshiro64Star = Xoroshiro64Star::from_entropy();
    let multipliable = Uniform::new(1., 10.);

    for p in params.iter() {
        let p = p.as_view();
        const_map.insert(
            p,
            Complex::<f64>::new(rng.sample(multipliable), rng.sample(multipliable)).into(),
        );
    }
}

fn main() {
    let one = Complex::<f64>::new(1.0, 0.0);

    let notnorm: u8 = 0b10000000;
    let mut f: u8 = 3;
    f |= notnorm;
    println!("{:?}", f);
    f |= notnorm;
    println!("{:?}", f);

    let vbar = [
        one.mul_fallible(&3.0).unwrap(),
        one.mul_fallible(&3.1).unwrap(),
        one.mul_fallible(&3.2).unwrap(),
        one.mul_fallible(&3.3).unwrap(),
    ];
    let u = [
        one.mul_fallible(&4.0).unwrap(),
        one.mul_fallible(&4.1).unwrap(),
        one.mul_fallible(&4.2).unwrap(),
        one.mul_fallible(&4.3).unwrap(),
    ];
    let spacings: [i32; 2] = [2, 4];
    let mut start = 1;
    let mut ranges = Vec::new();

    for &spacing in spacings.iter() {
        ranges.push((start..start + spacing).chain(std::iter::once(-1)));
        start += spacing;
    }

    let vec: Vec<i32> = ranges.into_iter().flatten().collect();

    let mut net = gamma_net_param(&vec, vbar, u);
    let params = net.generate_params();
    let mut const_map = AHashMap::new();
    const_map_gen(&params, &mut const_map);

    let i = Atom::new_var(State::I);
    const_map.insert(i.as_view(), Complex::<f64>::new(0., 1.).into());

    // net.contract_algo(|tn| tn.edge_to_min_degree_node_with_depth(2));

    // for (i, n) in &net.graph.nodes {
    //     match n {
    //         MixedTensor::Symbolic(s) => {
    //             for (_, a) in s.try_as_dense().unwrap().iter_flat() {
    //                 println!("{}", a);
    //             }
    //         }
    //         _ => {}
    //     }
    // }

    // for p in const_map.keys() {
    //     if let AtomView::Fun(f) = p {
    //         println!(
    //             "Map {}, with id {:?},{:?}",
    //             State::get_name(f.get_symbol()),
    //             f.get_symbol(),
    //             f
    //         );
    //     }
    // }
    net.contract();
    let now = std::time::Instant::now();
    net.evaluate_complex(|r| r.into(), &const_map);
    println!("Time: {:?}", now.elapsed());
    println!(
        "finished {:?}",
        net.result_tensor()
            .unwrap()
            .try_into_concrete()
            .unwrap()
            .try_as_complex()
            .unwrap()
            .data()[0]
    );
}
