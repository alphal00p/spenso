use std::ops::Neg;

use spenso::{
    ufo::{euclidean_four_vector, gamma},
    AbstractIndex, ContractionCountStructure, FallibleMul, MixedTensor, Representation, Slot,
    TensorNetwork, ToSymbolic,
};
use spenso::{Complex, FlatCoefficent};

use ahash::{AHashMap, AHashSet, HashMap};
use criterion::{criterion_group, criterion_main, Criterion};

use rand::{distributions::Uniform, Rng, SeedableRng};
use rand_xoshiro::Xoroshiro64Star;
use symbolica::{
    atom::{Atom, AtomView},
    state::State,
};

fn indices(n: i32, m: i32) -> Vec<i32> {
    let spacings: [i32; 2] = [n, m];
    let mut start = 1;
    let mut ranges = Vec::new();

    for &spacing in spacings.iter() {
        ranges.push((start..start + spacing).chain(std::iter::once(-1)));
        start += spacing;
    }

    ranges.into_iter().flatten().collect()
}

fn gamma_net_param(
    minkindices: &[i32],
    vbar: [Complex<f64>; 4],
    u: [Complex<f64>; 4],
) -> TensorNetwork<MixedTensor<f64, ContractionCountStructure>, Atom> {
    let mut i: i32 = 0;
    let mut contracting_index = 0.into();
    let mut result: Vec<MixedTensor<f64, ContractionCountStructure>> =
        vec![euclidean_four_vector(contracting_index, &vbar).into()];
    let p = State::get_symbol(&"p");
    for m in minkindices {
        let ui = contracting_index;
        contracting_index += 1.into();
        let uj = contracting_index;
        if *m > 0 {
            let ps: ContractionCountStructure = vec![Slot::from((
                usize::try_from(*m).unwrap().into(),
                Representation::Lorentz(4.into()),
            ))]
            .into_iter()
            .collect();
            i += 1;
            let id = Atom::new_num(i);

            result.push(MixedTensor::param(
                ps.to_dense_labeled(|_, index| FlatCoefficent {
                    name: Some(p),
                    index,
                    args: Some([id.clone()]),
                })
                .into(),
            ));

            result.push(gamma(usize::try_from(*m).unwrap().into(), (ui, uj)).into());
        } else {
            result.push(
                gamma(
                    AbstractIndex::from(usize::try_from(m.neg()).unwrap() + 10000),
                    (ui, uj),
                )
                .into(),
            );
        }
    }
    result.push(euclidean_four_vector(contracting_index, &u).into());
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

    for (_i, p) in params.iter().enumerate() {
        let p = p.as_view();
        const_map.insert(
            p,
            Complex::<f64>::new(rng.sample(multipliable), rng.sample(multipliable)).into(),
        );
    }
}
fn criterion_benchmark(c: &mut Criterion) {
    let one = Complex::<f64>::new(1.0, 0.0);

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
    let minkindices = indices(20, 24);

    let mut net = gamma_net_param(&minkindices, vbar, u);

    let params = net.generate_params();

    println!("{:?}", params.len());
    net.contract_algo(|tn| tn.edge_to_min_degree_node_with_depth(2));
    let mut const_map = AHashMap::new();

    let i = Atom::new_var(State::I);
    const_map.insert(i.as_view(), Complex::<f64>::new(0., 1.).into());

    let mut group = c.benchmark_group("evaluate_net");

    group.bench_function("Evaluate_net", |b| {
        b.iter_batched(
            || net.clone(),
            |mut net| {
                const_map_gen(&params, &mut const_map);

                net.evaluate_complex(|r| r.into(), &const_map);

                net.contract();
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
