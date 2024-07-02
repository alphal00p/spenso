use std::{fs::File, io::BufReader};

use ahash::AHashMap;
use rand::{distributions::Uniform, Rng, SeedableRng};
use rand_xoshiro::Xoroshiro64Star;
use spenso::{
    Complex, DataIterator, DenseTensor, HasStructure, SetTensorData, SparseTensor, SymbolicTensor,
    TensorStructure,
};
use symbolica::{
    atom::{Atom, AtomView},
    state::State,
};

fn test_tensor<D, S>(structure: S, seed: u64, range: Option<(D, D)>) -> SparseTensor<D, S>
where
    S: TensorStructure,
    D: rand::distributions::uniform::SampleUniform,
    Uniform<D>: Copy,

    rand::distributions::Standard: rand::distributions::Distribution<D>,
{
    let mut rng: Xoroshiro64Star = Xoroshiro64Star::seed_from_u64(seed);

    let mut tensor = SparseTensor::empty(structure);

    let density = tensor.size(); //rng.gen_range(0..tensor.size());

    if let Some((low, high)) = range {
        let multipliable = Uniform::new(low, high);
        for _ in 0..density {
            tensor
                .set_flat(
                    rng.gen_range(0..tensor.size()).into(),
                    rng.sample(multipliable),
                )
                .unwrap();
        }
    } else {
        for _ in 0..density {
            tensor
                .set_flat(rng.gen_range(0..tensor.size()).into(), rng.gen())
                .unwrap();
        }
    }

    tensor
}

fn main() {
    let expr = concat!("-64/729*G^4*ee^6",
    "*(MT*id(aind(bis(4,47),bis(4,135)))+Q(15,aind(lor(4,149)))*γ(aind(lor(4,149),bis(4,47),bis(4,135))))",
    "*(MT*id(aind(bis(4,83),bis(4,46)))+Q(6,aind(lor(4,138)))*γ(aind(lor(4,138),bis(4,83),bis(4,46))))",
    "*(MT*id(aind(bis(4,88),bis(4,82)))+γ(aind(lor(4,140),bis(4,88),bis(4,82)))*Q(7,aind(lor(4,140))))",
    "*(MT*id(aind(bis(4,96),bis(4,142)))+γ(aind(lor(4,141),bis(4,96),bis(4,142)))*Q(8,aind(lor(4,141))))",
    "*(MT*id(aind(bis(4,103),bis(4,95)))+γ(aind(lor(4,143),bis(4,103),bis(4,95)))*Q(9,aind(lor(4,143))))",
    "*(MT*id(aind(bis(4,110),bis(4,102)))+γ(aind(lor(4,144),bis(4,110),bis(4,102)))*Q(10,aind(lor(4,144))))",
    "*(MT*id(aind(bis(4,117),bis(4,109)))+γ(aind(lor(4,145),bis(4,117),bis(4,109)))*Q(11,aind(lor(4,145))))",
    "*(MT*id(aind(bis(4,122),bis(4,116)))+γ(aind(lor(4,146),bis(4,122),bis(4,116)))*Q(12,aind(lor(4,146))))",
    "*(MT*id(aind(bis(4,129),bis(4,123)))+γ(aind(lor(4,147),bis(4,129),bis(4,123)))*Q(13,aind(lor(4,147))))",
    "*(MT*id(aind(bis(4,134),bis(4,130)))+γ(aind(lor(4,148),bis(4,134),bis(4,130)))*Q(14,aind(lor(4,148))))",
    // "*id(coaf(3,46),cof(3,47))*id(coaf(3,82),cof(3,83))*id(coaf(3,95),cof(3,96))*id(coaf(3,109),cof(3,110))*id(coaf(3,116),cof(3,117))*id(coaf(3,130),cof(3,129))",
    "*γ(aind(lor(4,45),bis(4,47),bis(4,46)))*γ(aind(lor(4,81),bis(4,83),bis(4,82)))*γ(aind(lor(4,87),bis(4,88),bis(4,142)))*γ(aind(lor(4,94),bis(4,96),bis(4,95)))",
    "*γ(aind(lor(4,101),bis(4,103),bis(4,102)))*γ(aind(lor(4,108),bis(4,110),bis(4,109)))*γ(aind(lor(4,115),bis(4,117),bis(4,116)))*γ(aind(lor(4,121),bis(4,122),bis(4,123)))",
    "*γ(aind(lor(4,128),bis(4,129),bis(4,130)))*γ(aind(lor(4,133),bis(4,134),bis(4,135)))*Metric(aind(lor(4,121),lor(4,87)))*Metric(aind(lor(4,133),lor(4,101)))",
    // "*T(coad(8,87),cof(3,88),coaf(3,46))*T(coad(8,101),cof(3,103),coaf(3,102))*T(coad(8,121),cof(3,122),coaf(3,123))*T(coad(8,133),cof(3,134),coaf(3,135))",
    "*ϵ(0,aind(lor(4,45)))*ϵ(1,aind(lor(4,81)))*ϵbar(2,aind(lor(4,94)))*ϵbar(3,aind(lor(4,108)))*ϵbar(4,aind(lor(4,115)))*ϵbar(5,aind(lor(4,128)))"
);

    let params = ["MT", "G", "ee"].map(|p| Atom::parse(p).unwrap());

    let atom = Atom::parse(expr).unwrap();

    let sym_tensor: SymbolicTensor = atom.try_into().unwrap();

    let mut const_map = AHashMap::new();
    const_map.insert(params[0].as_view(), 0.2);
    const_map.insert(params[1].as_view(), 0.32243234);
    const_map.insert(params[2].as_view(), 0.932);

    let mut network = sym_tensor.to_network().unwrap();

    // for p in &network.params {
    //     println!("Param {}", p);
    // }

    println!("Network dot: {}", network.dot());

    let file = File::open("./examples/data.json").unwrap();
    let reader = BufReader::new(file);

    let string_map: AHashMap<String, Complex<f64>> = serde_json::from_reader(reader).unwrap();

    let mut atom_map: AHashMap<Atom, Complex<f64>> = string_map
        .into_iter()
        .map(|(k, v)| (Atom::parse(&k).unwrap(), v))
        .collect();

    let i = Atom::new_var(State::I);
    atom_map.insert(i, Complex::i());

    let const_map: AHashMap<AtomView<'_>, symbolica::domains::float::Complex<f64>> = atom_map
        .iter()
        .map(|(k, &v)| (k.as_view(), v.into()))
        .collect();

    network.evaluate_complex(|i| i.into(), &const_map);
    network.contract();

    println!(
        "{}",
        network
            .result_tensor()
            .unwrap()
            .try_into_concrete()
            .unwrap()
            .try_into_complex()
            .unwrap()
            .try_into_dense()
            .unwrap()
    );
}
