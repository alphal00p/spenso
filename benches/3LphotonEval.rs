use std::{fs::File, io::BufReader};

use ahash::AHashMap;
use criterion::{criterion_group, criterion_main, Criterion};
use spenso::{
    complex::Complex, parametric::atomcore::TensorAtomOps, structure::representation::PhysReps,
    symbolic::SymbolicTensor,
};
use symbolica::{
    atom::{Atom, Symbol},
    domains::rational::Rational,
    evaluate::{CompileOptions, FunctionMap, InlineASM, OptimizationSettings},
};

fn criterion_benchmark(c: &mut Criterion) {
    let expr = concat!("-64/729*G^4*ee^6",
    "*(MT*id(bis(4,47),bis(4,135))+Q(15,mink(4,149))*γ(mink(4,149),bis(4,47),bis(4,135)))",
    "*(MT*id(bis(4,83),bis(4,46))+Q(6,mink(4,138))*γ(mink(4,138),bis(4,83),bis(4,46)))",
    "*(MT*id(bis(4,88),bis(4,82))+γ(mink(4,140),bis(4,88),bis(4,82))*Q(7,mink(4,140)))",
    "*(MT*id(bis(4,96),bis(4,142))+γ(mink(4,141),bis(4,96),bis(4,142))*Q(8,mink(4,141)))",
    "*(MT*id(bis(4,103),bis(4,95))+γ(mink(4,143),bis(4,103),bis(4,95))*Q(9,mink(4,143)))",
    "*(MT*id(bis(4,110),bis(4,102))+γ(mink(4,144),bis(4,110),bis(4,102))*Q(10,mink(4,144)))",
    "*(MT*id(bis(4,117),bis(4,109))+γ(mink(4,145),bis(4,117),bis(4,109))*Q(11,mink(4,145)))",
    "*(MT*id(bis(4,122),bis(4,116))+γ(mink(4,146),bis(4,122),bis(4,116))*Q(12,mink(4,146)))",
    "*(MT*id(bis(4,129),bis(4,123))+γ(mink(4,147),bis(4,129),bis(4,123))*Q(13,mink(4,147)))",
    "*(MT*id(bis(4,134),bis(4,130))+γ(mink(4,148),bis(4,134),bis(4,130))*Q(14,mink(4,148)))",
    "*γ(mink(4,45),bis(4,47),bis(4,46))*γ(mink(4,81),bis(4,83),bis(4,82))*γ(mink(4,87),bis(4,88),bis(4,142))*γ(mink(4,94),bis(4,96),bis(4,95))",
    "*γ(mink(4,101),bis(4,103),bis(4,102))*γ(mink(4,108),bis(4,110),bis(4,109))*γ(mink(4,115),bis(4,117),bis(4,116))*γ(mink(4,121),bis(4,122),bis(4,123))",
    "*γ(mink(4,128),bis(4,129),bis(4,130))*γ(mink(4,133),bis(4,134),bis(4,135))*Metric(mink(4,121),mink(4,87))*Metric(mink(4,133),mink(4,101))",
    "*ϵ(0,mink(4,45))*ϵ(1,mink(4,81))*ϵbar(2,mink(4,94))*ϵbar(3,mink(4,108))*ϵbar(4,mink(4,115))*ϵbar(5,mink(4,128))"
);

    let atom = Atom::parse(expr).unwrap();

    let sym_tensor: SymbolicTensor = atom.try_into().unwrap();

    let mut network = sym_tensor.to_network::<PhysReps>().unwrap();

    let mut group = c.benchmark_group("nested_evaluate_net");

    let file = File::open("./examples/data.json").unwrap();
    let reader = BufReader::new(file);

    let data_string_map: AHashMap<String, Complex<f64>> = serde_json::from_reader(reader).unwrap();

    let file = File::open("./examples/const.json").unwrap();
    let reader = BufReader::new(file);

    let const_string_map: AHashMap<String, Complex<f64>> = serde_json::from_reader(reader).unwrap();

    let data_atom_map: (Vec<Atom>, Vec<Complex<f64>>) = data_string_map
        .into_iter()
        .map(|(k, v)| (Atom::parse(&k).unwrap(), v))
        .unzip();

    let const_atom_map: AHashMap<Symbol, Complex<f64>> = const_string_map
        .into_iter()
        .map(|(k, v)| (Symbol::new(k), v))
        .collect();

    let fn_map: FunctionMap<Rational> = FunctionMap::new();

    let mut params = data_atom_map.0.clone();
    let mut paramdata = data_atom_map.1.clone();

    for (s, c) in const_atom_map.iter() {
        params.push(Atom::new_var(*s));
        paramdata.push(*c);
    }
    params.push(Atom::new_var(Atom::I));
    paramdata.push(Complex::i());

    network.contract().unwrap();
    let res = network.result_tensor_smart().unwrap();

    let eval = res
        .try_into_parametric()
        .unwrap()
        .evaluator(&fn_map, &params, OptimizationSettings::default())
        .unwrap();

    let mut evalt = eval.clone().map_coeff(&|a| Complex::new_re(a.to_f64()));

    let mut compiledt = eval
        .export_cpp("oneloop", "oneloop", true, InlineASM::X64)
        .unwrap()
        .compile("oneloop.out", CompileOptions::default())
        .unwrap()
        .load()
        .unwrap();

    compiledt.evaluate(&paramdata);
    println!("hi");
    evalt.evaluate(&paramdata);

    group.bench_function("3LPhysical linearized", |b| {
        b.iter_batched(
            || evalt.clone(),
            |mut evalt| {
                evalt.evaluate(&paramdata);
            },
            criterion::BatchSize::SmallInput,
        )
    });
    group.bench_function("3LPhysical compiled", |b| {
        b.iter_batched(
            || compiledt.clone(),
            |mut compiledt| {
                compiledt.evaluate(&paramdata);
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
