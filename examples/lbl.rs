use std::{fs::File, io::BufReader};

use ahash::AHashMap;

use spenso::{
    network, Complex, Levels, MixedTensor, SmartShadowStructure, SymbolicTensor, TensorNetwork,
};
use symbolica::{
    atom::{Atom, AtomView, Symbol},
    domains::{
        float::{NumericalFloatComparison, NumericalFloatLike},
        rational::Rational,
    },
    evaluate::FunctionMap,
    state::State,
};

use symbolica::domains::float::Complex as SymComplex;

fn main() {
    let expr = concat!("256/81*(MT*id(aind(bis(4,185)*bis(4,218)))+Q(4,aind(lor(4,234)))",
    "*γ(aind(lor(4,234)*bis(4,185)*bis(4,218))))*(MT*id(aind(bis(4,186)*bis(4,227)))+Q(7,aind(lor(4,237)))",
    "*γ(aind(lor(4,237)*bis(4,186)*bis(4,227))))*(MT*id(aind(bis(4,217)*bis(4,223)))+Q(5,aind(lor(4,235)))",
    "*γ(aind(lor(4,235)*bis(4,217)*bis(4,223))))*(MT*id(aind(bis(4,222)*bis(4,228)))+Q(6,aind(lor(4,236)))",
    "*γ(aind(lor(4,236)*bis(4,222)*bis(4,228))))*sqrt(pi)^4*sqrt(aEW)^4",
    "*id(aind(coaf(3,185),cof(3,186)))*id(aind(coaf(3,217),cof(3,218)))*id(aind(coaf(3,222),cof(3,223)))*id(aind(coaf(3,227),cof(3,228)))",
    "*γ(aind(lor(4,187)*bis(4,185)*bis(4,186)))*γ(aind(lor(4,219)*bis(4,217)*bis(4,218)))",
    "*γ(aind(lor(4,224)*bis(4,222)*bis(4,223)))*γ(aind(lor(4,229)*bis(4,227)*bis(4,228)))",
    "*ϵ(0,aind(lor(4,187)))*ϵbar(1,aind(lor(4,219)))*ϵbar(2,aind(lor(4,224)))*ϵbar(3,aind(lor(4,229)))",
);

    let atom = Atom::parse(expr).unwrap();

    let sym_tensor: SymbolicTensor = atom.try_into().unwrap();

    let network = sym_tensor.to_network().unwrap();

    for (n, t) in &network.graph.nodes {
        println!("{}", t)
    }

    println!("Network dot: {}", network.dot());
}
