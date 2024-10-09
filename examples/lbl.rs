#[cfg(feature = "shadowing")]
use spenso::symbolic::SymbolicTensor;
#[cfg(feature = "shadowing")]
use symbolica::atom::Atom;

fn main() {
    #[cfg(feature = "shadowing")]
    let expr = concat!("256/81*(MT*id(aind(bis(4,186),bis(4,227)))+Q(7,aind(loru(4,237)))",
    "*γ(aind(lord(4,237),bis(4,186),bis(4,227))))*(MT*id(aind(bis(4,218),bis(4,185)))+Q(4,aind(loru(4,234)))",
    "*γ(aind(lord(4,234),bis(4,218),bis(4,185))))*(MT*id(aind(bis(4,223),bis(4,217)))+Q(5,aind(loru(4,235)))",
    "*γ(aind(lord(4,235),bis(4,223),bis(4,217))))*(MT*id(aind(bis(4,228),bis(4,222)))+Q(6,aind(loru(4,236)))",
    "*γ(aind(lord(4,236),bis(4,228),bis(4,222))))*sqrt(pi)^4*sqrt(aEW)^4",
    // "*id(aind(coaf(3,185),cof(3,186)))*id(aind(coaf(3,217),cof(3,218)))*id(aind(coaf(3,222),cof(3,223)))*id(aind(coaf(3,227),cof(3,228)))",
    "*γ(aind(lord(4,187),bis(4,186),bis(4,185)))",
    "*γ(aind(lord(4,219),bis(4,218),bis(4,217)))",
    "*γ(aind(lord(4,224),bis(4,223),bis(4,222)))",
    "*γ(aind(lord(4,229),bis(4,228),bis(4,227)))",
    "*ϵ(0,aind(loru(4,187)))*ϵbar(1,aind(loru(4,219)))*ϵbar(2,aind(loru(4,224)))*ϵbar(3,aind(loru(4,229)))",
);
    #[cfg(feature = "shadowing")]
    let atom = Atom::parse(expr).unwrap();
    #[cfg(feature = "shadowing")]
    let sym_tensor: SymbolicTensor = atom.try_into().unwrap();
    #[cfg(feature = "shadowing")]
    let network = sym_tensor.to_network().unwrap();
    #[cfg(feature = "shadowing")]
    for (_, t) in &network.graph.nodes {
        println!("{}", t)
    }
    #[cfg(feature = "shadowing")]
    println!("Network dot: {}", network.dot());
}
