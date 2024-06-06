{...}: {
  perSystem = {
    pkgs,
    config,
    ...
  }: let
    crateName = "spencer";
  in {
    # declare projects
    nci.projects."simple".path = ./.;
    # configure crates
    nci.crates.${crateName} = {};
  };
}
