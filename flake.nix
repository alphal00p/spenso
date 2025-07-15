{
  description = "Build a cargo workspace project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

    crane = {
      url = "github:ipetkov/crane";
    };

    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.rust-analyzer-src.follows = "";
    };

    flake-utils.url = "github:numtide/flake-utils";

    advisory-db = {
      url = "github:rustsec/advisory-db";
      flake = false;
    };
  };

  outputs = {
    self,
    nixpkgs,
    crane,
    fenix,
    flake-utils,
    advisory-db,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = nixpkgs.legacyPackages.${system};

      inherit (pkgs) lib;

      craneLib =
        (crane.mkLib nixpkgs.legacyPackages.${system}).overrideToolchain
        fenix.packages.${system}.stable.toolchain;

      src = lib.cleanSourceWith {
        src = craneLib.path ./.; # The original, unfiltered source
        filter = path: type: (craneLib.filterCargoSources path type) || (builtins.match ".*snap$" path != null);
      };

      # Common arguments can be set here to avoid repeating them later
      commonArgs = {
        inherit src;
        strictDeps = true;

        SYMBOLICA_LICENSE =
          if builtins.getEnv "SYMBOLICA_LICENSE" != ""
          then builtins.getEnv "SYMBOLICA_LICENSE"
          else "dummy";
        RUST_BACKTRACE = 1;
        nativeBuildInputs = [
          pkgs.git
          pkgs.gmp.dev
          pkgs.gnum4
          pkgs.clang
          pkgs.mpfr.dev
          pkgs.cargo-insta
        ];
        buildInputs =
          [
            # Add additional build inputs here
          ]
          ++ lib.optionals pkgs.stdenv.isDarwin [
            # Additional darwin specific inputs can be set here
            pkgs.libiconv
          ];

        # Additional environment variables can be set directly
        # MY_CUSTOM_VAR = "some value";
      };

      craneLibLLvmTools =
        craneLib.overrideToolchain
        (fenix.packages.${system}.stable.withComponents [
          "cargo"
          "llvm-tools"
          "rustc"
        ]);

      # Build *just* the cargo dependencies, so we can reuse
      # all of that work (e.g. via cachix) when running in CI
      cargoArtifacts = craneLib.buildDepsOnly (commonArgs
        // {
          pname = "spenso-workspace-deps";
          version = "0.1.0";
        });

      # Build the entire workspace
      workspace = craneLib.buildPackage (commonArgs
        // {
          inherit cargoArtifacts;
          pname = "spenso-workspace";
          version = "0.4.1";
          doCheck = false;
          cargoExtraArgs = "--workspace";
        });

      # Helper function to create checks for a specific crate using manifest path
      mkChecksForCrate = crateName: let
        manifestPath = "${crateName}/Cargo.toml";
        crateInfo = craneLib.crateNameFromCargoToml {cargoToml = ./${manifestPath};};
      in {
        "${crateInfo.pname}-clippy" = craneLib.cargoClippy (commonArgs
          // {
            inherit cargoArtifacts;
            inherit (crateInfo) pname version;
            cargoClippyExtraArgs = "--manifest-path ${manifestPath} -- --deny warnings";
          });

        "${crateInfo.pname}-doc" = craneLib.cargoDoc (commonArgs
          // {
            inherit cargoArtifacts;
            inherit (crateInfo) pname version;
            cargoExtraArgs = "--manifest-path ${manifestPath}";
          });

        "${crateInfo.pname}-nextest" = craneLib.cargoNextest (commonArgs
          // {
            inherit cargoArtifacts;
            inherit (crateInfo) pname version;
            nativeBuildInputs =
              commonArgs.nativeBuildInputs
              ++ [
                pkgs.cargo-insta
              ];
            partitions = 1;
            partitionType = "count";
            cargoNextestExtraArgs = "--manifest-path ${manifestPath}";
          });

        "${crateInfo.pname}-tarpaulin" = craneLib.cargoTarpaulin (commonArgs
          // {
            inherit cargoArtifacts;
            inherit (crateInfo) pname version;
            cargoTarpaulinExtraArgs = "--manifest-path ${manifestPath} --skip-clean --out xml --output-dir $out";
          });
      };

      # Helper function to create checks with feature flags
      mkChecksWithFeatures = features: featuresName: {
        "workspace-clippy-${featuresName}" = craneLib.cargoClippy (commonArgs
          // {
            inherit cargoArtifacts;
            pname = "spenso-workspace";
            version = "0.4.1";
            cargoClippyExtraArgs = "--workspace ${features} -- --deny warnings";
          });

        "workspace-nextest-${featuresName}" = craneLib.cargoNextest (commonArgs
          // {
            inherit cargoArtifacts;
            pname = "spenso-workspace";
            version = "0.4.1";
            nativeBuildInputs =
              commonArgs.nativeBuildInputs
              ++ [
                pkgs.cargo-insta
              ];
            partitions = 1;
            partitionType = "count";
            cargoNextestExtraArgs = "--workspace ${features}";
          });

        "workspace-tarpaulin-${featuresName}" = craneLib.cargoTarpaulin (commonArgs
          // {
            inherit cargoArtifacts;
            pname = "spenso-workspace";
            version = "0.4.1";
            cargoTarpaulinExtraArgs = "--workspace ${features} --skip-clean --out xml --output-dir $out";
          });
      };

      # Helper function to create per-crate checks with features
      mkCrateChecksWithFeatures = crateName: features: featuresName: let
        manifestPath = "${crateName}/Cargo.toml";
        crateInfo = craneLib.crateNameFromCargoToml {cargoToml = ./${manifestPath};};
      in {
        "${crateInfo.pname}-clippy-${featuresName}" = craneLib.cargoClippy (commonArgs
          // {
            inherit cargoArtifacts;
            inherit (crateInfo) pname version;
            cargoClippyExtraArgs = "--manifest-path ${manifestPath} ${features} -- --deny warnings";
          });

        "${crateInfo.pname}-nextest-${featuresName}" = craneLib.cargoNextest (commonArgs
          // {
            inherit cargoArtifacts;
            inherit (crateInfo) pname version;
            nativeBuildInputs =
              commonArgs.nativeBuildInputs
              ++ [
                pkgs.cargo-insta
              ];
            partitions = 1;
            partitionType = "count";
            cargoNextestExtraArgs = "--manifest-path ${manifestPath} ${features}";
          });
      };

      # Create checks for each crate
      spensoChecks = mkChecksForCrate "spenso";
      spensoMacrosChecks = mkChecksForCrate "spenso-macros";
      spensoHepLibChecks = mkChecksForCrate "spenso-hep-lib";
      idensoChecks = mkChecksForCrate "idenso";

      # Create feature flag checks
      defaultFeatureChecks = mkChecksWithFeatures "" "default";
      shadowingFeatureChecks = mkChecksWithFeatures "--features shadowing" "shadowing";
      allFeatureChecks = mkChecksWithFeatures "--all-features" "all-features";
      noDefaultFeatureChecks = mkChecksWithFeatures "--no-default-features" "no-default";

      # Create per-crate feature checks for important combinations
      idensoFeatureChecks = mkCrateChecksWithFeatures "idenso" "--features bincode" "bincode";
      spensoShadowingChecks = mkCrateChecksWithFeatures "spenso" "--features shadowing" "shadowing";

      # Workspace-wide checks
      workspaceChecks = {
        # Build the workspace as part of `nix flake check` for convenience
        inherit workspace;

        # Workspace-wide clippy
        workspace-clippy = craneLib.cargoClippy (commonArgs
          // {
            inherit cargoArtifacts;
            pname = "spenso-workspace";
            version = "0.4.1";
            cargoClippyExtraArgs = "--workspace -- --deny warnings";
          });

        # Workspace-wide doc
        workspace-doc = craneLib.cargoDoc (commonArgs
          // {
            inherit cargoArtifacts;
            pname = "spenso-workspace";
            version = "0.4.1";
            cargoExtraArgs = "--workspace";
          });

        # Check formatting (workspace-wide)
        workspace-fmt = craneLib.cargoFmt {
          inherit src;
          pname = "spenso-workspace";
          version = "0.4.1";
        };

        # Audit dependencies (workspace-wide)
        workspace-audit = craneLib.cargoAudit {
          inherit src advisory-db;
          pname = "spenso-workspace";
          version = "0.4.1";
        };

        # Run tests with cargo-nextest (workspace-wide)
        workspace-nextest = craneLib.cargoNextest (commonArgs
          // {
            inherit cargoArtifacts;
            pname = "spenso-workspace";
            version = "0.4.1";
            nativeBuildInputs =
              commonArgs.nativeBuildInputs
              ++ [
                pkgs.cargo-insta
              ];
            partitions = 1;
            partitionType = "count";
            cargoNextestExtraArgs = "--workspace";
          });

        # Workspace-wide tarpaulin coverage
        workspace-tarpaulin = craneLib.cargoTarpaulin (commonArgs
          // {
            inherit cargoArtifacts;
            pname = "spenso-workspace";
            version = "0.4.1";
            cargoTarpaulinExtraArgs = "--workspace --skip-clean --out xml --output-dir $out";
          });
      };
    in {
      checks =
        workspaceChecks
        // spensoChecks
        // spensoMacrosChecks
        // spensoHepLibChecks
        // idensoChecks
        // defaultFeatureChecks
        // shadowingFeatureChecks
        // allFeatureChecks
        // noDefaultFeatureChecks
        // idensoFeatureChecks // spensoShadowingChecks;

      packages =
        {
          # Default to the workspace build
          default = workspace;
          inherit workspace;
        }
        // lib.optionalAttrs (!pkgs.stdenv.isDarwin) {
          # Coverage packages (workspace-wide)
          workspace-llvm-coverage = craneLibLLvmTools.cargoLlvmCov (commonArgs
            // {
              inherit cargoArtifacts;
              pname = "spenso-workspace";
              version = "0.4.1";
              cargoExtraArgs = "--workspace";
            });

          workspace-tarpaulin-coverage = craneLib.cargoTarpaulin (commonArgs
            // {
              inherit cargoArtifacts;
              pname = "spenso-workspace";
              version = "0.4.1";
              cargoTarpaulinExtraArgs = "--skip-clean --out xml --output-dir $out --workspace";
            });
        };

      devShells.default = craneLib.devShell {
        # Inherit inputs from checks.
        checks = self.checks.${system};

        # Additional dev-shell environment variables can be set directly
        # MY_CUSTOM_DEVELOPMENT_VAR = "something else";
        # RUSTFLAGS = "-C target-cpu=native -Clink-arg=-fuse-ld=${pkgs.mold}/bin/mold";

        # Extra inputs can be added here; cargo and rustc are provided by default.
        packages = [
          pkgs.cargo-insta
          pkgs.quarto
          pkgs.nodejs
          pkgs.uv
          pkgs.marimo
          pkgs.python313
          pkgs.deno
          pkgs.asciinema
          pkgs.nixd
          pkgs.nil
          pkgs.nixfmt-classic
          pkgs.bfg-repo-cleaner
        ];
      };
    });
}
