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

      src = craneLib.cleanCargoSource ./.;

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

      # Build *just* the cargo dependencies (of the entire workspace),
      # so we can reuse all of that work (e.g. via cachix) when running in CI
      # It is *highly* recommended to use something like cargo-hakari to avoid
      # cache misses when building individual top-level-crates
      cargoArtifacts = craneLib.buildDepsOnly (
        commonArgs
        // {
          pname = "spenso-workspace-deps";
          version = "1.0.0";
        }
      );

      individualCrateArgs =
        commonArgs
        // {
          inherit cargoArtifacts;
          # NB: we disable tests since we'll run them all via cargo-nextest
          doCheck = false;
        };

      fileSetForCrate = crate:
        lib.fileset.toSource {
          root = ./.;
          fileset = lib.fileset.unions [
            ./Cargo.toml
            ./Cargo.lock
            (craneLib.fileset.commonCargoSources ./spenso)
            (craneLib.fileset.commonCargoSources ./spenso-macros)
            (craneLib.fileset.commonCargoSources ./spenso-hep-lib)
            (craneLib.fileset.commonCargoSources ./idenso)
            (craneLib.fileset.commonCargoSources crate)
          ];
        };

      # Build the top-level crates of the workspace as individual derivations.
      # This allows consumers to only depend on (and build) only what they need.
      spenso = craneLib.buildPackage (
        individualCrateArgs
        // {
          inherit (craneLib.crateNameFromCargoToml {cargoToml = ./spenso/Cargo.toml;}) pname version;
          cargoExtraArgs = "-p spenso";
          src = fileSetForCrate ./spenso;
        }
      );
      spenso-macros = craneLib.buildPackage (
        individualCrateArgs
        // {
          inherit (craneLib.crateNameFromCargoToml {cargoToml = ./spenso-macros/Cargo.toml;}) pname version;
          cargoExtraArgs = "-p spenso-macros";
          src = fileSetForCrate ./spenso-macros;
        }
      );
      spenso-hep-lib = craneLib.buildPackage (
        individualCrateArgs
        // {
          inherit (craneLib.crateNameFromCargoToml {cargoToml = ./spenso-hep-lib/Cargo.toml;}) pname version;
          cargoExtraArgs = "-p spenso-hep-lib";
          src = fileSetForCrate ./spenso-hep-lib;
        }
      );
      idenso = craneLib.buildPackage (
        individualCrateArgs
        // {
          inherit (craneLib.crateNameFromCargoToml {cargoToml = ./idenso/Cargo.toml;}) pname version;
          cargoExtraArgs = "-p idenso";
          src = fileSetForCrate ./idenso;
        }
      );

      # Build the entire workspace
      workspace = craneLib.buildPackage (commonArgs
        // {
          inherit cargoArtifacts;
          pname = "spenso-workspace";
          version = "1.0.0";
          doCheck = false;
          cargoExtraArgs = "--workspace";
        });

      # Helper function to create checks for a specific crate using workspace subset pattern
      mkChecksForCrate = crateName: let
        crateInfo = craneLib.crateNameFromCargoToml {cargoToml = ./${crateName}/Cargo.toml;};
      in {
        "${crateInfo.pname}-clippy" = craneLib.cargoClippy (commonArgs
          // {
            inherit cargoArtifacts;
            inherit (crateInfo) pname version;
            cargoExtraArgs = "-p ${crateInfo.pname}";
            cargoClippyExtraArgs = "--all-targets -- --deny warnings";
          });

        "${crateInfo.pname}-doc" = craneLib.cargoDoc (commonArgs
          // {
            inherit cargoArtifacts;
            inherit (crateInfo) pname version;
            cargoExtraArgs = "-p ${crateInfo.pname}";
          });

        "${crateInfo.pname}-nextest" = craneLib.cargoNextest (commonArgs
          // {
            inherit cargoArtifacts;
            inherit (crateInfo) pname version;
            cargoExtraArgs = "-p ${crateInfo.pname}";
            partitions = 1;
            partitionType = "count";
          });

        "${crateInfo.pname}-tarpaulin" = craneLib.cargoTarpaulin (commonArgs
          // {
            inherit cargoArtifacts;
            inherit (crateInfo) pname version;
            cargoExtraArgs = "-p ${crateInfo.pname}";
            cargoTarpaulinExtraArgs = "--skip-clean --out xml --output-dir $out";
          });
      };

      # Helper function to create per-crate checks with features using workspace subset pattern
      mkCrateChecksWithFeatures = crateName: features: featuresName: let
        crateInfo = craneLib.crateNameFromCargoToml {cargoToml = ./${crateName}/Cargo.toml;};
      in {
        "${crateInfo.pname}-clippy-${featuresName}" = craneLib.cargoClippy (commonArgs
          // {
            inherit cargoArtifacts;
            inherit (crateInfo) pname version;
            cargoExtraArgs = "-p ${crateInfo.pname} ${features}";
            cargoClippyExtraArgs = "--all-targets -- --deny warnings";
          });

        "${crateInfo.pname}-nextest-${featuresName}" = craneLib.cargoNextest (commonArgs
          // {
            inherit cargoArtifacts;
            inherit (crateInfo) pname version;
            cargoExtraArgs = "-p ${crateInfo.pname} ${features}";
            partitions = 1;
            partitionType = "count";
          });

        "${crateInfo.pname}-doc-${featuresName}" = craneLib.cargoDoc (commonArgs
          // {
            inherit cargoArtifacts;
            inherit (crateInfo) pname version;
            cargoExtraArgs = "-p ${crateInfo.pname} ${features}";
          });

        "${crateInfo.pname}-tarpaulin-${featuresName}" = craneLib.cargoTarpaulin (commonArgs
          // {
            inherit cargoArtifacts;
            inherit (crateInfo) pname version;
            cargoExtraArgs = "-p ${crateInfo.pname} ${features}";
            cargoTarpaulinExtraArgs = "--skip-clean --out xml --output-dir $out";
          });
      };

      # Create checks for each crate
      spensoChecks = mkChecksForCrate "spenso";
      spensoMacrosChecks = mkChecksForCrate "spenso-macros";
      spensoHepLibChecks = mkChecksForCrate "spenso-hep-lib";
      idensoChecks = mkChecksForCrate "idenso";

      # Create per-crate feature checks for important combinations
      idensoFeatureChecks = mkCrateChecksWithFeatures "idenso" "--features bincode" "bincode";
      spensoShadowingChecks = mkCrateChecksWithFeatures "spenso" "--features shadowing" "shadowing";

      # Create per-crate all-features checks
      spensoAllFeaturesChecks = mkCrateChecksWithFeatures "spenso" "--all-features" "all-features";
      spensoMacrosAllFeaturesChecks = mkCrateChecksWithFeatures "spenso-macros" "--all-features" "all-features";
      spensoHepLibAllFeaturesChecks = mkCrateChecksWithFeatures "spenso-hep-lib" "--all-features" "all-features";
      idensoAllFeaturesChecks = mkCrateChecksWithFeatures "idenso" "--all-features" "all-features";

      # Workspace-wide checks
      workspaceChecks = {
        # Build the crates as part of `nix flake check` for convenience
        inherit spenso spenso-macros spenso-hep-lib idenso workspace;

        # Run clippy (and deny all warnings) on the workspace source,
        # again, reusing the dependency artifacts from above.
        #
        # Note that this is done as a separate derivation so that
        # we can block the CI if there are issues here, but not
        # prevent downstream consumers from building our crate by itself.
        # Workspace-wide checks
        workspace-clippy = craneLib.cargoClippy (commonArgs
          // {
            inherit cargoArtifacts;
            pname = "spenso-workspace";
            version = "1.0.0";
            cargoClippyExtraArgs = "--all-targets -- --deny warnings";
          });

        workspace-doc = craneLib.cargoDoc (commonArgs
          // {
            inherit cargoArtifacts;
            pname = "spenso-workspace";
            version = "1.0.0";
          });

        # Check formatting
        workspace-fmt = craneLib.cargoFmt {
          inherit src;
          pname = "spenso-workspace-fmt";
          version = "1.0.0";
        };

        # Run tests with cargo-nextest
        # Consider setting `doCheck = false` on other crate derivations
        # if you do not want the tests to run twice
        workspace-nextest = craneLib.cargoNextest (commonArgs
          // {
            inherit cargoArtifacts;
            pname = "spenso-workspace";
            version = "1.0.0";
            partitions = 1;
            partitionType = "count";
          });

        # Workspace-wide tarpaulin coverage
        workspace-tarpaulin = craneLib.cargoTarpaulin (commonArgs
          // {
            inherit cargoArtifacts;
            pname = "spenso-workspace";
            version = "1.0.0";
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
        // idensoFeatureChecks
        // spensoShadowingChecks
        // spensoAllFeaturesChecks
        // spensoMacrosAllFeaturesChecks
        // spensoHepLibAllFeaturesChecks
        // idensoAllFeaturesChecks;

      packages =
        {
          inherit spenso spenso-macros spenso-hep-lib idenso workspace;
          default = workspace;
        }
        // lib.optionalAttrs (!pkgs.stdenv.isDarwin) {
          # Coverage packages (workspace-wide)
          workspace-llvm-coverage = craneLibLLvmTools.cargoLlvmCov (commonArgs
            // {
              inherit cargoArtifacts;
              pname = "spenso-workspace";
              version = "1.0.0";
              cargoExtraArgs = "--workspace";
            });

          workspace-tarpaulin-coverage = craneLib.cargoTarpaulin (commonArgs
            // {
              inherit cargoArtifacts;
              pname = "spenso-workspace";
              version = "1.0.0";
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
