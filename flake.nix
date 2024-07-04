{
  description = "Build a cargo project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

    crane = {
      url = "github:ipetkov/crane";
      inputs.nixpkgs.follows = "nixpkgs";
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

        SYMBOLICA_LICENSE = builtins.getEnv "SYMBOLICA_LICENSE";
        RUST_BACKTRACE = 1;
        nativeBuildInputs = [
          pkgs.git
          pkgs.gmp.dev
          pkgs.gnum4
          pkgs.mpfr.dev
          pkgs.cargo-insta
        ];
        buildInputs =
          [
            pkgs.git
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
      cargoArtifacts = craneLib.buildDepsOnly commonArgs;

      # Build the actual crate itself, reusing the dependency
      # artifacts from above.
      spenso = craneLib.buildPackage (commonArgs
        // {
          inherit cargoArtifacts;
          doCheck = true;
          cargoExtraArgs = " --all-features";
        });
    in {
      checks = {
        # Build the crate as part of `nix flake check` for convenience
        inherit spenso;

        # Run clippy (and deny all warnings) on the crate source,
        # again, reusing the dependency artifacts from above.
        #
        # Note that this is done as a separate derivation so that
        # we can block the CI if there are issues here, but not
        # prevent downstream consumers from building our crate by itself.
        spenso-clippy = craneLib.cargoClippy (commonArgs
          // {
            inherit cargoArtifacts;
            cargoClippyExtraArgs = "--all-features";
          });

        spenso-doc = craneLib.cargoDoc (commonArgs
          // {
            inherit cargoArtifacts;

            cargoExtraArgs = "--all-features";
          });

        # Check formatting
        spenso-fmt = craneLib.cargoFmt {
          inherit src;
        };

        # Audit dependencies
        spenso-audit = craneLib.cargoAudit {
          inherit src advisory-db;
          cargoExtraArgs = "-- --all-features";
        };

        # Run tests with cargo-nextest
        # Consider setting `doCheck = false` on `spenso` if you do not want
        # the tests to run twice
        spenso-nextest = craneLib.cargoNextest (commonArgs
          // {
            inherit cargoArtifacts;
            nativeBuildInputs =
              commonArgs.nativeBuildInputs
              ++ [
                pkgs.cargo-insta
                # pkgs.breakpointHook
              ];
            partitions = 1;
            partitionType = "count";
            cargoNextestExtraArgs = "--all-features";
          });
      };

      packages =
        {
          default = spenso;
        }
        // lib.optionalAttrs (!pkgs.stdenv.isDarwin) {
          spenso-llvm-coverage = craneLibLLvmTools.cargoLlvmCov (commonArgs
            // {
              inherit cargoArtifacts;
              cargoExtraArgs = "";
            });

          spenso-tarpaulin-coverage = craneLib.cargoTarpaulin (commonArgs
            // {
              inherit cargoArtifacts;
              cargoTarpaulinExtraArgs = "--skip-clean --out xml --output-dir $out --all-features";
            });
        };

      devShells.default = craneLib.devShell {
        # Inherit inputs from checks.
        checks = self.checks.${system};

        # Additional dev-shell environment variables can be set directly
        # MY_CUSTOM_DEVELOPMENT_VAR = "something else";

        # Extra inputs can be added here; cargo and rustc are provided by default.
        packages = [
          pkgs.cargo-insta
          pkgs.git
          # pkgs.ripgrep
          pkgs.quarto
          pkgs.deno

          pkgs.bfg-repo-cleaner
        ];
      };
    });
}
