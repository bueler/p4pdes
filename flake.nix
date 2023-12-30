{
  description = "flake for p4pdes repo";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-23.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShellNoCC {
          packages = [
            pkgs.petsc
            pkgs.openmpi
          ];
          shellHook = ''
            export PETSC_DIR=${pkgs.petsc.outPath}
          '';
        };
      }
    );
}
