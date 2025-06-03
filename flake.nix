{
  description = "Dev shell for the RPS";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.05";
  };

  outputs = {
    self,
    nixpkgs,
    ...
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
      };
    };
  in {
    devShells.${system}.default = pkgs.mkShell {
      buildInputs = [
        pkgs.ansible
        pkgs.ansible-lint
        pkgs.sshpass
        pkgs.black

        (pkgs.python39.withPackages (ps: [
          ps.flask
          ps.opencv4
          ps.numpy
        ]))
      ];
    };
  };
}
