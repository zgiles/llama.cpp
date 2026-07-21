{ inputs, ... }:

{
  perSystem =
    {
      config,
      lib,
      system,
      ...
    }:
    {
      devShells =
        let
          pkgs = import inputs.nixpkgs { inherit system; };
          stdenv = pkgs.stdenv;
          scripts = config.packages.python-scripts;
        in
        lib.pipe (config.packages) [
          (lib.concatMapAttrs (
            name: package: {
              ${name} = pkgs.mkShell {
                name = "${name}";
                inputsFrom = [ package ];
                shellHook = ''
                  echo "Entering ${name} devShell"
                  # Allow -march=native: nixpkgs sets NIX_ENFORCE_NO_NATIVE=1 by default, which
                  # silently strips -march=native from GGML_NATIVE builds -> generic no-AVX512/VNNI
                  # binaries (much slower on CPU, and can invert perf results). We build for the
                  # local host here, so opt back into native codegen.
                  export NIX_ENFORCE_NO_NATIVE=0
                '';
              };
              "${name}-extra" =
                if (name == "python-scripts") then
                  null
                else
                  pkgs.mkShell {
                    name = "${name}-extra";
                    inputsFrom = [
                      package
                      scripts
                    ];
                    # Extra packages that *may* be used by some scripts
                    packages = [
                        pkgs.python3Packages.tiktoken
                    ];
                    shellHook = ''
                      echo "Entering ${name} devShell"
                      addToSearchPath "LD_LIBRARY_PATH" "${lib.getLib stdenv.cc.cc}/lib"
                      # See note above: opt back into -march=native for CPU-optimized builds.
                      export NIX_ENFORCE_NO_NATIVE=0
                    '';
                  };
            }
          ))
          (lib.filterAttrs (name: value: value != null))
        ];
    };
}
