{
  description = "gvc - generative video conversion: video -> captions -> generated frames -> video";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        # CUDA libraries are unfree.
        config.allowUnfree = true;
      };

      python = pkgs.python3;

      pythonDeps =
        ps: with ps; [
          httpx
          typer
          rich
          pillow
          imagehash
        ];

      gvc = python.pkgs.buildPythonApplication {
        pname = "gvc";
        version = "0.1.0";
        pyproject = true;
        src = ./.;
        build-system = [ python.pkgs.hatchling ];
        dependencies = pythonDeps python.pkgs;
        nativeCheckInputs = [
          python.pkgs.pytestCheckHook
          pkgs.ffmpeg-headless
        ];
        makeWrapperArgs = [
          "--prefix PATH : ${pkgs.lib.makeBinPath [ pkgs.ffmpeg-headless ]}"
        ];
      };

      # Only the AI runtimes get CUDA; everything else stays on the binary cache.
      sdCuda = pkgs.stable-diffusion-cpp.override { cudaSupport = true; };

      devPython = python.withPackages (ps: pythonDeps ps ++ [ ps.pytest ]);

      # Runs the checked-out sources, so edits apply without rebuilding.
      gvcDev = pkgs.writeShellScriptBin "gvc" ''
        exec ${devPython}/bin/python -m gvc "$@"
      '';

      mkShell =
        { ollama, sd }:
        pkgs.mkShell {
          packages = [
            devPython
            gvcDev
            pkgs.ffmpeg-headless
            ollama
            sd
          ];
          shellHook = ''
            export PYTHONPATH="$PWD/src''${PYTHONPATH:+:$PYTHONPATH}"
          '';
        };
    in
    {
      packages.${system}.default = gvc;

      devShells.${system} = {
        # GPU box (NVIDIA, CUDA).
        default = mkShell {
          ollama = pkgs.ollama-cuda;
          sd = sdCuda;
        };
        # CPU-only, for development and smoke tests on machines without an NVIDIA GPU.
        cpu = mkShell {
          ollama = pkgs.ollama;
          sd = pkgs.stable-diffusion-cpp;
        };
      };
    };
}
