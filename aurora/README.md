# RFdiffusion IDP co-diffusion on Aurora XPU

Clone the Aurora port and enter its checkout:

```bash
git clone https://github.com/moeenmeigooni/RFdiffusionIDP-Aurora.git
cd RFdiffusionIDP-Aurora
```

This is the paper-era implementation for Liu *et al.*, *Nature* (2025),
“Diffusing protein binders to intrinsically disordered proteins”.  The active
checkout is pinned to RFdiffusion commit `909fc01c032d6e3697a6b50fbb0121abfe007755`
(2025-05-16).  It contains the three scripts named in the paper:

- `examples/design_ppi_flexible_peptide.sh` — sequence-input co-diffusion;
- `examples/design_ppi_flexible_peptide_with_secondarystructure_specification.sh` —
  secondary-structure-conditioned sequence input; and
- `examples/design_partialdiffusion_withseq.sh` — two-sided partial diffusion.

Set `AURORA_PROJECT_ROOT` to a shared project directory visible from both the
login and compute nodes. By default, the scripts use
`/lus/flare/projects/FRAME-IDP/$USER`. The checkout can live anywhere on shared
storage; the environment defaults to
`${AURORA_PROJECT_ROOT}/envs/rfdiffusion-idp-aurora`.

## What changed for Aurora

Upstream RFdiffusion v1.1.0 selects CUDA and uses NVIDIA's DGL-based
SE(3)-Transformer.  DGL does not provide an Aurora/PyTorch-XPU graph backend.
This port retains the model architecture and weights, but replaces only the
inference graph primitives with native PyTorch operations: directed graph
construction, destination-normalized edge softmax, edge-to-node sum, and
edge/node dot product. These operate on `torch.xpu` with Aurora's default
`frameworks` module (currently `frameworks/2026.1.0` on Aurora). Set
`AURORA_FRAMEWORKS_MODULE` to select another site-supported module. The
inference device selection is XPU first,
then CUDA, then CPU; the NVIDIA NVTX profiling ranges are no-ops on XPU.
Although IPEX is available in the framework, this inference port deliberately
uses native PyTorch XPU without importing IPEX: IPEX's TorchScript fusion pass
cannot compile the integer-index arithmetic in the legacy SE(3) basis script.

The paper-era checkout also predates an upstream fix for generated contig
chains: de novo residues have no source-PDB chain identifier, so the original
code aborts before inference. This port backports only that fix, assigning an
unused chain letter to a wholly generated output chain. It is independent of
the XPU changes and is required for the released flexible-peptide examples.

`e3nn==0.3.3`, the version used by upstream, is retained.  A narrowly-scoped
compatibility shim allowlists Python's built-in `slice` while that package loads
its own fixed Clebsch-Gordan constants under PyTorch 2.10's safe loading
default.  RFdiffusion checkpoint loading is not weakened.

The port is inference-only.  The unused DGL data loaders and CUDA training
utilities remain unsupported.

## Activate and validate

On a UAN or compute node:

```bash
repo="$PWD"
export RFDIFFUSION_IDP_REPO="$repo"
source "$repo/aurora/activate_rfdiffusion_idp_aurora.sh"
```

For an allocated-node compatibility smoke that runs all three paper modes with
RFdiffusion's minimum legal 15-step schedule but only two reverse-model
evaluations (not scientifically useful designs):

```bash
cd /path/to/RFdiffusionIDP-Aurora
qsub aurora/idp_codiffusion_smoke.pbs
```

It first validates the XPU kernel and native graph reductions, then writes one
output for each of sequence-input co-diffusion, secondary-structure-conditioned
co-diffusion, and two-sided partial diffusion to a unique subdirectory of
`${AURORA_PROJECT_ROOT}/runs/rfdiffusion-idp-xpu-smoke`.

For a real campaign, begin from an accepted parent complex for two-sided
partial diffusion and use the paper's 50-step schedule with `partial_T` in the
5–25 range; retain the full downstream ProteinMPNN and AF2+initial-guess
screening workflow. This deliberately abbreviated port smoke verifies
execution only and must not be treated as a design campaign.

## Weights and cache policy

The two official Zenodo checkpoints are stored outside the checkout at
`${AURORA_PROJECT_ROOT}/cache/rfdiffusion-idp-aurora/models`
and exposed as the checkout's `models` symlink:

- `InpaintSeq_ckpt.pt` — MD5 `a6f8652938bb45c332ffa683d8ad3509`;
- `InpaintSeq_Fold_ckpt.pt` — MD5 `1e9245a486262dff3cb3286f22a3014d`.

The activation script redirects `HOME`, pip, XDG, temporary, Torch,
TorchInductor, extension-build, matplotlib, and Python-user-site state under
that same project cache root.  It also sets `PYTHONNOUSERSITE=1`; no runtime
cache is intentionally written to the user home directory.

To recreate the environment, use
`bash aurora/install_aurora.sh --weights` from the active checkout.
