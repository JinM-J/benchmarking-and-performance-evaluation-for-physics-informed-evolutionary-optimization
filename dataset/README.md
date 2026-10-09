# Reference data

Paper problems F1–F11 use `f01.npz`–`f11.npz`. Their generators are
`generate_f01.py`–`generate_f11.py`. Precomputed arrays are distributed separately;
[reference_manifest.json](reference_manifest.json) records their SHA256, size,
shape, and dtype.

## Import and check

Run from the repository root:

```bash
python dataset/import_data.py --source-dir /path/to/reference-download
python dataset/check_data.py --sha256
```

Input names are `f01.npz`–`f11.npz`. Use `--problems F6 F7`, for example,
to import a subset of the reference fields.

The importer validates every selected input and existing destination before
writing. It keeps matching destinations and rejects mismatches or an identical
source/destination directory. Files are copied to this directory by default;
`--destination-dir` selects another location. Optional `--link` creates hard links
on the same filesystem: both paths share data and must be treated as immutable.
Imports preserve the source file bytes and leave the source files intact.

## Generate a field

Generation is explicit and can require substantial time and RAM:

```bash
python dataset/generate_f06.py --out dataset/f06.npz
python dataset/check_data.py --problems f06
```

Generators refuse existing outputs unless `--overwrite` is given. For an
installation check, use the small [F6 demonstration](../examples/demo_f06.py).
A regenerated file need not match the archived byte hash because ZIP metadata,
compression, and numerical libraries can differ.

| Problem | Stored u shape | dtype | Construction |
| --- | --- | --- | --- |
| F1 | 6001×80001 | float64 | Centered differences, explicit Euler |
| F2 | 16001×6401 | float64 | Wave leapfrog, Taylor startup, CFL=1 |
| F3 | 20001×20001 | float64 | Explicit reaction–diffusion, preserved endpoint updates |
| F4 | 1201×640001 | float64 | Explicit heat/source update |
| F5 | 1001×50001 | float64 | Explicit heat/source update, zero endpoints |
| F6 | 24553×9601 | float64 | Manufactured exact solution sampled on a grid |
| F7 | 1000×1000 | float64 | Nine-point Poisson stencil, preconditioned CG |
| F8 | 200×10001 | float64 | Periodic finite differences, RK4, subsampled in time |
| F9 | 500×2001×1000 | float32 | Reaction half-steps and FFT diffusion splitting |
| F10 | 1000×2001×1000 | float32 | RK2/Rusanov and DST diffusion; 4001 internal time points |
| F11 | 500×2001×1000 | float32 | Source half-steps and FFT diffusion splitting |

Plan for peak working memory, not compressed file size: F4's u alone is about
6.15 GB. Reference loading promotes parametric fields to float64, and sorting
F10's parameter axis can require another copy. Generation also uses temporary
grids. The metadata label `Strang` does not establish global second-order
accuracy: reaction half-steps use Euler and source half-steps reuse
the old-time source.

## Array conventions

- Two-dimensional fields store `x`, `t`, and `u`. F7 uses `t` for its second
  spatial coordinate and stores u in transposed spatial order.
- F6 also stores `meta_json` with its coefficients and packet definition.
- Parametric fields add `mu` or `lognu`, metadata, and a stored `j` array.
  F10's stored `j` integrates u²; its optimization objective instead computes
  dissipation through the problem interface.

See [problem definitions](../problems/README.md) for discrete objective integrals
and the F3/F5 boundary conventions. Shape and dtype alone do not prove
field identity; a changed field requires numerical validation before reuse as
the reference dataset.
