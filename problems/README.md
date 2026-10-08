# Benchmark problem definitions: F1–F11

The eleven benchmark problems combine PDE states with algebraic objectives and
constraints, spanning the PDE dynamics, landscape characteristics,
state–decision coupling, and feasible-region geometry described in the paper.

The linked problem pages present the **manuscript's problem introductions and
mathematical formulations**, including domains, PDEs, initial/boundary conditions,
objectives, constraints and reference solutions. Each page separately describes
the numerical conventions used by the executable implementation. The
[benchmark design](../docs/benchmark_design.md) introduces the unified formulation
and four design axes; the [problem gallery](../docs/problem_gallery.md) provides
the corresponding manuscript illustrations.

Read [all F1–F11 definitions in one document](../docs/problem_definitions.md),
or select a problem below to read its individual definition. Bounds and transformations
in this overview describe the supplied two- or three-coordinate implementation.

| Paper / code | Decision bounds | Objective family | Algebraic constraints |
| :--- | :--- | :--- | :--- |
| [F1 / f01](../docs/problems/f01.md) | x∈[−1,1], t∈[0,1] | Sinusoidal fractional, z₁=3(x+1), z₂=3(2−u) | Two inequalities |
| [F2 / f02](../docs/problems/f02.md) | x∈[0,5], t∈[0,1], z∈[0,10] | Quadratic, including −2xz | Two equalities with 0.1 bands |
| [F3 / f03](../docs/problems/f03.md) | x∈[−5,5], t∈[0,1] | Rastrigin, z₁=x−1.9004, z₂=u+1.153 | None |
| [F4 / f04](../docs/problems/f04.md) | x∈[0,π], t∈[0,2] | Six-hump camel, state offset u−1 | None |
| [F5 / f05](../docs/problems/f05.md) | x∈[0,3], t∈[0,2] | −x−u+0.5 | Two polynomial inequalities |
| [F6 / f06](../docs/problems/f06.md) | x∈[−1,1], t∈[0,1] | Right-side 5% response objective | None |
| [F7 / f07](../docs/problems/f07.md) | (x₁,x₂)∈[0,5]² outside four disks | Branin(x₁,u) | Geometric exclusion; see below |
| [F8 / f08](../docs/problems/f08.md) | x∈[0,2], t∈[0,1] | x²+(u−1)² | u−x²=0 with 0.001 band |
| [F9 / f09](../docs/problems/f09.md) | x∈[−2,2], t∈[0,2], μ∈[−1,1] | Goldstein–Price, b=J+0.3 | None |
| [F10 / f10](../docs/problems/f10.md) | x∈[−1,1], t∈[0,1], logν∈[ln(0.001),0] | Shifted Rosenbrock in u,J | None |
| [F11 / f11](../docs/problems/f11.md) | x∈[−2,2], t∈[0,2], μ∈[−1,1] | Ackley in 32u,20(J−0.5) | None |

For F2, z is an algebraic decision variable; it is not a PDE query coordinate.
For F9–F11, the third decision coordinate parameterizes the PDE. No additional
arbitrary-dimensional algebraic extensions are included in the tested main preset.
State values are determined by the stored field; no universal additional interval
constraint on u is applied.

## Shared evaluation conventions

The experiments substitute the reference or surrogate state into the objective
and algebraic constraints. For inequalities $g_i\leq0$ and equality bands
$|h_j|\leq\epsilon_j$, the aggregate algebraic violation is

$$
\mathrm{CV}=\sum_i\max(0,g_i)
            +\sum_j\max(0,|h_j|-\epsilon_j).
$$

F2 uses $\epsilon_1=\epsilon_2=0.1$ and F8 uses $\epsilon=10^{-3}$.
Reported feasibility additionally requires finite values and
$\mathrm{CV}\leq10^{-4}$. F7's spatial exclusions are handled by its geometric
query check and penalty state, as described on its problem page.

The best reported objective is the minimum feasible reference-evaluated value
among candidates whose HF observations were consumed by the algorithm.
Reference optimum calculations and diagnostic evaluations are separate from this
search archive. A PDE discretization error, an algebraic root residual and a
discrete objective gap measure different quantities.

## Common problem interface

The executable definitions are in `problems/f01.py` through `f11.py`.
Their `physics` interfaces provide PDE operators, source terms and
initial/boundary conditions. Objectives and algebraic violations are separate
methods, so the same problem can be used with different surrogate models and
optimizers.

| Interface | Meaning |
| --- | --- |
| `decision_bounds` | Bounds in the coordinate order searched by the optimizer |
| `query_bounds`, `decision_to_query` | PDE query domain and decision-to-state coordinate mapping |
| `physics` | PDE residual, derivative requirements, initial and boundary conditions |
| `attach_data`, `load_reference` | Attach dataset metadata and load the reference interpolator |
| `evaluate_fitness` | Reduced objective using a supplied state provider |
| `violation`, `constraint_components` | Aggregate violation and individual algebraic constraints |
| `is_valid_query` | Geometric validity of a PDE query |

For example, after importing `dataset/f06.npz` using the
[data instructions](../dataset/README.md), evaluate an F6 decision from the
repository root:

```python
import numpy as np
from core.registry import make_problem
from evaluation.oracle import HighFidelityOracle
from evaluation.decision_eval import DecisionEvaluator

problem = make_problem("f06")
data_path = "dataset/f06.npz"
problem.attach_data(data_path)
reference = problem.load_reference(data_path)
oracle = HighFidelityOracle(reference)
evaluator = DecisionEvaluator(problem, oracle, reference)

decisions = np.array([[0.1, 0.7]])
objective, components, violation = evaluator.evaluate(decisions)
print(objective, violation, oracle.n_state_queries)
```

This charges one state-label query. The interpolated-field value can differ
slightly from the exact continuous minimum. For the complete
load–train–optimize workflow on a small field, use the
[F6 demonstration](../examples/demo_f06.py).

## Preserved boundary behavior

**F3 (f03).** The generator advances the interior reaction–diffusion nodes with
explicit Euler. At initialization it copies the right endpoint and its adjacent
node to their left counterparts. At positive time layers both endpoints remain
zero, and the left adjacent node is overwritten by the right adjacent node after
the interior update. The surrogate physics interface specifies equality of the
boundary values only. This behavior is retained for compatibility; it is not a
standard periodic finite-difference discretization and should not be described
as enforcing both u and uₓ periodicity.

**F5 (f05).** The generator holds both endpoints at zero after initialization;
the right initial value differs from zero only by sine roundoff. The surrogate
interface requires equality of endpoint values without fixing their common
value and without matching derivatives. Thus the data satisfy a more specific
boundary condition than the information supplied by this interface.

These conventions define the implemented benchmark. Changing them produces a
different data/problem version and requires separate numerical validation.

## Other numerical details

- **F1:** only the objective clamps z₁ below by `1e-10`; constraints use the
  original transformation. The decision interval includes its endpoints.
- **F4:** `u_t=u_xx+1.5(S-u)`, zero IC/Dirichlet BC; S has four π/4-wide indicator
  segments, with coefficients `1.1+5 sin(t/4+i/10)`, i=1,…,4.
- **F6:** the fixed solution is
  `u=0.3 exp(-t) sin(πx)+0.03 exp(-((x+0.25-0.5t)/0.10)^2)`.
  It satisfies `u_t+0.5u_x-0.0005u_xx=s` with the explicitly implemented source
  and exact IC/BC values. The array samples that expression; its interpolation
  is still approximate. The right-side 5% response objective has continuous minimum 0 at
  `(0.1+0.1 sqrt(log(20)),0.7)`; the archived field is unchanged.
- **F7:** the sign is **Δu=f**, with
  `f=20(20+x₁²+x₂²)sin(2πx₁)sin(4πx₂)`. Outer BC is 0.2; hole-wall BC is 1.
  Disk centers/radii are `(4.3,3.5,0.5)`, `(1.5,1.2,0.4)`, `(2.2,2.7,0.6)` and
  `(0.6,0.5,0.3)`. Discrete hole walls use a band of width 0.75h. Stored u axes
  are transposed on loading. Hole-interior NaNs are filled for interpolation;
  invalid geometric queries are assigned the objective penalty state `u=1e6`.
  This is not a positive algebraic `violation` channel: that method returns zero.
- **F8:** the solver samples the initial profile on `[0,2)` and uses periodic
  differences with RK4. The periodic extension has an initial seam; do not infer
  initial smoothness. The physics interface declares periodic derivatives through order three; the
  fixed training configuration uses u and uₓ boundary losses. Reference
  interpolation wraps spatially and evaluates the exact IC at t=0.
- **F9/F11:** periodic generator grids omit the spatial right endpoint.
  F9 wraps/clips query coordinates in its normalization path; F11 clips them.
- **F10:** log-viscosity is stored in descending order and reversed with the
  state axis on loading. ν=0 is not in the domain.
- **F11:** the nominal diffusivity is 0.05; attachment reads its float32 metadata
  value, `0.05000000074505806`.

## Integral objectives

All indices below start at zero. U is the state returned by the relevant provider,
which can be the reference interpolator or the surrogate. NPZ members named `j`
are **not** used by `evaluate_fitness`.

| Paper / code | Grid used by objective | Discrete J |
| :--- | :--- | :--- |
| F9 / f09 | xᵢ=−2+0.4i, tⱼ=0.2j; i,j=0,…,9 | `0.08 Σᵢⱼ Uᵢⱼ` |
| F10 / f10 | xᵢ=−1+2i/9, tⱼ=0.1j; i,j=0,…,9 | `ν(2/9)(0.1) Σᵢ₌₁⁸Σⱼ₌₀⁹ [(Uᵢ₊₁,ⱼ−Uᵢ₋₁,ⱼ)/(4/9)]²` |
| F11 / f11 | xᵢ=−2+4i/9, tⱼ=0.2j; i,j=0,…,9 | `(4/9)(0.2) Σᵢⱼ Uᵢⱼ·2(1+μ)sin(πxᵢ)cos(2πtⱼ)` |

These are fixed rectangular sums. F10's stored `j` is an integral of u², a
different quantity from the dissipation objective. F11 clips the state query at
x=2 to the last stored spatial point, while retaining source weights computed at
the original integration coordinates. F10/F11 archive/pool evaluation rounds J
to float32, preserving the implementation's numerical conventions.

Reference targets inherited from algebraic functions are not, by themselves,
proofs that the coupled PDE field attains them at a complete decision tuple.
See [reference verification](../reference/README.md) for algebraic bounds,
stored-field attainability checks and the distinction between continuous and
discrete integral objectives.
F2/F8 exact-equality reference targets also differ in meaning from optima of their
implemented tolerance bands.
