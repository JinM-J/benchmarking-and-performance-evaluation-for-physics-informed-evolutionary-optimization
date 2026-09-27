# Benchmark test suite design

[Overview](../README.md) · [F1–F11 definitions](../problems/README.md) · [Problem gallery](problem_gallery.md)

The following description follows the formulation and terminology of
*Benchmarking and Performance Evaluation for Physics-Informed Evolutionary
Optimization*. The problem pages reproduce the manuscript's mathematical
definitions and place numerical implementation conventions in separate sections.

## Motivation and scope

In practice, expensive constrained optimization problems (ECOPs) with physical
information exhibit diverse characteristics arising from different types of
governing PDEs, constraint formulations, and algebraic landscapes. These
landscapes may involve multimodality, strong nonlinearity, flat or deceptive
fitness regions, and complex feasible boundaries induced by physical constraints.

The benchmark aims to extract and systematically control the key factors that
determine optimization difficulty in ECOPs with PDE-based constraints. Such an
abstraction enables reproducible evaluation and facilitates the analysis of how
different physical and optimization factors affect algorithmic performance.

In this work, we focus on a surrogate-assisted form of physics-informed
evolutionary optimization, where physics is primarily incorporated into surrogate
modeling to support the evolutionary search for ECOPs with PDE-based constraints.

## Unified formulation

Based on the above analysis, a unified formulation incorporating the four-axis
properties is adopted as follows:

$$
\begin{aligned}
\min_{\boldsymbol{x}\in\mathbb{R}^{D},\;t,\;u,\;
      \boldsymbol{\eta}\in\mathbb{R}^{D_e}}
    &\quad F(\boldsymbol{x},u,t,\boldsymbol{\eta})\\
\text{s.t.}\quad
    &\mathcal{E}(\boldsymbol{x},u,t;\boldsymbol{\eta})=0,\\
    &g_i(\boldsymbol{x},u,t)\leq0,\quad i=1,\ldots,p,\\
    &h_j(\boldsymbol{x},u,t)=0,\quad j=1,\ldots,q.
\end{aligned}
$$

Here $\boldsymbol{x}$ denotes the spatial variables, $u$ denotes the PDE state
(the PDE solution), $t$ denotes the temporal variable, and
$\boldsymbol{\eta}$ is the physical parameter vector. $F$ is the algebraic
objective, $\mathcal{E}$ represents the governing PDE residual, and $g_i$ and
$h_j$ denote inequality and equality algebraic constraints. Initial and boundary
conditions complete each PDE definition.

The unified formulation provides a common structure for systematically combining
and controlling the above practically motivated characteristics. The reported
reference optima are used only for post-hoc performance assessment and are not
used during the optimization process.

**Implemented formulation.** The experiments use a reduced formulation in which
the reference PDE state $u(\boldsymbol{v})$, with
$\boldsymbol{v}=[\boldsymbol{x},t,\boldsymbol{\eta}]^\top$, is substituted into
the objective and algebraic constraints. Surrogates approximate this state during
search, while final objective values and algebraic feasibility are assessed by
reference evaluations. These evaluations query precomputed PDE data using
multilinear interpolation, with a fixed budget simulating expensive evaluation
conditions. One state evaluation corresponds to one queried state value.
Diagnostic queries for modeling-error assessment are outside the search budget.

The supplied problems have two or three decision coordinates. F2 adds an
algebraic variable $z$, F7 is steady and has two spatial coordinates, and F9–F11
include one PDE parameter. Their exact coordinate order is specified on each
problem page. The [evaluation conventions](../methods/README.md#budget-and-reporting)
also specify the treatment of auxiliary integral queries.

## Four-axis design principles

The benchmark test functions are characterized along four complementary axes,
which represent different sources of complexity arising from PDE-based
constraints, algebraic objective landscapes, and their interactions.

### Type of PDE dynamics

Six representative PDE classes are considered: diffusion-dominated,
reaction–diffusion, convection–diffusion, wave-propagation, elliptic-equilibrium,
and high-order nonlinear dissipative systems. These classes are not intended to
constitute an exhaustive mathematical classification of PDEs, but rather to
provide compact coverage of representative behaviors including diffusion,
nonlinear reaction, transport and shock formation, wave propagation, steady-state
spatial coupling, and high-order nonlinear dynamics.

| PDE class | Representative behavior | Problems |
| --- | --- | --- |
| Diffusion-dominated | Dissipative and smoothing effects with external forcing | [F4](problems/f04.md), [F5](problems/f05.md), [F11](problems/f11.md) |
| Reaction–diffusion | Interplay between diffusion and nonlinear local reaction dynamics | [F3](problems/f03.md), [F9](problems/f09.md) |
| Convection–diffusion | Transport, moving fronts and dissipative smoothing | [F1](problems/f01.md), [F6](problems/f06.md), [F10](problems/f10.md) |
| Wave propagation | Oscillatory dynamics and finite-speed information propagation | [F2](problems/f02.md) |
| Elliptic equilibrium | Steady-state spatial interactions and global coupling effects | [F7](problems/f07.md) |
| High-order nonlinear dissipation | Nonlinear transport, instability growth and fourth-order dissipation | [F8](problems/f08.md) |

Different boundary and initial conditions may further introduce diverse PDE
solution behaviors and additional optimization challenges.

### Type of algebraic functions

The objective function is formulated in an algebraic form and may additionally
involve algebraic constraints. This axis characterizes the landscape properties
of the underlying algebraic functions, including quadratic, Rastrigin-type,
Rosenbrock-type, and fractional oscillatory forms. The resulting landscapes may
range from smooth and unimodal to highly multimodal, with oscillatory or
ridge-like structures.

The selected functions should preserve engineering-related characteristics while
serving as modular landscape components with well-understood optimization
properties. Ackley, Rastrigin, Rosenbrock, Branin, Goldstein–Price and camel-type
functions are included in the suite. Each problem page states the exact
transformation connecting the PDE state to its algebraic objective.

### State–decision coupling mode

The first category corresponds to problems in which the decision variables do
not directly affect the PDE solution field. The parameters in
$\boldsymbol{\eta}$ are fixed. The main challenges are twofold. First, the
nonlinear response of the PDE state $u$ to the decision variables
$\boldsymbol{x}$ and $t$ can lead to distorted and multimodal objective
landscapes, often with ridge-like structures. Second, the constraints are defined
through PDE state rather than explicit forms. As a result, the feasible region
may contain narrow channels, low-dimensional manifolds, or disconnected
components. F1–F8 belong to this category.

The second category corresponds to problems in which the decision variables
directly influence the PDE solution field and its spatiotemporal evolution. In
this setting, the parameters in $\boldsymbol{\eta}$ are no longer fixed.
Consequently, changes in these parameters modify both the PDE structure and its
corresponding solution field. The objective function is typically defined as an
integral functional, resulting in a family of parameterized PDE systems and the
induced optimization problems. F9–F11 belong to this category and combine a local
state value with an integral of the field, its dissipation, or the input work.

Each integral is stated both as a mathematical functional and as the discrete
quantity used in the released implementation. The distinction is particularly
relevant to the reference values for F9–F11.

### Feasible-region geometry

This axis mainly reflects the interactions between PDE-based constraints and
algebraic constraints in shaping the feasible region geometry. Four
representative levels are considered:

1. The first contains no additional algebraic constraints and serves as a
   baseline for isolating PDE-induced difficulty.
2. The second introduces inequality constraints that restrict or deform the
   feasible region.
3. The third uses equality constraints that confine feasible solutions to
   low-dimensional manifolds.
4. The fourth represents disconnected or highly complex feasible domains induced
   by geometric boundaries or coupled constraints.

The [feasible-region illustration](figures/feasible_region_geometry.png) and
[problem gallery](problem_gallery.md) visualize these effects. Numerical equality
tolerances are specified separately from exact mathematical equalities.

The first two axes characterize the two fundamental components of an ECOP with
PDE-based constraints: the governing PDE dynamics and the algebraic optimization
landscape. The latter two axes characterize their interactions with optimization:
state–decision coupling describes how decision variables and PDE states are
coupled, while feasible-region geometry reflects how the PDE state and additional
algebraic or physical constraints jointly determine the feasible solution space.
Their effects may naturally interact in the resulting optimization problem.

## From problem definitions to evaluation

The eleven benchmark problems combine PDE states with algebraic objectives and
constraints, spanning the PDE dynamics, landscape characteristics,
state–decision coupling, and feasible-region geometry described above.

A multidimensional evaluation protocol is established to separately assess
surrogate modeling capability and evolutionary optimization capability,
distinguishing modeling error from optimization error. Standalone surrogate
accuracy is used only as one diagnostic aspect of the framework and does not by
itself determine optimization performance. A surrogate with lower prediction
error does not necessarily yield better optimization results.

Read the [F1–F11 mathematical definitions](../problems/README.md),
[reference-solution calculations](../reference/README.md), and
[experiment reproduction instructions](../reproduction/README.md) for the
corresponding executable components.
