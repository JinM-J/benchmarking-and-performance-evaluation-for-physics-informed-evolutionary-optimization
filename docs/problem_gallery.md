# F1–F11 problem gallery

[Benchmark overview](../README.md) · [Problem definitions](../problems/README.md) · [Reference solutions](../reference/README.md)

This gallery pairs objective landscapes with views of the corresponding PDE state surfaces. Colors show the objective value F; the height coordinate u in a state-surface panel shows the PDE state. Parameterized problems include several fixed-parameter slices.

**Figure version:** earlier manuscript illustrations distributed with the revision source on 26 September 2026. Original plot labels, color scales and star markers are retained. Use the linked problem definitions for the current equations and domains, and the reference scripts for numerical values and verification; the plotted stars alone do not establish optimality.

[F1](#f1) | [F2](#f2) | [F3](#f3) | [F4](#f4) | [F5](#f5) | [F6](#f6) | [F7](#f7) | [F8](#f8) | [F9](#f9) | [F10](#f10) | [F11](#f11)

## F1

**Viscous Burgers equation.** The state-dependent inequalities produce a narrow feasible region in the space–time domain.

[Definition](problems/f01.md) · [Reference calculation](../reference/f01.py)

| Objective landscape / decision-space slices | PDE state surface colored by objective |
| --- | --- |
| ![F1 objective landscape](figures/f01_landscape.png) | ![F1 PDE state geometry](figures/f01_geometry.png) |

## F2

**Wave equation.** Equality tolerance bands couple the wave state to an additional algebraic variable. The original panels use x1 for x and x2 for the algebraic variable z.

[Definition](problems/f02.md) · [Reference calculation](../reference/f02.py)

| Objective landscape / decision-space slices | PDE state surface colored by objective |
| --- | --- |
| ![F2 objective landscape](figures/f02_landscape.png) | ![F2 PDE state geometry](figures/f02_geometry.jpg) |

## F3

**Reaction–diffusion equation.** A Rastrigin-type objective maps the nonlinear state field into a multimodal optimization landscape.

[Definition](problems/f03.md) · [Reference calculation](../reference/f03.py)

| Objective landscape / decision-space slices | PDE state surface colored by objective |
| --- | --- |
| ![F3 objective landscape](figures/f03_landscape.png) | ![F3 PDE state geometry](figures/f03_geometry.jpg) |

## F4

**Forced heat equation.** A six-hump camel objective is coupled to a field driven by spatially segmented, time-dependent forcing.

[Definition](problems/f04.md) · [Reference calculation](../reference/f04.py)

| Objective landscape / decision-space slices | PDE state surface colored by objective |
| --- | --- |
| ![F4 objective landscape](figures/f04_landscape.jpg) | ![F4 PDE state geometry](figures/f04_geometry.png) |

## F5

**Forced heat equation with inequalities.** Polynomial inequalities restrict the state-dependent feasible region.

[Definition](problems/f05.md) · [Reference calculation](../reference/f05.py)

| Objective landscape / decision-space slices | PDE state surface colored by objective |
| --- | --- |
| ![F5 objective landscape](figures/f05_landscape.png) | ![F5 PDE state geometry](figures/f05_geometry.jpg) |

## F6

**Convection–diffusion equation with a localized feature.** A small moving state feature creates a localized objective valley around the target time.

[Definition](problems/f06.md) · [Reference calculation](../reference/f06.py)

| Objective landscape / decision-space slices | PDE state surface colored by objective |
| --- | --- |
| ![F6 objective landscape](figures/f06_landscape.png) | ![F6 PDE state geometry](figures/f06_geometry.png) |

## F7

**Poisson equation on a perforated domain.** Circular exclusions shape the spatial domain, while a Branin-type objective depends on the steady state.

[Definition](problems/f07.md) · [Reference calculation](../reference/f07.py)

| Objective landscape / decision-space slices | PDE state surface colored by objective |
| --- | --- |
| ![F7 objective landscape](figures/f07_landscape.png) | ![F7 PDE state geometry](figures/f07_geometry.png) |

## F8

**Kuramoto–Sivashinsky equation.** An equality tolerance band selects narrow feasible regions on a field with high-order nonlinear dynamics.

[Definition](problems/f08.md) · [Reference calculation](../reference/f08.py)

| Objective landscape / decision-space slices | PDE state surface colored by objective |
| --- | --- |
| ![F8 objective landscape](figures/f08_landscape.png) | ![F8 PDE state geometry](figures/f08_geometry.jpg) |

## F9

**Parametric reaction–diffusion equation.** Parameter slices illustrate changes in the field and the Goldstein–Price-type objective. The plotted parameter eta corresponds to mu in the code.

[Definition](problems/f09.md) · [Reference calculation](../reference/f09.py)

| Objective landscape / decision-space slices | PDE state surface colored by objective |
| --- | --- |
| ![F9 objective landscape](figures/f09_landscape.png) | ![F9 PDE state geometry](figures/f09_geometry.jpg) |

## F10

**Parametric Burgers equation.** Viscosity changes the state field and a dissipation-dependent Rosenbrock objective. The plotted eta is viscosity nu; the optimization coordinate is lognu.

[Definition](problems/f10.md) · [Reference calculation](../reference/f10.py)

| Objective landscape / decision-space slices | PDE state surface colored by objective |
| --- | --- |
| ![F10 objective landscape](figures/f10_landscape.jpg) | ![F10 PDE state geometry](figures/f10_geometry.jpg) |

## F11

**Parametric forced heat equation.** Parameterized forcing changes the field and an Ackley-type objective involving input work. The plotted eta corresponds to mu in the code.

[Definition](problems/f11.md) · [Reference calculation](../reference/f11.py)

| Objective landscape / decision-space slices | PDE state surface colored by objective |
| --- | --- |
| ![F11 objective landscape](figures/f11_landscape.png) | ![F11 PDE state geometry](figures/f11_geometry.png) |
