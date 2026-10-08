# problems/base.py
"""Mathematical interfaces for PDE-constrained optimization problems.

The decision space is searched by the optimizer and may include variables
not passed to the surrogate (for example, z in F02). The query space Q is
the surrogate/oracle input space with named axes such as (x,t), (x1,x2),
or (x,t,mu). A state provider evaluates the PDE solution u at query points.

PhysicsSpec uses generic coordinates. time_axis is metadata (None for
steady equations such as F07). Derivatives are tuples of axis names:
("x","x") denotes u_xx and mixed derivatives use the same convention.
Initial and boundary conditions specify regions, targets, and derivatives.
Steady problems return no initial conditions.

Problems define mathematics independently of training implementations.
evaluate_fitness handles pointwise or integral objectives. Evolutionary
settings and surrogate hyperparameters belong in protocols/."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class StateProvider(Protocol):
    """Interface for evaluating a PDE state at query points."""

    def evaluate(self, query_points: np.ndarray) -> np.ndarray:
        """Map query_points of shape (n,dim_query) to u of shape (n,)."""
        ...


@dataclass
class Region:
    """Geometric region supporting contains(Q) and sample(n).

    Problems supply the geometry, including outer edges and circular holes.
    Sampling here is deterministic, using grids or parameterized curves."""
    contains_fn: Callable[[np.ndarray], np.ndarray]
    sample_fn: Callable[[int], np.ndarray]

    def contains(self, Q: np.ndarray) -> np.ndarray:
        return self.contains_fn(np.asarray(Q, dtype=float))

    def sample(self, n: int) -> np.ndarray:
        return self.sample_fn(int(n))


@dataclass
class InitialCondition:
    """Initial condition with a target defined on a region.

    derivative=() specifies u=target(Q); ("t",) specifies initial velocity.
    n_points overrides the surrogate sampling count; None uses its default."""
    region: Region
    target: Callable[[np.ndarray], np.ndarray]
    derivative: tuple = ()
    n_points: Optional[int] = None


@dataclass
class BoundaryCondition:
    """Boundary condition and its sampling budget.

    dirichlet: u=target(Q) on region.
    periodic: equal states on the paired regions.
    periodic_derivative: equal specified derivatives on the paired regions;
    include_value also enforces state equality (enabled by default).
    neumann: the specified derivative equals target(Q); targets can depend
    on any query column, including parameter axes.

    n_points overrides the surrogate count. Otherwise points_key selects
    a surrogate budget: bc_points by default, obs_bc_points for F07 holes.
    The problem selects a budget name; its size belongs to the protocol."""
    kind: str
    region: Optional[Region] = None
    target: Optional[Callable[[np.ndarray], np.ndarray]] = None
    region_pair: tuple = ()
    derivative: tuple = ()
    n_points: Optional[int] = None
    points_key: str = "bc_points"
    include_value: bool = True


@dataclass
class LinearOpTerm:
    """One linear-operator term: coef(Q) times a derivative of u.

    deriv lists derivative orders in coordinate_names order; for example,
    u_xx on (x,t) uses (2,0). coef is a scalar or callable(Q)->(N,) evaluated
    at physical coordinates, such as exp(lognu) for F10."""
    deriv: tuple
    coef: object


@dataclass
class LinearOperator:
    """Exact linear part of a PDE: sum coef(Q)*derivative(u) = rhs(Q).

    Operator-informed PIGP constructs L_x L_x' k from these terms.
    Nonlinear equations retain only their exact linear terms: Burgers and
    KS omit u*u_x, Allen-Cahn retains the linear +5u reaction but omits
    -5u^3, and F09 omits alpha*u^2*(1-u). Their rhs is zero; omitted terms
    are absorbed by a learned residual-block noise parameter.
    rhs is callable(Q)->(N,) for known forcing; None denotes zero.
    Coordinates are physical, not normalized."""
    terms: tuple                      # Sequence of LinearOpTerm objects.
    rhs: Optional[Callable[[np.ndarray], np.ndarray]] = None   # None ≡ 0


class PhysicsSpec(ABC):
    """PDE coordinates, derivative requirements, residual, and conditions.

    The problem defines condition locations and values. Sampling counts,
    training epochs, and loss weights belong to surrogate configuration.
    residual consumes precomputed u and derivatives keyed by axis tuples;
    derivative computation belongs to the surrogate backend."""

    @property
    @abstractmethod
    def coordinate_names(self) -> tuple:
        """Query-axis names, such as (x,t), (x1,x2), or (x,t,mu)."""

    @property
    @abstractmethod
    def time_axis(self):
        """Index of the time coordinate, or None for a steady problem."""

    @abstractmethod
    def required_derivatives(self) -> set:
        """Required derivatives expressed as tuples of coordinate names.

        Burgers: {("t",),("x",),("x","x")}.
        KS also needs ("x","x","x","x"); the wave equation needs
        ("x","x") and ("t","t"); F07 needs second derivatives in x1,x2."""

    @abstractmethod
    def residual(self, Q, u, d):
        """Evaluate the residual from derivatives d keyed by axis tuples (NumPy or Torch)."""

    def linear_operator(self):
        """Return the exact linear operator for PIGP, or None if unavailable.

        PIGP rejects a missing operator rather than falling back to plain GP."""
        return None

    def initial_conditions(self) -> tuple:
        """Return initial conditions; steady problems return an empty tuple."""
        return ()

    def boundary_conditions(self) -> tuple:
        """Return separate boundary conditions for outer edges, hole walls, and so on."""
        return ()


class PDEProblem(ABC):
    """PDE-constrained problem with decision/query spaces, objective, and constraints."""

    name: str = "PDEProblem"

    # Decision and query spaces.
    @property
    @abstractmethod
    def decision_bounds(self) -> np.ndarray:
        """Decision bounds of shape (dim_decision,2)."""

    @property
    @abstractmethod
    def query_bounds(self) -> np.ndarray:
        """Surrogate query bounds of shape (dim_query,2)."""

    @property
    @abstractmethod
    def physics(self) -> PhysicsSpec:
        """Mathematical PDE specification."""

    # Decision-to-query mapping.
    @abstractmethod
    def decision_to_query(self, decision: np.ndarray) -> np.ndarray:
        """Map a decision to the coordinates that determine its PDE state."""

    # Query-domain validity; perforated domains override the default.
    def is_valid_query(self, Q: np.ndarray) -> np.ndarray:
        """Return a Boolean mask of shape (n,) for Q of shape (n,dim_query)."""
        return np.ones(len(np.asarray(Q)), dtype=bool)

    # Optimizer-to-surrogate input dtype; None preserves float64.
    # The F09-F11 fitness path rounds query coordinates to float32.
    surrogate_query_dtype = None

    # Optional problem capabilities; defaults retain F01-F11 behavior.
    real_watch_enabled = True      # Snapshot reference monitoring; backends may disable extra evaluations.
    pool_f_block_mode = False      # Whether pool objectives require complete query blocks per decision.
    hf_parameter_axes = None       # Distinct HF parameter axes; None retains the default third-axis rule
                                   # for dim>=3; extensions may declare explicit parameter axes.
    has_pointwise_decision_map = True   # Whether a pointwise decision_to_query mapping exists.
                                        # Blockwise decisions may require entire query blocks.

    def metric_eval_set(self, reference):
        """Return a surrogate evaluation set (Q,u) for dim>3, or None.

        Extended problems may implement this instead of a rectangular grid.
        MetricGrid raises an error if the required set is unavailable."""
        return None

    def metric_voltage_mse(self, surrogate, reference):
        """Optional engineering-observable MSE, evaluated once per snapshot.

        An extension may compare an observable with reference data. This diagnostic
        is excluded from FE. Return None when no such metric is defined."""
        return None

    def operator_axis_roles(self):
        """Return PI-DeepONet axis roles: (trunk coordinates, branch parameters).

        By default the first two axes are coordinates (x,t) and the rest
        are parameters. Extensions can override these axis roles."""
        dim = int(np.asarray(self.query_bounds).shape[0])
        return (tuple(range(2)), tuple(range(2, dim)))

    def normalize_queries(self, Q: np.ndarray) -> np.ndarray:
        """Normalize oracle/reference queries and evaluation grids.

        Examples include periodic mapping and clipping to dataset bounds.
        The default is identity; optimizer fitness queries bypass this hook."""
        return np.asarray(Q, dtype=float)

    # Query sampling for policy top-up; default is uniform by axis.
    def sample_query_candidates(self, n: int, rng) -> np.ndarray:
        """Sample n query candidates, filtered later by is_valid_query.

        Consume one rng.uniform(lo,hi,size=n) call per axis in order.
        Problems requiring boundary margins override this method."""
        qb = np.asarray(self.query_bounds, dtype=float)
        cols = [rng.uniform(qb[j, 0], qb[j, 1], size=int(n))
                for j in range(qb.shape[0])]
        return np.column_stack(cols)

    def sample_interior_queries(self, n: int, seed: int):
        """Return n interior residual points, or None for surrogate-default sampling.

        Problems with restricted domains or margins return (n,dim_query),
        for example the F07 perforated domain."""
        return None

    def query_to_decision(self, queries: np.ndarray) -> np.ndarray:
        """Map a sampled query back to a decision; identity by default.

        Problems with extra decision coordinates, such as F02's z, must
        override this method when sampling-policy top-up is supported."""
        return np.asarray(queries, dtype=float)

    # Evaluation axes for 3D parameterized problems; 2D uses framework defaults.
    def metric_axes_3d(self, reference, nx: int, nt: int, nmu: int):
        """Return three float32 coordinate arrays for a 3D evaluation grid.

        Bounds and endpoint conventions are problem-specific: F09 uses
        problem bounds with endpoint=False in x; F10 uses data bounds
        with all endpoints included."""
        raise NotImplementedError("3D problems must implement metric_axes_3d")

    def evaluate_fitness_for_pool(self, decision: np.ndarray, state_provider) -> float:
        """Evaluate a pool objective; defaults to evaluate_fitness.

        Override when pool and optimizer arithmetic differ, such as
        F10 rounding J to float32 before objective evaluation."""
        return self.evaluate_fitness(decision, state_provider)

    # Reference dataset attachment.
    def attach_data(self, data_path: str) -> None:
        """Attach data-dependent fields, such as the F07 source; no operation by default."""

    def load_reference(self, data_path: str):
        """Load reference data; override for special layouts or filling rules such as F07."""
        from evaluation.reference import ReferenceDataset
        return ReferenceDataset(data_path)

    # Reduced objective and constraints.
    @abstractmethod
    def evaluate_fitness(self, decision: np.ndarray, state_provider: StateProvider) -> float:
        """Evaluate the objective using state_provider to query u.

        Pointwise formulas and internal numerical quadrature share this interface."""

    @abstractmethod
    def violation(self, decision: np.ndarray, state_provider: StateProvider) -> float:
        """Aggregate constraint violation: zero if feasible, positive otherwise."""

    def constraint_components(self, decision: np.ndarray, state_provider) -> list:
        """Return individual constraints for Ji SAEA surrogate channels.

        Each dictionary contains kind (g or h), value, and tol. Inequalities
        require g<=tol; equalities require abs(h)<=tol. The default returns
        aggregate violation as one inequality with zero tolerance.
        Overrides must use the same component definitions as violation()."""
        return [{"kind": "g",
                 "value": self.violation(decision, state_provider),
                 "tol": 0.0}]

    def constraint_meta(self) -> list:
        """Return constraint channel metadata [{kind,tol}, ...] without values.

        The default matches the aggregate fallback in constraint_components.
        Any component override must provide matching metadata here."""
        return [{"kind": "g", "tol": 0.0}]
