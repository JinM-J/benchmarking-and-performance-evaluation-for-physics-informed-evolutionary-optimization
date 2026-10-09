"""Finite-difference PDE residuals on a shared physical grid using frozen predictions."""
from collections import Counter
import numpy as np
import torch

# Centered differences are O(h**2); the fourth derivative uses five points.
STENCILS={1:((-1,1),(-.5,.5)),2:((-1,0,1),(1.,-2.,1.)),4:((-2,-1,0,1,2),(1.,-4.,6.,-4.,1.))}
PARAMETER_FRACTIONS=(.25,.5,.75)
SUPPORT_GRID=32


def predict_finite(model, q, batch_size=4096):
    q=np.asarray(q,dtype=float)
    values=np.concatenate([np.asarray(model.predict(q[i:i+batch_size]),dtype=float).reshape(-1)
                           for i in range(0,len(q),batch_size)])
    if values.shape!=(len(q),) or not np.isfinite(values).all():
        raise FloatingPointError('Nonfinite predictions at shared evaluation points; no points removed or zero-filled')
    return values


def grid_queries(problem, n):
    """Cell centers on the first two axes with fixed parameter slices, independent of training samples."""
    bounds=np.asarray(problem.query_bounds,dtype=float)
    axes=[lo+(hi-lo)*(np.arange(n)+.5)/n for lo,hi in bounds[:2]]
    if len(bounds)==3:
        lo,hi=bounds[2];axes.append(lo+(hi-lo)*np.array(PARAMETER_FRACTIONS))
    shape=tuple(len(a) for a in axes)
    q=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1)
    return q,shape,(bounds[:2,1]-bounds[:2,0])/n


def centered_derivatives(field, names, required, spacing):
    """Retain a two-cell stencil margin; never slice or differentiate parameter axes."""
    core=(slice(2,-2),slice(2,-2))+(slice(None),)*(field.ndim-2)
    derivatives={}
    for key in sorted(required):
        counts=Counter(key)
        if len(counts)!=1:raise NotImplementedError(f'Mixed-derivative stencil is undefined: {key}')
        name,order=next(iter(counts.items()));axis=names.index(name)
        if axis>=2 or order not in STENCILS:raise NotImplementedError(key)
        shifts,weights=STENCILS[order]
        out=np.zeros_like(field[core],dtype=np.float64)
        for shift,weight in zip(shifts,weights):
            section=list(core);section[axis]=slice(2+shift,field.shape[axis]-2+shift)
            out+=weight*field[tuple(section)]
        derivatives[key]=out/spacing[axis]**order
    return field[core],derivatives,core


def interior_mask(problem,q,spacing):
    """Method-independent mask with a 2*max(h) stencil clearance from hole boundaries."""
    shape=q.shape[:-1];points=q.reshape(-1,q.shape[-1])
    mask=np.asarray(problem.is_valid_query(points),dtype=bool)
    # Keep the same physical support for 32/64/128 grid comparisons.
    bounds=np.asarray(problem.query_bounds,dtype=float)
    support_spacing=(bounds[:2,1]-bounds[:2,0])/SUPPORT_GRID
    for axis in range(2):
        margin=2.5*support_spacing[axis]
        mask &= (points[:,axis]>=bounds[axis,0]+margin-1e-12)&(points[:,axis]<=bounds[axis,1]-margin+1e-12)
    if hasattr(problem,'holes'):
        margin=2*max(support_spacing)
        for x,y,r in problem.holes:
            mask &= (points[:,0]-x)**2+(points[:,1]-y)**2>(r+margin)**2
        source=problem.physics._f_interp(points)
        if not np.isfinite(source[mask]).all():raise FloatingPointError('Nonfinite source values in the shared Poisson evaluation domain')
    # Exclude stencils crossing the F04 source discontinuity.
    if problem.name=='F04':
        lo,hi=problem.query_bounds[0]
        for fraction in (.25,.5,.75):mask &= np.abs(points[:,0]-(lo+fraction*(hi-lo)))>2*support_spacing[0]
    return mask.reshape(shape)


def rmse(residual):
    values=np.asarray(residual,dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all():raise FloatingPointError('Residual values are empty or nonfinite')
    result=float(np.sqrt(np.mean(values*values,dtype=np.float64)))
    if not np.isfinite(result):raise FloatingPointError('Mean squared residual overflowed')
    return result


def pde_metrics(problem,model,n):
    q,shape,spacing=grid_queries(problem,n)
    valid=np.asarray(problem.is_valid_query(q.reshape(-1,q.shape[-1])),dtype=bool)
    # Only predict valid nodes outside holes; share the mask across methods.
    field=np.full(np.prod(shape),np.nan,dtype=float)
    field[valid]=predict_finite(model,q.reshape(-1,q.shape[-1])[valid])
    field=field.reshape(shape)
    u,derivatives,core=centered_derivatives(field,problem.physics.coordinate_names,
                                           problem.physics.required_derivatives(),spacing)
    points=q[core];mask=interior_mask(problem,points,spacing)
    qt=points[mask];ut=u[mask]
    dt={key:values[mask] for key,values in derivatives.items()}
    if not np.isfinite(ut).all() or any(not np.isfinite(v).all() for v in dt.values()):
        raise FloatingPointError('Shared finite-difference stencil contains nonfinite model values')
    residual=problem.physics.residual(torch.as_tensor(qt,dtype=torch.float64),
          torch.as_tensor(ut[:,None],dtype=torch.float64),
          {key:torch.as_tensor(v[:,None],dtype=torch.float64) for key,v in dt.items()})
    residual=np.asarray(residual.detach().cpu(),dtype=float).reshape(-1)
    metrics=dict(pde_rmse=rmse(residual),pde_points=len(qt),grid_size=n)
    if qt.shape[1]==3:
        metrics['parameter_slices']=[dict(parameter=float(value),n=int((qt[:,2]==value).sum()),
             rmse=rmse(residual[qt[:,2]==value])) for value in np.unique(qt[:,2])]
    return metrics


def condition_points(problem,region,n):
    """Sample declared faces on cell-center lines and circles by their fixed parameterization."""
    bounds=np.asarray(problem.query_bounds,dtype=float)
    probes=region.sample(17)
    for axis in range(min(2,len(bounds))):
        if np.ptp(probes[:,axis])<1e-12:
            fixed=float(probes[0,axis]);axes=[]
            for j,(lo,hi) in enumerate(bounds):
                if j==axis:axes.append(np.array([fixed]))
                elif j>=2:axes.append(lo+(hi-lo)*np.array(PARAMETER_FRACTIONS))
                else:axes.append(lo+(hi-lo)*(np.arange(n)+.5)/n)
            q=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,len(bounds))
            if region.contains(q).all():return q
    q=region.sample(max(4*n,256))
    if not region.contains(q).all():raise ValueError('Boundary samples fall outside the declared region')
    return q


def higher_condition_value(problem, model, q, derivative, n):
    """Second-order stencils for repeated second/third boundary derivatives."""
    order = len(derivative)
    if len(set(derivative)) != 1 or order not in (2, 3):
        raise NotImplementedError(f'Unsupported condition derivative: {derivative}')
    if n < order + 2:
        raise ValueError('Condition grid is too small for the boundary stencil')
    axis = problem.physics.coordinate_names.index(derivative[0])
    lo, hi = problem.query_bounds[axis]
    h = (hi - lo) / n
    if order == 2:
        radius, centered = 1, ((-1, 0, 1), (1., -2., 1.))
        forward = ((0, 1, 2, 3), (2., -5., 4., -1.))
    else:
        radius, centered = 2, ((-2, -1, 0, 1, 2), (-.5, 1., 0., -1., .5))
        forward = ((0, 1, 2, 3, 4), (-2.5, 9., -12., 7., -1.5))
    left = q[:, axis] - radius * h < lo
    right = q[:, axis] + radius * h > hi
    middle = ~(left | right)
    backward = (tuple(-shift for shift in forward[0]),
                tuple((-1)**order * weight for weight in forward[1]))
    out = np.empty(len(q))
    for mask, (offsets, weights) in ((left, forward), (right, backward), (middle, centered)):
        if not mask.any():
            continue
        value = np.zeros(mask.sum())
        for shift, weight in zip(offsets, weights):
            points = q[mask].copy()
            points[:, axis] += shift * h
            if (points[:, axis] < lo - 1e-12).any() or (points[:, axis] > hi + 1e-12).any():
                raise ValueError('Derivative stencil exceeds domain bounds')
            value += weight * predict_finite(model, points)
        out[mask] = value / h**order
    return out


def condition_value(problem,model,q,derivative,n):
    if not derivative:return predict_finite(model,q)
    if len(derivative)!=1:return higher_condition_value(problem,model,q,derivative,n)
    axis=problem.physics.coordinate_names.index(derivative[0]);lo,hi=problem.query_bounds[axis]
    h=(hi-lo)/n
    # Use centered interior and second-order one-sided boundary differences.
    out=np.empty(len(q));left=q[:,axis]-h<lo;right=q[:,axis]+h>hi;middle=~(left|right)
    for mask,offsets,weights in [(left,(0,1,2),(-1.5,2.,-.5)),(right,(0,-1,-2),(1.5,-2.,.5)),(middle,(-1,1),(-.5,.5))]:
        if not mask.any():continue
        value=np.zeros(mask.sum())
        for shift,weight in zip(offsets,weights):
            points=q[mask].copy();points[:,axis]+=shift*h
            if (points[:,axis]<lo-1e-12).any() or (points[:,axis]>hi+1e-12).any():raise ValueError('Derivative stencil exceeds domain bounds')
            value+=weight*predict_finite(model,points)
        out[mask]=value/h
    return out


def condition_metrics(problem,model,n):
    """Report value and derivative errors separately, without training-loss weights."""
    components=[];groups={}
    def add(name,group,q,error):
        error=np.asarray(error).reshape(-1)
        components.append(dict(name=name,quantity=group,points=len(q),rmse=rmse(error)))
        groups.setdefault(group,[]).append(error)
    for i,condition in enumerate(problem.physics.initial_conditions()):
        q=condition_points(problem,condition.region,n)
        values=condition_value(problem,model,q,condition.derivative,n)
        group='ic_u' if not condition.derivative else 'ic_d'+''.join(condition.derivative)
        add(f'ic_{i}',group,q,values-np.asarray(condition.target(q)).reshape(-1))
    for i,condition in enumerate(problem.physics.boundary_conditions()):
        if condition.kind=='dirichlet':
            q=condition_points(problem,condition.region,n)
            add(f'bc_{i}','bc_u',q,predict_finite(model,q)-np.asarray(condition.target(q)).reshape(-1))
        elif condition.kind in ('periodic','periodic_derivative'):
            qa=condition_points(problem,condition.region_pair[0],n)
            qb=condition_points(problem,condition.region_pair[1],n)
            assert qa.shape==qb.shape
            different=np.any(np.abs(qa-qb)>1e-10,axis=0)
            assert different.sum()==1,'Free coordinates on periodic boundaries are not paired pointwise'
            if getattr(condition,'include_value',True):
                add(f'bc_{i}_value','bc_u',qa,predict_finite(model,qa)-predict_finite(model,qb))
            if condition.kind=='periodic_derivative':
                group='bc_d'+''.join(condition.derivative)
                add(f'bc_{i}_derivative',group,qa,condition_value(problem,model,qa,condition.derivative,n)-condition_value(problem,model,qb,condition.derivative,n))
        else:raise NotImplementedError(condition.kind)
    metrics={key+'_rmse':rmse(np.concatenate(values)) for key,values in groups.items()}
    return dict(**metrics,condition_components=components)
