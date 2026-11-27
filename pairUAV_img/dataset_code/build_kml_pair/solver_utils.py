"""Validate binary MILP incumbents independently of solver termination status."""
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint

def valid_binary(x, matrix, lower, upper):
    if x is None or not np.all(np.isfinite(x)):
        return False
    rounded = np.rint(x)
    if np.max(np.abs(x-rounded), initial=0) > 1e-5 or np.any(rounded<0) or np.any(rounded>1):
        return False
    values = matrix @ rounded
    return bool(np.all(values >= np.asarray(lower)-1e-7) and np.all(values <= np.asarray(upper)+1e-7))

def solve_feasibility(matrix, lower, upper, time_limit):
    n=matrix.shape[1]
    attempts=[]
    for seconds in (time_limit, time_limit*3):
        result=milp(np.zeros(n), integrality=np.ones(n), bounds=Bounds(0,1),
                    constraints=LinearConstraint(matrix,lower,upper),
                    options={'time_limit':seconds,'mip_rel_gap':0.0,'presolve':True})
        attempts.append({'seconds':seconds,'status':int(result.status),'message':result.message})
        if valid_binary(result.x,matrix,lower,upper):
            result.x=np.rint(result.x)
            return result,True,attempts
        if result.status != 1:
            break
    return result,False,attempts
