import cvxpy as cp
import pandas as pd
from constraints.constraints import beta_neutral_constraints

def test_beta_neutral_constraints_uses_tol():
    w = cp.Variable(2)
    betas = pd.Series([1.0, -1.0])
    cons = beta_neutral_constraints(w, betas, tol=0.5)
    assert len(cons) == 2
    assert all(isinstance(c, cp.constraints.nonpos.Inequality) for c in cons)

def test_beta_neutral_constraints_zero_tol_equality():
    w = cp.Variable(2)
    betas = pd.Series([0.5, 0.5])
    cons = beta_neutral_constraints(w, betas, tol=0)
    assert len(cons) == 1
    assert isinstance(cons[0], cp.constraints.zero.Equality)
