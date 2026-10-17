# Six economic response features in two coordinates

A constructed encoder and decoder recover six distinct response features through two coordinates. Recovery is exact for zero input and each single active feature. The circuit uses fixed hexagon directions. It uses no training.

Each feature is a consumption response from a separate log-utility growth economy. Each economy has Cobb–Douglas production, alpha=0.36, beta=0.96, and full capital depreciation. The economies have no transfers. Equal log weights do not introduce interactions.

The exact policies are c=(1-alpha beta)zk^alpha and next capital=alpha beta zk^alpha. Their sum equals production. The Euler equation is 1/c=beta alpha z_next (k_next)^(alpha-1)/c_next. Substitution reduces its right side to alpha beta/[(1-alpha beta)k_next], which equals its left side.

At fixed capital one, baseline consumption is c0=0.6544. Productivity is either one or two. The normalized response is x=(c-c0)/c0=z-1. Each input is binary.

## Exact construction

For i=0,...,5, set v_i=(cos(i pi/3), sin(i pi/3)). Let E have these six columns. Set D=2 E-transpose and b=-ones(6). The response decoder is ReLU(D E x+b).

The encoder has rank two. Its six columns give six distinct feature directions. Each column has length one. Distinct column inner products belong to {1/2,-1/2,-1}.

For zero input, every decoder preactivation is minus one. Thus the output is zero. For singleton input x=e_i, output coordinate j is max(0,2 dot(v_j,v_i)-1). At j=i, this expression is one. At every other coordinate, it is zero.

Thus all seven declared support inputs are recovered exactly. Six known feature directions share two coordinates. The nonlinear decoder selects the active feature on this support.

## Pair interference

All 15 pairs fail exact recovery. The exact output follows from the integer matrix 2 E-transpose E. Its circular row entries are (2,1,-1,-2,-1,1).

- Six adjacent pairs produce value two at both active coordinates. Other outputs are zero.
- Six pairs separated by 120 degrees produce value one only at the intermediate coordinate.
- Three opposite pairs produce zero at every coordinate.

The JSON enumerates all 64 binary inputs. Exactly seven inputs recover correctly: zero and the six singletons. A six-dimensional identity control recovers all 64 inputs. The economic teacher remains additive for every input. Pair errors arise from this response code, not economic interaction.

## Scope

This is an existence proof for a designed response representation. It is independent of the failed learned runs. It does not show that training finds this circuit. It does not solve an economic policy or establish a hardness theorem.

Exact recovery is limited to binary singleton amplitudes. For input t e_i, the active output is max(0,2t-1). Neighbor outputs are max(0,t-1). Thus the proof does not extend to arbitrary positive amplitudes.

The independent verifier uses Python integer arithmetic for the full binary truth table. It also checks the saved two-dimensional matrices within tolerance 1e-12. It verifies the teacher resource constraint, Euler equation, response normalization, and additivity. The support floating-point error is below 7e-16.
