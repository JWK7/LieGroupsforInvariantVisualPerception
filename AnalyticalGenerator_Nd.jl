using LinearAlgebra 
using Plots
using Symbolics

@variables x

### utility ###
NumToFloat(A::Array{Num, 2}) = (i->A[i].val).(CartesianIndices(size(A)))
NumToFloat(A::Array{Num, 4}) = (i->A[i].val).(CartesianIndices(size(A)))
NumToFloat(A::Array{Num, 6}) = (i->A[i].val).(CartesianIndices(size(A)))

### filters ###
q(x, N) = 1/N*(1 + 2*sum( cos.( 2π*x*(1:(Int(N/2) - 1))/N ) ) )

Grid(Dim, nDim) = CartesianIndices( ntuple( i->(0:(Dim-1)), nDim) )  

### 1D ###
Q₁(x, N, v) = ( z -> q(z.I[1] + x*v[1] - z.I[2], N) ).( Grid(N, 2*length(v)) ) 
G₁(N, v) = substitute.(expand_derivatives.(Differential(x).(Q₁(x, N, v))), x=>0.0)

### 2D ###
### v - direction vector, e.g., v = [1, 1] for diagonal translation ###
                   ### 1st out dim    1st in dim   2nd out dim      2nd in dim	 ###
Q₂(x, N, v) = ( z -> q(z.I[1] + x*v[1] - z.I[3], N)*q(z.I[2] + x*v[2] - z.I[4], N) ).( Grid(N, 2*length(v)) ) 
### dimensionality of 2D transformation G is 4 ###
# d₁ - output 1st spacial dim
# d₂ - output 2nd spacial dim
# d₃ - input 1st spacial dim
# d₄ - input 2nd spacial dim
G₂(N, v) = substitute.(expand_derivatives.(Differential(x).(Q₂(x, N, v))), x=>0.0)

### 3D ###
Q₃(x, N, v) = ( z -> q(z.I[1] + x*v[1] - z.I[4], N)*q(z.I[2] + x*v[2] - z.I[5], N)*q(z.I[3] + x*v[3] - z.I[6], N) ).( Grid(N, 2*length(v)) ) 
G₃(N, v) = substitute.(expand_derivatives.(Differential(x).(Q₃(x, N, v))), x=>0.0)


### dims of input  ###
# I₀ = [1, 1, N, N]
### dims of output, [N, N, 1, 1]  ###
# Iₓ = sum(G .* I₀, dims=(3, 4))

### "main" ###
N = 10

### 1D case ###
# v = [1]
# LieOp = NumToFloat(G₁(N, v))
# size(LieOp) |> display
# LieOp |> display


### 2D case ###
v = [1, 1]
LieOp = NumToFloat(G₂(N, v))
size(LieOp) |> display
LieOp |> display

print(size(LieOp))

### 3D case ###
# v = [1, 1, 1]
# LieOp = NumToFloat(G₃(N, v))
# size(LieOp) |> display
# LieOp |> display

using DelimitedFiles
writedlm( "LieOpOpt_20_zvl_2D.csv",  LieOp, ',')
