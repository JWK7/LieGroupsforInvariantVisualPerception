using LinearAlgebra 
using Plots
using Symbolics

@variables x

### utility ###
NumToFloat(A::Array{Num, 2}) = (i->A[i].val).(CartesianIndices(size(A)))

### filters ###

q(x, N) = 1/N*(1 + 2*sum( cos.( 2π*x*(1:(Int(N/2) - 1))/N ) ) )

Q(x, N) = ( z -> q(z.I[1] + x - z.I[2], N) ).( CartesianIndices( ( 0:(N-1), 0:(N-1) ) ) )

G(N) = substitute.(expand_derivatives.(Differential(x).(Q(x, N))), x=>0.0)

### "main" ###
N = 20
LieOp = NumToFloat(G(N))

### plots ###
aspect_ratio=(length(LieOp[N/2 |> Int, :])/(maximum(LieOp[N/2 |> Int, :]) - minimum(LieOp[N/2 |> Int, :])))
p1 = heatmap(LieOp', c=:Greys_3, yflip = true, aspect_ratio=:equal) 
p2 = plot(LieOp[N/2 |> Int, :], aspect_ratio=aspect_ratio, xlim=[1, N], label="Lie Op #N/2")
p3 = plot(real.(eigvals(LieOp, sortby=(z->abs(z)))), label="Real Eig of LieOp")
p4 = plot(imag.(eigvals(LieOp, sortby=(z->-abs(z)))), label="Imag Eig of LieOp")
plot(p1,p2, p3, p4, size = (800, 800), layout  = @layout [a b; c d])
