using LinearAlgebra 
using Plots
using Symbolics
using NPZ

@variables x
@variables x1,x2
@variables theta

### utility ###
NumToFloat(A::Array{Num, 3}) = (i->A[i].val).(CartesianIndices(size(A)))

### filters ###

q(x, N) = 1/N*(1 + 2*sum( cos.( 2π*x*(1:(Int(N/2) - 1))/N ) ) )

# multiply(q1,q2) = reshape(q1,(length(q1),1)).*reshape(q2,(1,length(q2)))


# function q(x1,x2,N1,N2)
#     print("hi")
#     a = q(x1,N1)
#     b = q(x2,N2)
#     # c = reshape(a,:UnitRange{Int64}, 10, 1, 1)
#     print(a)
#     return multiply(q(x1,N1),q(x2,N2))
# end
# q(x1,x2,N1,N2) = (q(x1,N1)*q(x2,N2))

(z-> q(z.I[1][:]+x-z.I[2][:],N1)*q(z.I[:][1]+x*ratio-z.I[:][2],N2)).( CartesianIndices( ( 0:(N1-1), 0:(N1-1) , 0:(N2-1) ) ) )

Q(x, N) = ( z -> q(z.I[1] + x - z.I[2], N) ).( CartesianIndices( ( 0:(N-1), 0:(N-1) ) ) )

rotQ(theta,N1,N2) = ( z -> q(z.I[1] + cos(theta) - z.I[2],z.I[1] + sin(theta) - z.I[2], N1,N2) ).( CartesianIndices( ( 0:(N1-1), 0:(N1-1) , 0:(N2-1) ) ) )

Q(x , N1,N2,ratio) = ( z -> q(z.I[1][:] + x - z.I[2][:],   z.I[:][1] + ratio*x - z.I[:][2]   , N1,N2) ).( CartesianIndices( ( 0:(N1-1), 0:(N1-1) , 0:(N2-1) ) ) )
# Q(x1,x2,N1,N2) = ( z -> q(z.I[1] + x - z.I[2], N) ).( CartesianIndices( ( 0:(N-1), 0:(N-1) ) ) )

G(N) = substitute.(expand_derivatives.(Differential(x).(Q(x, N))), x=>0.0)

G(N1,N2,ratio1,ratio2) = substitute.(expand_derivatives.((Differential(x1)).(Q(x ,N1,N2,ratio1,ratio2))), x=>(0.0))

rotG(N1,N2) = substitute.(expand_derivatives.(Differential(theta).(rotQ(theta,N1,N2))), theta=>0.0)


# G(N1,N2)
### "main" ###
N = 20
LieOp = NumToFloat(G(20,20,5,1))
npzwrite("2DTranslationG.npz",LieOp)
# print(size(LieOp))

### plots ###
# aspect_ratio=(length(LieOp[N/2 |> Int, :])/(maximum(LieOp[N/2 |> Int, :]) - minimum(LieOp[N/2 |> Int, :])))
# p1 = heatmap(LieOp', c=:Greys_3, yflip = true, aspect_ratio=:equal) 
# p2 = plot(LieOp[N/2 |> Int, :], aspect_ratio=aspect_ratio, xlim=[1, N], label="Lie Op #N/2")
# p3 = plot(real.(eigvals(LieOp, sortby=(z->abs(z)))), label="Real Eig of LieOp")
# p4 = plot(imag.(eigvals(LieOp, sortby=(z->-abs(z)))), label="Imag Eig of LieOp")
# plot(p1,p2, p3, p4, size = (800, 800), layout  = @layout [a b; c d])
