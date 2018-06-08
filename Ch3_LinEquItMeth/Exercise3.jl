#Exercise 3
#a)
for n in [5,10,20,50]
    A = eye(n,n)*2 + diagm(ones(n-1),1) + diagm(ones(n-1),-1)
    b = zeros(n,1)
    x0 = ones(n,1)
    println("GaussJacobi : n = $n")
    x = GaussJacobi(A,b,x0,1e-10,100)
    println(x)
end
for n in [5,10,20,50]
    A = eye(n,n)*2 + diagm(ones(n-1),1) + diagm(ones(n-1),-1)
    b = zeros(n,1)
    x0 = ones(n,1)
    println("GaussSeidel : n = $n")
    GaussSeidel(A,b,x0)
end
for n in [5,10,20,50]
    A = eye(n,n)*2 + diagm(ones(n-1),1) + diagm(ones(n-1),-1)
    b = zeros(n,1)
    x0 = ones(n,1)
    println("SuccessiveOverrelaxation : n = $n")
    for l in 1:20
        println("ω = $(0.1*l)")
        SuccessiveOverrelaxation(A,b,x0,0.1*l)
    end
end

#b)

function GaussJacobi(A,b,x0,tol=1e-10,iter=1e5)

    n = size(A,1)

    dist = 1e6
    oldx = x0
    k = 1
    while (dist > tol) & (k < iter)
        println("$k  $dist")
        newx = zeros(n,1)
        newx[1,1] = (b[1,1] - sum(A[1,2:end].*oldx[2:end,1]))/A[1,1]
        for i in 2:n-1
            newx[i,1] = (b[i,1] - sum(A[i,1:i-1].*oldx[1:i-1,1]) - sum(A[i,i+1:end].*oldx[i+1:end,1]))/A[i,i]
        end
        newx[n,1] = (b[n,1] - sum(A[n,1:(n-1)].*oldx[1:(n-1),1]))/A[n,n]

        dist = maximum(abs.(newx))
        oldx = newx
        k += 1
    end

    return (oldx,k)
end

function GaussSeidel(A,b,x0,tol=1e-10,iter=25)

    n = size(A,1)

    dist = 1e6
    oldx = x0
    k = 1
    while (dist > tol) & (k < iter)
        println("$k  $dist")
        newx = zeros(n,1)
        newx[1,1] = (b[1,1] - sum(A[1,2:end].*oldx[2:end,1]))/A[1,1]
        for i in 2:n-1
            newx[i,1] = (b[i,1] - sum(A[i,1:i-1].*newx[1:i-1,1]) - sum(A[i,i+1:end].*oldx[i+1:end,1]))/A[i,i]
        end
        newx[n,1] = (b[n,1] - sum(A[n,1:(n-1)].*newx[1:(n-1),1]))/A[n,n]

        dist = maximum(abs.(newx))
        oldx = newx
        k += 1
    end

    return (oldx,k)
end

function SuccessiveOverrelaxation(A,b,x0,omega=1.1,tol=1e-10,iter=1e5)

    n = size(A,1)

    dist = 1e6
    oldx = x0
    k = 1
    while (dist > tol) & (k < iter)
        println("$k  $dist")
        newx = zeros(n,1)
        newx[1,1] = omega*(b[1,1] - sum(A[1,2:end].*oldx[2:end,1]))/A[1,1] + (1-omega)*oldx[1,1]
        for i in 2:n-1
            newx[i,1] = omega*(b[i,1] - sum(A[i,1:i-1].*newx[1:i-1,1]) - sum(A[i,i+1:end].*oldx[i+1:end,1]))/A[i,i] + (1-omega)*oldx[i,1]
        end
        newx[n,1] = omega*(b[n,1] - sum(A[n,1:(n-1)].*newx[1:(n-1),1]))/A[n,n] + (1-omega)*oldx[n,1]

        dist = maximum(abs.(newx))
        oldx = newx
        k += 1
    end

    return (oldx,k)
end

ResultsJacobiSeidel = Array{Any,3}(2,4,3)

for (i,e) in enumerate([1e-2,1e-3,1e-4])
    for (j,n) in enumerate([5,10,20,50])
        A = eye(n,n)*2 + diagm(ones(n-1),1) + diagm(ones(n-1),-1)
        b = zeros(n,1)
        x0 = ones(n,1)
        _,iter = GaussJacobi(A,b,x0,e,1e5)
        ResultsJacobiSeidel[1,j,i] = iter
        _,iter = GaussSeidel(A,b,x0,e,1e5)
        ResultsJacobiSeidel[2,j,i] = iter
    end
end

println("Jacobi")
println(ResultsJacobiSeidel[1,:,:])
println("Seidel")
println(ResultsJacobiSeidel[2,:,:])

#c)

using PyPlot

n = 50
D = eye(n,n)*2
L = diagm(ones(n-1),-1)
U = diagm(ones(n-1),1)

M(ω) = D + ω*L
N(ω) = (1-ω)*D - ω*U

rho(ω) = maximum(abs.(eigvals(M(ω)\N(ω))))

gridω = linspace(1.0,2.1,10000)
resω = zeros(10000)
for (k,ω) in enumerate(gridω)
    resω[k] = rho(ω)
end

figure()
plot(gridω,resω)

#d)

A = D+L+U
b = zeros(n,1)
x0 = ones(n,1)
optimalω = gridω[indmin(resω )]
x1,_ = SuccessiveOverrelaxation(A,b,x0, 1., 1e-4, 1e5)
#846 iterations are necessary
x2,_ = SuccessiveOverrelaxation(A,b,x0, optimalω, 1e-4, 1e5)
#101 iterations are necessary, eight times fewer !
