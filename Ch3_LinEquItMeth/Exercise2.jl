#Exercise 2
#a)

function LowTriSolve(A,b)
    n = size(A,1)
    x = zeros(n,1)
    x[1] = b[1]/A[1,1]
    for k in 2:n
        x[k] = (b[k] - sum(A[k,1:k-1].*x[1:k-1]))/A[k,k]
    end
    return x
end

function UpTriSolve(A,b)
    n = size(A,1)
    x = zeros(n,1)
    x[n] = b[n]/A[n,n]
    for k in (n-1):-1:1
        x[k] = (b[k] - sum(A[k,k+1:end].*x[k+1:end]))/A[k,k]
    end
    return x
end

function LUSolve(A,b)
    L,U = LU(A)
    z = LowTriSolve(L,b)
    x = UpTriSolve(U,z)
    return x
end

b = ones(4,1)

x = LUSolve(A,b)

#b)

function CholevskySolve(A,b)
    L = Cholevsky(A)
    z = LowTriSolve(L,b)
    x = UpTriSolve(L',z)
    return x
end

x = CholevskySolve(A,b)

#c) Gauss-Jacobi

function GaussJacobi(A,b,x0,tol=1e-10,iter=40)

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

        dist = maximum(abs.(A*newx-b))
        oldx = newx
        k += 1
    end

    return oldx
end

x = GaussJacobi(A,b,[0.05,0.05,0.05,0.])

#d) Gauss-Seidel

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

        dist = maximum(abs.(A*newx-b))
        oldx = newx
        k += 1
    end

    return oldx
end

x = GaussSeidel(A,b,[0.05,0.05,0.05,0.])

#e) Successive Overrelaxation

function SuccessiveOverrelaxation(A,b,x0,omega=1.1,tol=1e-10,iter=25)

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

        dist = maximum(abs.(A*newx-b))
        oldx = newx
        k += 1
    end

    return oldx
end

x = SuccessiveOverrelaxation(A,b,[0.05,0.05,0.05,0.])

#h) Cramer Rule

function Cramer(A,b)

    n = size(A,1)
    x = zeros(n,1)
    D = det(A)
    for k in 1:n
        Ak = copy(A)
        Ak[:,k] = b
        x[k,1] = det(Ak)/D
    end

    return x
end

x = Cramer(A,b)
