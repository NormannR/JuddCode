
function LU(A)

    Ak = A
    n = size(A,1)
    Lk = eye(n,n)
    for k in 1:(n-1)
        l = zeros(n,1)
        for i in (k+1):n
            l[i,1] = Ak[i,k]/Ak[k,k]
        end
        Cleaner = zeros(n,n)
        Cleaner[:,k] = l
        Temp = (eye(n,n) - Cleaner)
        Lk = Temp*Lk
        Ak = Temp*Ak
    end

    U = Ak
    L = InvTriLow(Lk)

    return (L,U)

end

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

function InvTriLow(A)
    n = size(A,1)
    invA = zeros(n,n)
    for j in 1:n
        C = zeros(n,1)
        C[j,1] = 1/A[j,j]
        for k in (j+1):n
            C[k,1] = -sum(A[k,1:k-1].*C[1:k-1,1])/A[k,k]
        end
        invA[:,j] = C
    end
    return invA
end

function LUSolve(A,b)
    L,U = LU(A)
    z = LowTriSolve(L,b)
    x = UpTriSolve(U,z)
    return x
end
