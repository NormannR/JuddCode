#Exercise 1

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

#a)

A = [54. 14 -11 2 ; 14 50 -4 29 ; -11 -4 55 22 ; 2 29 22 95]
L,U = LU(A)

#b)

B = [54. -9 18 9 ; 6 23 -2 9 ; -12 14 34 -3 ; -6 13 20 49]
L,U = LU(B)

#c)

function Cholevsky(A)
    n = size(A,1)
    L = zeros(n,n)

    #Initialization

    L[1,1] = sqrt(A[1,1])
    for i in 2:n
        L[i,1] = A[i,1]/L[1,1]
    end

    #Loop for j in 2,...,n-1

    for j in 2:(n-1)
        L[j,j] = sqrt(A[j,j] - sum(L[j,1:j-1].^2))
        for i in (j+1):n
            L[i,j] = (A[i,j] - sum(L[j,1:(j-1)].*L[i,1:(j-1)]))/L[j,j]
        end
    end

    #End with L(n,n)
    L[n,n] = sqrt(A[n,n] - sum(L[n,1:(n-1)].^2))

    return L
end

L = Cholevsky(A)
