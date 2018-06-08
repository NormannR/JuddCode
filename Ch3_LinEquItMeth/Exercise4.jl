
#Exercise 4

using Distributions

function MarkovApproxCenter(rho,n)
    #Maximum value that can be attained by x if x0 = 0 is 1/(1-rho)
    a = -1/(1-rho)
    b = 1/(1-rho)
    PI = zeros(n,n)
    h = (b-a)/n
    for i in 1:n
        xi = (a + (i-1/2)*h)
        F(x) = cdf(Uniform(-1+rho*xi,1+rho*xi),x)
        for j in 1:n
            PI[i,j] = F(a+j*h) - F(a+(j-1)*h)
        end
    end
    return PI
end

function MarkovApproxAverage(rho,n)
    #Maximum value that can be attained by x if x0 = 0 is 1/(1-rho)
    a = -1/(1-rho)
    b = 1/(1-rho)
    PI = zeros(n,n)
    h = (b-a)/n
    for i in 1:n
        for j in 1:n
            u = a+(j-1)*h
            v = a+j*h
            w(y) = -1+rho*y
            x(y) = 1+rho*y
            f(y) = max(min(x(y),a+j*h) - max(w(y),a+(j-1)*h) ,0.)
            PI[i,j],_ = quadgk(f,a+(i-1)*h,a+i*h)
            PI[i,j] = PI[i,j]/(2*h)
        end
    end
    return PI
end

PI1 = MarkovApproxCenter(0.8,5)
PI2 = MarkovApproxAverage(0.8,5)
PI1 = MarkovApproxCenter(0.8,10)
PI2 = MarkovApproxAverage(0.5,10)

#Ergodic distribution

n = 5
rho = 0.8
PI = MarkovApproxAverage(rho,n)
ImPI = PI - eye(n,n)
ImPI[:,1] = ones(n,1)
b = zeros(n,1)
b[1,1] = 1

include("Exercise4_functions.jl")

x = LUSolve(ImPI',b)
