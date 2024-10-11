"""
Exercise 13
"""

X = [-1,0,1]
P = [1/5, 2/5, 2/5]

"""
Point a
E[X] = 0.2
"""
mean = sum(X[i]*P[i] for i in range(len(X)))
print(mean)


"""
Point b
Y = X**2 
Y(0) = X(0)^2 = 0^2 = 0
Y(-1) = X(-1)^2 = (-1)^2 = 1
Y(1) = X(1)^2 = (1)^2 = 1

Since Y(-1) = Y(1), we can say that
    P[Y = 1] = P[X = 1] + P[X = -1]
So: 
    Y    = [0,1]
    P[Y] = [2/5, 3/5]

E[Y] = 0.6
"""
Y = [0,1]
P_y = [2/5,3/5]
mean_y = sum(Y[i]*P_y[i] for i in range(len(Y)))
print(mean_y)

"""
Point c
For the "change of formula" variable, we need to find function g that is applied to X
In our case, g(x) = x**2
this means that E[X**2] = sum(x**2 * P[x] for x in X) which gives out 0.6
We can see that E[X**2] = E[Y]

"""
mean_x2 = sum((X[i]**2) * P[i] for i in range(len(X)))
print(mean_x2)

"""
Point d
Var(X) = E[X^2] - E[X]^2  (5.833333 in this case) (0.56 in our case)
"""
variance_x = mean_x2 - mean**2
print(variance_x)