

def fib(n: int)->int:
    x, y = 0, 1
    for i in range(2,n+1):
        x, y = y, x+y
    return y




def knapsack(W: int, weights: list[int], values: list[int]) -> int:
    n = len(weights)
    dp = [0] * (W + 1)
    
    for i in range(n):
        for w in range(W, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[W]

def lcs(x, y):
    m = len(x)
    n = len(y)
    c = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1,m+1):
        for j in range(1,n+1):
            if x[i-1] == y[j-1]: # Correction of the indices: they start from 0
                c[i][j] = 1 + c[i-1][j-1]
            else:
                c[i][j] = max(c[i-1][j], c[i][j-1])

    return c[-1][-1]


def change(v, C):
    n = len(v)
    m = [[10**9] * (C + 1) for _ in range(n)]
    for i in range(n): # initialize the matrices
        m[i][0] = 0
    for c in range(C):
        if (c % v[0]) == 0:
            m[0][c] = c / v[0]
        for i in range(1, n):
            for c in range(1,C+1):
                if v[i] > c or 1 + m[i][c-v[i]] > m[i-1][c]:
                    m[i][c] = m[i-1][c]
            else:
                m[i][c] = 1 + m[i][c-v[i]]
    return m[-1][-1]


def unique_path(m:int, n:int)->int:
    c = [[0] * (n) for _ in range(m)]
    for i in range(m):
        c[i][0] = 1
    for i in range(n):
        c[0][i] = 1
    for i in range(1, m):
        for j in range(1, n):
            c[i][j] = c[i][j-1] + c[i-1][j]

    return c[m-1][n-1]

print(unique_path(4, 4))

    
