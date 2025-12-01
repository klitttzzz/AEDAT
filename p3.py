

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

def lcs(X:str, Y:str)->int:
    m = len(X)
    n = len(Y)
    c = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1,m+1):
        for j in range(1,n+1):
            if X[i-1] == Y[j-1]: # Correction of the indices: they start from 0
                c[i][j] = 1 + c[i-1][j-1]
            else:
                c[i][j] = max(c[i-1][j], c[i][j-1])

    return c[-1][-1]


def coin_change(coins: list[int], amount: int) -> int:
    n = len(coins)

    m = [[float('inf')] * (amount + 1) for _ in range(n)]

    ch = [[ [0]*n for _ in range(amount + 1) ] for _ in range(n)]
    for i in range(n):
        m[i][0] = 0
        ch[i][0] = [0]*n

    for c in range(1, amount + 1):
        if c % coins[0] == 0:
            m[0][c] = c // coins[0]
            ch[0][c][0] = c // coins[0]
        else:
            m[0][c] = float('inf')

    for i in range(1, n):
        for c in range(1, amount + 1):
            m[i][c] = m[i-1][c]
            ch[i][c] = ch[i-1][c][:]

            if coins[i] <= c and m[i][c - coins[i]] + 1 < m[i][c]:
                m[i][c] = 1 + m[i][c - coins[i]]
                ch[i][c] = ch[i][c - coins[i]][:]  # copia
                ch[i][c][i] += 1

    return -1 if m[-1][-1] == float('inf') else m[-1][-1]


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



print(fib(10)) # Salida esperada: 55
weights = [2, 3, 4, 5]
values = [3, 4, 5, 6]
print(knapsack(5, weights, values)) # Salida esperada: 7
print(lcs("ABCBDAB", "BDCAB")) # Salida esperada: 4
coins = [1, 2, 10]
print(coin_change(coins, 18)) # Salida esperada: 3 (5+5+1)
print(unique_path(4, 4))    # Salida esperada: 20


    
