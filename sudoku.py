from itertools import product
from pprint import pprint

"""main function size=size of subblock"""

def solve_sudoku(size, grid):
    R, C = size
    N = R * C  #9(N) = 3(R)*3(C)
    X = ([("rc", rc) for rc in product(range(N), range(N))] +  # row and column combinations
         [("rn", rn) for rn in product(range(N), range(1, N + 1))] +  # row and number combinations
         [("cn", cn) for cn in product(range(N), range(1, N + 1))] +  # column and number combinations
         [("bn", bn) for bn in product(range(N), range(1, N + 1))])  # block and number combinations
    """ 
    X has constraints in it.
    1. numbers do not repeat in a row. every row must have 9 numbers in it and there are n rows = 9*9
    2. numbers do not repeat in a column. every col must have n numbers in it and there are n col = 9*9
    3. numbers do not repeat in a block. every block must have n numbers in it and there are n blocks = 9*9
    4. every cell must have 1 numbers in it and there are 81 cells. 
    total number of constraints are 4 * 81 = 324
    X is dictionary where {('bn', (0, 1)): {(0, 1, 1), (1, 1, 1), (0, 0, 1), (1, 0, 1)}} means 
    for block 0 and number 1 has above row, column, number combinations. 
    """
    #print(len(X))
    #pprint(X)
    Y = dict()
    for r, c, n in product(range(N), range(N), range(1, N + 1)):
        b = (r // R) * R + (c // C)  # Box number
        Y[(r, c, n)] = [
            ("rc", (r, c)),
            ("rn", (r, n)),
            ("cn", (c, n)),
            ("bn", (b, n))]
    """
    created a dictionary Y where key is equal to value of X and value is equal to key in X.
    for example in Y: {(0, 0, 1): [('rc', (0, 0)), ('rn', (0, 1)), ('cn', (0, 1)), ('bn', (0, 1))]}
    row 0, col 0 and number 1 can be made from any of the two combinations of values from above.
    """
    X, Y = exact_cover(X, Y)  # this functions creates a exact cover dictionary of the sudoku
    #print("\nX", len(X))
    #pprint(X)
    #print("\nY", len(Y))
    #pprint(Y)

    """ 
    we will not be changing the values which are already present in the matrix
    1. delete the keys in - X("rc") such that grid[r][c]!=0.
                          - X("rn") such that n is already present in row r
                          - X("cn") such that n is already present in column c
                          - X("bn") such that n is already present in block b
    2. delete the values in X such that:
                            X(key: (r,c,n)) such that grid[r][c]!=n and block made from r and c has a number n
    """
    for i, row in enumerate(grid):
        for j, n in enumerate(row):
            if n:
                select(X, Y, (i, j, n))

    # print("\nX", len(X))
    # pprint(X)
    # print("\nY", len(Y))
    # pprint(Y)

    """backtracking algorithm X"""
    for solution in solve(X, Y, []):
        for (r, c, n) in solution:
            grid[r][c] = n
        yield grid

"""create dictionary X such that X(key:value) is Y(value:key)"""
def exact_cover(X, Y):
    X = {j: set() for j in X}
    for i, row in Y.items():
        for j in row:
            X[j].add(i)
    return X, Y

"""algorithm X backtracking"""

def solve(X, Y, solution):
    """check if X is empty or not.
    if X is not empty than continue"""
    if not X:
        yield list(solution)
    else:
        """consider X with minimum number of values in it"""
        c = min(X, key=lambda c: len(X[c]))
        """take a value from above X: r"""
        for r in list(X[c]):
            #pprint(solution)
            """append r to the solution"""
            solution.append(r)
            """ select r. Y: {r:values}, search for these values in X and remove them from X """
            cols = select(X, Y, r)
            #pprint(cols)
            """recursively call the method solve till X is not empty. if X becomes empty than we found a solution """
            for s in solve(X, Y, solution):
                yield s
            """if X is not empty than we backtrack by adding the key:values back in X which we removed """
            deselect(X, Y, r, cols)
            solution.pop()

def select(X, Y, r):
    cols = []
    for j in Y[r]:
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].remove(i)
        cols.append(X.pop(j))
    return cols

def deselect(X, Y, r, cols):
    for j in reversed(Y[r]):
        X[j] = cols.pop()
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].add(i)