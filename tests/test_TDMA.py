import Columbia.TDMA as TDMA 

def test_TDMA(): 
    n = 128 
    Solver = TDMA.TDMA_solver(n)
    assert(n == Solver.n)

    return 