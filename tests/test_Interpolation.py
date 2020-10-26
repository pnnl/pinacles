import numpy as np
from scipy import stats
from pinacles import interpolation_impl


def test_interp_second():


    #Test for convergence of the interpolator
    def f(x):
        return  np.maximum(np.minimum(np.cos(x), 0.5),-0.5)
    
    error = []
    dxs = []
    skips = [1,2,4,8,16,32]
    skip=0
    for n in [ 33, 65, 129, 257, 513, 1025]:

        #Make a grid
        soln = []
        fit = []

        x = np.linspace(0.0,4.0 * np.pi,n)
        dx = x[1] - x[0]

        for xi in x:
            soln.append(f(xi))

            fit.append(interpolation_impl.centered_second(
            f(-0.5*dx + xi), f(0.5*dx + xi)))

        soln = np.array(soln)
        fit = np.array(fit)

        #print(x[::skips[skip]])

        error.append(np.linalg.norm(fit[::skips[skip]] - soln[::skips[skip]]))
        dxs.append(dx)


        #print(x[::skips[skip]])
        #print(error)
        skip += 1

    error = np.array(error)

    dxs = np.array(dxs)

    #Determine convergence
    slope, intercept, r_value, p_value, std_err = stats.linregress( np.log(dxs), np.log(error))
    assert(slope > 1.5)

    return

def test_interp_fourth():

    #Test for convergence of the interpolator
    def f(x):
        return np.maximum(np.minimum(np.cos(x), 0.5),-0.5)
    
    error = []
    dxs = []
    skips = [1,2,4,8,16,32]
    skip=0
    for n in [ 33, 65, 129, 257, 513, 1025]:

        #Make a grid
        soln = []
        fit = []

        x = np.linspace(0.0,4.0 * np.pi,n)
        dx = x[1] - x[0]

        for xi in x:
            soln.append(f(xi))

            fit.append(interpolation_impl.centered_fourth(
            f(-1.5*dx + xi),
            f(-0.5*dx + xi), f(0.5*dx + xi), f(1.5*dx + xi)))

        soln = np.array(soln)
        fit = np.array(fit)

        #print(x[::skips[skip]])

        error.append(np.linalg.norm(fit[::skips[skip]] - soln[::skips[skip]]))
        dxs.append(dx)
        #print(error)
        skip += 1
    error = np.array(error)

    dxs = np.array(dxs)

    #Determine convergence
    slope, intercept, r_value, p_value, std_err = stats.linregress( np.log(dxs), np.log(error))
    assert(slope > 3.5)

    return


def test_interp_sixth():
    
    #Test for convergence of the interpolator
    def f(x):
        return np.maximum(np.minimum(np.cos(x), 0.5),-0.5)
    
    error = []
    dxs = []
    skips = [1,2,4,8,16,32]
    skip=0
    for n in [ 33, 65, 129, 257, 513, 1025]:

        #Make a grid
        soln = []
        fit = []

        x = np.linspace(0.0,4.0 * np.pi,n)
        dx = x[1] - x[0]

        for xi in x:
            soln.append(f(xi))

            fit.append(interpolation_impl.centered_sixth(
            f(-2.5 * dx + xi),f(-1.5*dx + xi),
            f(-0.5*dx + xi), f(0.5*dx + xi), f(1.5*dx + xi), f(2.5*dx + xi)))

        soln = np.array(soln)
        fit = np.array(fit)

        error.append(np.linalg.norm(fit[::skips[skip]] - soln[::skips[skip]]))
        dxs.append(dx)
        skip += 1
    error = np.array(error)

    dxs = np.array(dxs)

    #Determine convergence
    slope, intercept, r_value, p_value, std_err = stats.linregress( np.log(dxs), np.log(error))
    assert(slope > 5.5)


    return


def test_interp_weno5():

    interp_val = interpolation_impl.interp_weno5(0.0, 1.0, 2.0, 3.0, 4.0)
    #assert(2.5 == interp_val)


    #Test for convergence of the interpolator
    def f(x):
        return np.maximum(np.minimum(np.cos(x), 0.5),-0.5)
    
    error = []
    dxs = []
    skips = [1,2,4,8,16]
    skip=0
    for n in [ 65, 129, 257, 513, 1025]:

        #Make a grid
        soln = []
        fit = []

        x = np.linspace(0.0,4.0 * np.pi,n)
        dx = x[1] - x[0]

        for xi in x:
            soln.append(f(xi))

            fit.append(interpolation_impl.interp_weno5(
            f(-2.5 * dx + xi),f(-1.5*dx + xi),
            f(-0.5*dx + xi), f(0.5*dx + xi), f(1.5*dx + xi)))

        soln = np.array(soln)
        fit = np.array(fit)

        error.append(np.linalg.norm(fit[::skips[skip]] - soln[::skips[skip]]))
        dxs.append(dx)
        skip += 1
    error = np.array(error)

    dxs = np.array(dxs)

    #Determine convergence
    slope, intercept, r_value, p_value, std_err = stats.linregress( np.log(dxs), np.log(error))
    assert(slope > 4.5)


    return

def test_interp_weno7():
    
    interp_val = interpolation_impl.interp_weno5(0.0, 1.0, 2.0, 3.0, 4.0)

    #Test for convergence of the interpolator
    def f(x):
        return np.maximum(np.minimum(np.cos(x), 0.5),-0.5)
    
    error = []
    dxs = []
    skips = [1,2,4,8,16]
    skip=0
    for n in [ 65, 129, 257, 513, 1025]:

        #Make a grid
        soln = []
        fit = []

        x = np.linspace(0.0,4.0 * np.pi,n)
        dx = x[1] - x[0]

        for xi in x:
            soln.append(f(xi))

            fit.append(interpolation_impl.interp_weno7(
            f(-3.5 * dx + xi),f(-2.5 * dx + xi),f(-1.5*dx + xi),
            f(-0.5*dx + xi), f(0.5*dx + xi), f(1.5*dx + xi),f(2.5*dx + xi)))

        soln = np.array(soln)
        fit = np.array(fit)

        #print(x[::skips[skip]])

        error.append(np.linalg.norm(fit[::skips[skip]] - soln[::skips[skip]]))
        dxs.append(dx)
        skip += 1
    error = np.array(error)

    dxs = np.array(dxs)

    #Determine convergence
    slope, intercept, r_value, p_value, std_err = stats.linregress( np.log(dxs), np.log(error))
    assert(slope > 6.5)



    return