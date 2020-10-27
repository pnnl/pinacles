import numpy as np
import pytest
from pinacles import Grid
from mpi4py import MPI


comm_size = MPI.COMM_WORLD.Get_size()

#For now these are just serial tests
def check_axes(n, l, n_halo, TestGrid):
    # Local Coordiante
    x_local = TestGrid.x_local
    y_local = TestGrid.y_local
    z_local = TestGrid.z_local

    # Test that the axes are properties
    with pytest.raises(AttributeError):
        TestGrid.x_local = None

    with pytest.raises(AttributeError):
        TestGrid.y_local = None

    with pytest.raises(AttributeError):
        TestGrid.z_local = None

    # Global Coordiante
    x_global = TestGrid.x_global
    x_edge_global = TestGrid.x_edge_global
    y_global = TestGrid.y_global
    y_edge_global = TestGrid.y_edge_global
    z_global = TestGrid.z_global
    z_edge_global = TestGrid.z_edge_global

    # Test that the axes are properties
    with pytest.raises(AttributeError):
        TestGrid.x_global = None

    with pytest.raises(AttributeError):
        TestGrid.x_edge_global = None

    with pytest.raises(AttributeError):
        TestGrid.y_global = None

    with pytest.raises(AttributeError):
        TestGrid.y_edge_global = None

    with pytest.raises(AttributeError):
        TestGrid.z_global = None

    with pytest.raises(AttributeError):
        TestGrid.z_edge_global = None

    # Check that the axes are monotone and that dx is correct
    assert(np.all(np.isclose(np.diff(x_local),l[0]/n[0])))
    assert(np.all(np.isclose(np.diff(x_global),l[0]/n[0])))
    assert(np.all(np.isclose(np.diff(x_edge_global),l[0]/n[0])))

    assert(np.all(np.isclose(np.diff(y_local),l[1]/n[1])))
    assert(np.all(np.isclose(np.diff(y_global),l[1]/n[1])))
    assert(np.all(np.isclose(np.diff(y_edge_global),l[1]/n[1])))

    assert(np.all(np.isclose(np.diff(z_local),l[2]/n[2])))
    assert(np.all(np.isclose(np.diff(z_global),l[2]/n[2])))
    assert(np.all(np.isclose(np.diff(z_edge_global),l[2]/n[2])))

    return


def test_muiltiple_namelist():


    # Set numer of grid potins to test
    ns = [(4,4,4), (8,8,8), (16, 8, 8), (8, 16, 8), (8, 8, 16), (8, 7, 8), (32, 32, 32)]
    ls = [(1000.0, 1000.0, 1000.0), (1000.0, 1000.0, 3000.0), (1000.0, 3000.0, 1000.0), (3000.0, 1000.0, 1000.0)]
    n_halos = [(1, 1, 1), (2, 2, 2), (3, 3, 3), (1, 2, 3), (8, 8, 8)]

    # Loop over various options for the inputs
    for n in ns:
        for l in ls:
            for n_halo in n_halos:
                namelist = {}
                namelist['grid'] = {}
                namelist['grid']['n'] = n
                namelist['grid']['l'] = l
                namelist['grid']['n_halo'] = n_halo

                # The grid class returns arrays
                n = np.array(n)
                l = np.array(l)
                n_halo = np.array(n_halo)

                # Instantiate a grid object based on the namelist
                TestGrid = Grid.RegularCartesian(namelist)

                # Test that properties are correctly set  and the correct errors are thrown
                # when they are attempted to be overwritten
                assert(np.all(n == TestGrid.n))
                with pytest.raises(AttributeError):
                    TestGrid.n = n

                assert(np.all(l == TestGrid.l))
                with pytest.raises(AttributeError):
                    TestGrid.l = l

                assert(np.all(n_halo == n_halo))
                with pytest.raises(AttributeError):
                    TestGrid.n_halo = n_halo

                # Total number of points in the global grid including ghost points
                correct_ngrid = n + 2 * n_halo
                assert(np.all(TestGrid.ngrid == correct_ngrid))
                with pytest.raises(AttributeError):
                    TestGrid.ngrid = correct_ngrid

                # Local grid spacing and inverse grid spacing
                dx = l/n
                assert(np.all(dx == TestGrid.dx))
                with pytest.raises(AttributeError):
                    TestGrid.dx = dx

                dxi = 1.0/dx
                assert(np.all(dxi == TestGrid.dxi))
                with pytest.raises(AttributeError):
                    TestGrid.dxi = dxi

                # These are tests that should only be run serially
                # TODO implement these in parallel
                if comm_size == 1:
                    assert(np.all(correct_ngrid == TestGrid.ngrid_local))
                    with pytest.raises(AttributeError):
                        TestGrid.ngrid_local = correct_ngrid

                    assert(np.all(n == TestGrid.nl))
                    with pytest.raises(AttributeError):
                        TestGrid.nl = n


                check_axes(n, l, n_halo, TestGrid)

    return