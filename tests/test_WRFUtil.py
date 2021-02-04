import pytest
import numpy as np
from pinacles import WRFUtil

@pytest.fixture()
def array_shapes():

    array_shapes = []
    array_shapes.append((9, 9, 9))
    array_shapes.append((8, 9, 8))
    array_shapes.append((7, 8, 9))

    return array_shapes


def test_simple_reorder(array_shapes):

    for n_halo in [(1,1,1), (2, 2, 2), (3, 3, 3), (1, 2, 3), (3, 2, 1), (1, 3, 2)]:
        n_halo_array = np.array(n_halo)
        for shape in array_shapes:
            our_shape = np.array(shape) + n_halo_array

            #Allocate appropriately sized fortran and C ordered arrays 
            our_array = np.random.randn(our_shape[0], our_shape[1], our_shape[2])
            wrf_array = np.empty((our_shape[0], our_shape[2], our_shape[1]), order='F')
            return_array = np.empty_like(our_array)

            # Now transpose into wrf order
            WRFUtil.to_wrf_order(n_halo_array, our_array, wrf_array)

            #Now transpose back into the return array
            WRFUtil.to_our_order(n_halo_array, wrf_array, return_array)
            
            #Now check and make sure that return_array and our array are identical over non-halo points

            assert np.array_equal(our_array[n_halo_array[0]:-n_halo_array[0],
                n_halo_array[1]:-n_halo_array[1],
                n_halo_array[2]:-n_halo_array[2]], 
                return_array[n_halo_array[0]:-n_halo_array[0],
                n_halo_array[1]:-n_halo_array[1],
                n_halo_array[2]:-n_halo_array[2]])

    return

def test_simple_reorder_withhalo(array_shapes):

    for n_halo in [(1,1,1), (2, 2, 2), (3, 3, 3), (1, 2, 3), (3, 2, 1), (1, 3, 2)]:
        n_halo_array = np.array(n_halo)
        for shape in array_shapes:
            our_shape = np.array(shape) + n_halo_array

            #Allocate appropriately sized fortran and C ordered arrays 
            our_array = np.random.randn(our_shape[0], our_shape[1], our_shape[2])
            wrf_array = np.empty((our_shape[0], our_shape[2], our_shape[1]), order='F')
            return_array = np.empty_like(our_array)

            # Now transpose into wrf order
            WRFUtil.to_wrf_order_halo(n_halo_array, our_array, wrf_array)

            #Now transpose back into the return array
            WRFUtil.to_our_order_halo(n_halo_array, wrf_array, return_array)
            
            #Now check and make sure that return_array and our array are identical over non-halo points

            assert np.array_equal(our_array, return_array)

    return

@pytest.fixture()
def array_shapes_4d():

    array_shapes = []
    array_shapes.append((9, 9, 9, 3))
    array_shapes.append((8, 9, 8, 4))
    array_shapes.append((7, 8, 9, 1))

    return array_shapes

def test_simple_reorder_4d(array_shapes_4d):

    for n_halo in [(1,1,1), (2, 2, 2), (3, 3, 3), (1, 2, 3), (3, 2, 1), (1, 3, 2)]:
        n_halo_array = np.append([0], np.array(n_halo))
        for shape in array_shapes_4d:
            our_shape = np.array(shape) + n_halo_array

            #Allocate appropriately sized fortran and C ordered arrays 
            our_array = np.random.randn( shape[3], our_shape[0], our_shape[1], our_shape[2])
            wrf_array = np.empty((our_shape[0], our_shape[2], our_shape[1], shape[3]), order='F')
            return_array = np.empty_like(our_array)

            # Now transpose into wrf order
            WRFUtil.to_wrf_order_4d(n_halo_array, our_array, wrf_array)

            #Now transpose back into the return array
            WRFUtil.to_our_order_4d(n_halo_array, wrf_array, return_array)
            
            #Now check and make sure that return_array and our array are identical over non-halo points

            assert np.array_equal(our_array[n_halo_array[0]:-n_halo_array[0],
                n_halo_array[1]:-n_halo_array[1],
                n_halo_array[2]:-n_halo_array[2],
                :], 
                return_array[n_halo_array[0]:-n_halo_array[0],
                n_halo_array[1]:-n_halo_array[1],
                n_halo_array[2]:-n_halo_array[2], 
                :])

    return


def test_simple_reorder_4d_halo(array_shapes_4d):

    for n_halo in [(1,1,1), (2, 2, 2), (3, 3, 3), (1, 2, 3), (3, 2, 1), (1, 3, 2)]:
        n_halo_array = np.append([0], np.array(n_halo))
        for shape in array_shapes_4d:
            our_shape = np.array(shape) + n_halo_array

            #Allocate appropriately sized fortran and C ordered arrays 
            our_array = np.random.randn( shape[3], our_shape[0], our_shape[1], our_shape[2])
            wrf_array = np.empty((our_shape[0], our_shape[2], our_shape[1], shape[3]), order='F')
            return_array = np.empty_like(our_array)


            # Now transpose into wrf order
            WRFUtil.to_wrf_order_4d_halo(n_halo_array, our_array, wrf_array)

            #Now transpose back into the return array
            WRFUtil.to_our_order_4d_halo(n_halo_array, wrf_array, return_array)
            
            #Now check and make sure that return_array and our array are identical over non-halo points
            assert np.array_equal(our_array, return_array)
    
    return