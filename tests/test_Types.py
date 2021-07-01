import numpy as np
from pinacles import Types

def test_double():
    namelist = {}
    # First test that everything defaults ot double when precision options 
    # are not set in the namelist
    types = Types.Types(namelist)

    assert(types.float is np.float64)
    assert(types.complex is np.complex128)
    assert(types.float_pressure is np.float64)
    assert(types.complex_pressure is np.complex128)

    # Not that you could set precision to double in the namelist, but it is not needed

    return 


def test_mixed_pressure_only():

    namelist = {}
    namelist['precision'] = 'mixed_pressure_only'
    
    types = Types.Types(namelist)
    assert(types.float is np.float64)
    assert(types.complex is np.complex128)
    assert(types.float_pressure is np.float32)
    assert(types.complex_pressure is np.complex64)

    return

def test_mixed():
    namelist = {}
    namelist['precision'] = 'mixed'

    types = Types.Types(namelist)
    assert(types.float is np.float32)
    assert(types.complex is np.complex64)
    assert(types.float_pressure is np.float32)
    assert(types.complex_pressure is np.complex64)
    return