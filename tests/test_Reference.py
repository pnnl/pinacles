from pinacles import Grid
from pinacles import ReferenceState


class test_ReferenceState:
    def __init__(self, namelist):

        self._namelist = namelist
        self._Grid = Grid.RegularCartesian(self._namelist)
        self._Ref = ReferenceState.ReferenceDry(self._namelist, self._Grid)

        return

    def test_setSurface(self):

        assert 1 == 0

        return


def test_multiple_namelist():

    ns = [(4, 4, 4)]
    ls = [
        (1000.0, 1000.0, 1000.0),
        (1000.0, 1000.0, 2000.0),
        (1000.0, 1000.0, 8000.0),
        (1000.0, 1000.0, 10000.0),
    ]
    n_halos = [(1, 1, 1), (2, 2, 2), (3, 3, 3), (1, 2, 3), (8, 8, 8)]

    for n in ns:
        for l in ls:
            for n_halo in n_halos:
                namelist = {}
                namelist["grid"] = {}
                namelist["grid"]["n"] = n
                namelist["grid"]["l"] = l
                namelist["grid"]["n_halo"] = n_halo

                test_ReferenceState(namelist)

    return
