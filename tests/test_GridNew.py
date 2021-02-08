import pytest
from pinacles import Grid


@pytest.fixture
def grid_fix():
    list_of_grids = []


    this_nml = {}
    this_nml['grid'] = {}
    this_nml['grid']['n'] = [10, 10, 10]
    this_nml['grid']['n_halo'] = [3, 3, 3]
    this_nml['grid']['l'] = [1000.0, 1000.0, 1000.0]

    list_of_grids.append(Grid.RegularCartesian(this_nml))

    return list_of_grids


def test_point_on_rank(grid_fix):

    assert all(hasattr(grid, 'point_on_rank') for grid in grid_fix)

    for grid in grid_fix:
        on_rank = grid.point_on_rank(100.0, 100.0, 100.0)
        assert on_rank

        on_rank = grid.point_on_rank(-100.0, 100.0, 100.0)
        assert not on_rank

        on_rank = grid.point_on_rank(100.0, -100.0, 100.0)
        assert not on_rank

    return

def test_point_indicies(grid_fix):

    assert all(hasattr(grid, 'point_indicies') for grid in grid_fix)

    for grid in grid_fix:
        on_rank = grid.point_on_rank(100.0, 100.0, 100.0)
        
        assert on_rank

        if on_rank:
            xi, yi, zi = grid.point_indicies(100.0, 100.0, 100.0)

            assert xi == grid.n_halo[0]
            assert yi == grid.n_halo[1]
            assert zi == grid.n_halo[2]

    return