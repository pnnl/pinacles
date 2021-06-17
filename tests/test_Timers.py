from pinacles import Timers
import pytest
import time
import netCDF4 as nc
import os


@pytest.fixture
def mock_timers(tmpdir):

    timer_mocks = []

    namelist = {}

    key = "stats"
    namelist[key] = {}
    namelist[key]["frequency"] = 30.0

    key = "meta"
    namelist[key] = {}
    namelist[key]["simname"] = "timer_test"
    namelist[key]["output_directory"] = tmpdir

    class TimeSteppingController:
        def __init__(self):
            self._time = 10.0

    TSC = TimeSteppingController()

    timer_mocks.append(Timers.Timer(namelist, TSC))

    return timer_mocks


def test_timers_attrs(mock_timers):

    attrs = [
        "add_timer",
        "start_timer",
        "end_timer",
        "_frequency",
        "frequency",
        "_n_calls",
        "finish_timestep",
        "get_accumulated_time",
        "update",
        "initialize",
    ]
    for attr in attrs:
        assert all(hasattr(timer, attr) for timer in mock_timers)

    return


def test_full(mock_timers):

    n_timers = 2  # number of timers to create in test

    # Loop over the mock timers
    for timer in mock_timers:

        # Create n_timers (number set above)
        for n in range(n_timers):

            timer_name = "test_create" + str(n)
            timer.add_timer(timer_name)

            assert timer.n_timers - 1 == n
            assert timer.frequency == 30.0

        timer.initialize()

    for timer in mock_timers:
        for n in range(n_timers):

            timer_name = "test_create" + str(n)

            for i in range(1, 5):
                timer.start_timer(timer_name)
                time.sleep(1e-5 * n)
                timer.end_timer(timer_name)

                assert timer.get_accumulated_time(timer_name) >= i * n * 1e-5

            assert timer._n_calls[timer_name] == 4

    # Make sure the correct paths exists
    for timer in mock_timers:
        assert os.path.exists(timer._output_root)
        assert os.path.exists(timer._output_path)
        assert os.path.exists(timer._stats_file)

    # Now write data to the netcdf file
    for timer in mock_timers:
        timer.update()

    # Now make sure that the netCDF file has the correct information
    for timer in mock_timers:
        rt_grp = nc.Dataset(timer._stats_file, "r")
        assert "time" in rt_grp.dimensions
        assert rt_grp["time"].units == "s"

        for n in range(n_timers):
            timer_name = "test_create" + str(n) + "_max"
            assert rt_grp[timer_name].units == "s"

            timer_name = "test_create" + str(n) + "_min"
            assert rt_grp[timer_name].units == "s"

            timer_name = "test_create" + str(n) + "_pertimestep_max"
            assert rt_grp[timer_name].units == "s"

            timer_name = "test_create" + str(n) + "_pertimestep_min"
            assert rt_grp[timer_name].units == "s"

        rt_grp.close()

    return
