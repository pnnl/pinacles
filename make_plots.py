import proplot
import collections
import xarray as xr
import os
import pylab as plt


def main(out_path, case_dict):

    plot_timeseries(out_path, case_dict, "VelocityState")
    plot_timeseries(out_path, case_dict, "MicroBase")
    # var_list = get_var_list(case_dict,
    #    'ScalarState/timeseries')

    return


def plot_timeseries(out_path, case_dict, group):

    # Loop over cases and open group
    xr_dict = {}
    group = group + "/timeseries"
    for case in case_dict:
        xr_dict[case] = xr.open_mfdataset(case_dict[case].path, group=group)

    list_of_vars = get_unique_vars(xr_dict)

    fig_path = os.path.join(os.path.join(out_path, group), group)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    for case in xr_dict:
        for var in list_of_vars:
            print("Plotting ", var)
            time = xr_dict[case]["time"][:]
            data = xr_dict[case][var][:]
            plt.figure()
            plt.plot(time / 3600.0, data, label=case)
            plt.xlabel("Time (h)")
            plt.legend()
            plt.savefig(os.path.join(fig_path, var + ".png"))
            plt.close()

    return


def get_unique_vars(xr_dict):
    list_of_vars = []
    for case in xr_dict:
        for key in xr_dict[case].variables.keys():
            if key not in list_of_vars:
                list_of_vars.append(key)

    return list_of_vars


class Case:
    def __init__(self, path, color, label):
        self._path = path
        self._color = color
        self._label = label

        return

    @property
    def path(self):
        return self._path

    @property
    def color(self):
        return self._color

    @property
    def label(self):
        return self._label


if __name__ == "__main__":

    case_dict = collections.OrderedDict()

    out_path = "./plots"

    case_dict["bomex"] = Case("./bomex/stats.nc", color="blue", label="p3")

    main(out_path, case_dict)
