import argparse
import glob
import json
import os
import pickle as pkl
import numpy as np
import h5py
import tqdm



def generate_template():

    _template = {}
    _template["out_path"] = "./"
    _template["in_path"] = "./"

    with open("template.json", "w") as _tf:
        json.dump(_template, _tf, sort_keys=True, indent=4)


def combine_restarts(inputdict):
    path = inputdict["in_path"]
    files = glob.glob(os.path.join(path, "*.pkl"))

    out_path = inputdict["out_path"]

    _fx = h5py.File(os.path.join(out_path, "restart.h5"), "w")
    n_file = 0
    for _f in tqdm.tqdm(files):

        with open(_f, "rb") as _fh:
            _rd = pkl.load(_fh)

        containers = [_rd["ScalarState"], _rd["VelocityState"], _rd["DiagnosticState"]]

        _n = _rd["RegularCartesianGrid"]["_n"]
        _n_halo = _rd["RegularCartesianGrid"]["_n_halo"]
        _l = _rd["RegularCartesianGrid"]["_l"]


        _local_start = _rd["RegularCartesianGrid"]["_local_start"]
        _local_end = _rd["RegularCartesianGrid"]["_local_end"]

        if n_file == 0:
            _fx.attrs["nx"] = _n[0] 
            _fx.attrs["ny"] = _n[1]
            _fx.attrs["nz"] = _n[2]

            _fx.attrs["lx"] = _l[0]
            _fx.attrs["ly"] = _l[1]
            _fx.attrs["lz"] = _l[2]
            
            
            _fx.attrs["x"] = _rd["RegularCartesianGrid"]["_global_axes"][0][:]
            _fx.attrs["y"] = _rd["RegularCartesianGrid"]["_global_axes"][1][:]
            _fx.attrs["z"] = _rd["RegularCartesianGrid"]["_global_axes"][2][:]
            
            _fx.attrs["x_edge"] = _rd["RegularCartesianGrid"]["_global_axes_edge"][0][:]
            _fx.attrs["y_edge"] = _rd["RegularCartesianGrid"]["_global_axes_edge"][1][:]
            _fx.attrs["z_edge"] = _rd["RegularCartesianGrid"]["_global_axes_edge"][2][:]
            
            _fx.attrs['time'] = _rd["TimeStepManager"]["_time"]

        if n_file == 0:
            for _c in containers:
                for dof in _c["_dofs"]:
                    dset = _fx.create_dataset(dof, tuple(_n + 2*_n_halo), dtype="d")


        for _c in containers:
            state_array = _c["_state_array"]
            for dof in _c["_dofs"]:
                dset = _fx[dof]
                indx = _c["_dofs"][dof]

                arr = state_array[
                    indx,
                    :,
                    :,
                    :,
                ]
                
                
                dset[
                    _local_start[0]  : _local_end[0] + 2*_n_halo[0],
                    _local_start[1]: _local_end[1] + 2*_n_halo[1],
                    _local_start[2] : _local_end[2] + 2*_n_halo[2],
                ] = arr

            _c["_state_array"] = None
            _c["_tend_array"] = None

        if n_file == 0:
            opkl_file = os.path.join(out_path, "Restart.pkl")

            opkl = {}
            opkl["RegularCartesianGrid"] = _rd["RegularCartesianGrid"]
            for _v in [
                "_local_axes",
                "_local_axes_edge",
                "_ngrid_local",
                "_local_shape",
                "_local_start",
                "_local_end",
            ]:
                opkl["RegularCartesianGrid"][_v] = None

            opkl["namelist"] = _rd["namelist"]
            opkl["TimeStepManager"] = _rd["TimeStepManager"]

            for _c in ["ScalarState", "VelocityState", "DiagnosticState"]:
                opkl[_c] = _rd[_c]

            opkl["restart_type"] = "portable"

            with open(opkl_file, "wb") as fhopkl:
                pkl.dump(opkl, fhopkl)

        n_file += 1

    _fx.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Combine restart files.")
    parser.add_argument("--generate_template", default=False)
    parser.add_argument("--input_file")
    args = parser.parse_args()

    if args.generate_template:
        generate_template()

    if not args.generate_template and args.input_file is not None:
        with open(args.input_file, 'r') as input_file:
            input_dict = json.load(input_file)
        
        #print(input_dict)
        combine_restarts(input_dict)
