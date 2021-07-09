import glob
import os
import shutil


def main(path, out_dir):

    times = glob.glob(os.path.join(path, "*"))

    # Sort files by creation time
    times = sorted(times, key=os.path.getmtime)

    # Now loop through time directories and split the strings
    for ti in times:
        actual_time = float(ti.split("/")[-1])
        renamed_time = str(int(100000000 + actual_time))

        out_file = os.path.join(out_dir, renamed_time + ".nc")

        in_file = os.path.join(ti, "0.nc")
        print("Moving", in_file, out_file)

        # print(ti, out_file)
        shutil.move(in_file, out_file)

    return


def main_fix(path, out_dir):

    times = glob.glob(os.path.join(path, "*"))

    # Sort files by creation time
    times = sorted(times, key=os.path.getmtime)

    # Now loop through time directories and split the strings
    for ti in times:
        # print(ti)
        actual_time = ti.split("/")[-1]
        old_file = os.path.join(ti, "0.nc")
        new_file = os.path.join(out_dir, actual_time.split(".")[0] + ".nc")
        # renamed_time = str(int(100000000 + actual_time))

        # out_file = os.path.join(out_dir, renamed_time + '.nc')

        print("Moving", old_file, new_file)
        shutil.move(old_file, new_file)

    return


if __name__ == "__main__":

    path = "/Users/pres026/Research/PINACLES_LDRD/PINACLES/bomex_started_2021_03_05-07_23_43_AM_nest/fields"
    out_dir = "./nest_combined/"
    main(path, out_dir)
