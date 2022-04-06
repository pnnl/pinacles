import subprocess
import os


def main():

    # Build the fast_sbm scheme
    build_script(
        "pinacles/externals/wrf_fast_sbm_wrapper", "debug_compile.sh", "fast_sbm"
    )

    # Build P3
    build_script("pinacles/externals/wrf_p3_wrapper", "build_p3.sh", "p3")
    
    # Build M2005_MA
    build_script("pinacles/externals/sam_m2005_ma_wrapper", "build_m2005_ma.sh", "m2005_ma")
    
    # Build Kessler
    f2py_file(
        "pinacles/externals/wrf_kessler_wrapper", "module_mp_kessler.f95", "kessler"
    )

    # Now optionally build RRTMG
    rrtmg_path = "pinacles/externals/rrtmg_wrapper"
    rrtmg_lw_exists = os.path.exists(os.path.join(rrtmg_path, "librrtmglw.so"))
    rrtmg_sw_exists = os.path.exists(os.path.join(rrtmg_path, "librrtmgsw.so"))
    if not rrtmg_lw_exists and not rrtmg_sw_exists:
        # RRTMG does not appear to be compiled so we will compile it no
        build_script(rrtmg_path, "debug_compile.sh", "rrtmg")
    else:
        print("Using existing compilation of rrtmg.")

    return


def f2py_file(path, source, extname):
    """Build f2py wrapper for an external Fortran dependency.

    Args:
        path (str): path to directory containing the Fortran source for the module
        source (str): filename for the fortran source
        extname ([type]): name of the module, this is just for printing to terminal
    """

    orig_path = os.getcwd()
    os.chdir(path)
    cmd = "f2py " + " -c " + source + " -m " + extname
    subprocess.call([cmd], shell=True)
    os.chdir(orig_path)

    return


def build_script(path, source, extname):
    """Build cffi wrapper for an external Fortran dependency.

    Args:
        path (str): path to directory containing a build script for the module
        source (str): filename of the build script
        extname (str): name of the module, this is just for printing to terminal
    """

    print("Running build script for: ", extname)
    orig_path = os.getcwd()
    os.chdir(path)
    cmd = "sh " + source
    subprocess.call([cmd], shell=True)
    os.chdir(orig_path)

    return


if __name__ == "__main__":
    main()
