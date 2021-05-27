import subprocess 
import os

def main():
    build_script('pinacles/wrf_physics', 'build_p3.sh', 'p3')
    f2py_file('pinacles/wrf_physics/', 'module_mp_kessler.f95', 'kessler')

    return 

def f2py_file(path, source, extname):

    orig_path = os.getcwd()
    os.chdir(path)
    cmd = 'f2py ' + ' -c ' + source +  ' -m ' + extname
    subprocess.call([cmd], shell=True)
    os.chdir(orig_path)

    return

def build_script(path, source, extname):

    print('Running build script for: ', extname)
    orig_path = os.getcwd()
    os.chdir(path)
    cmd = 'sh ' + source
    subprocess.call([cmd], shell=True)
    os.chdir(orig_path)

    return


if __name__ == '__main__':
    main()
