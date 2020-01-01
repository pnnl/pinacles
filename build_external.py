import subprocess 
import os

def main():
    
    f2py_file('Columbia/wrf_physics/', 'module_mp_kessler.f95', 'kessler')

    return 

def f2py_file(path, source, extname):

    orig_path = os.getcwd()
    os.chdir(path)
    cmd = 'f2py3 ' + ' -c ' + source +  ' -m ' + extname
    subprocess.call([cmd], shell=True)

    os.chdir(orig_path)

    return


if __name__ == '__main__':
    main()
