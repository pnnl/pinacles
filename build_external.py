import subprocess 
import os

def main():
    f2py_file('Columbia/wrf_physics/', 'module_mp_p3.f95', 'p3')
    f2py_file('Columbia/wrf_physics/', 'module_mp_kessler.f95', 'kessler')

    download_data('Columbia/wrf_physics/') 

    return 

def download_data(path):
    urls = ['https://raw.githubusercontent.com/wrf-model/WRF/master/run/p3_lookup_table_1.dat-v4.1',
            'https://raw.githubusercontent.com/wrf-model/WRF/master/run/p3_lookup_table_2.dat-v4.1']

    orig_path = os.getcwd()
    os.chdir(path)
    
    for u in urls:
        cmd = 'wget ' + u 
        subprocess.call([cmd], shell=True)

    os.chdir(orig_path)
    
    return 

def f2py_file(path, source, extname):

    orig_path = os.getcwd()
    os.chdir(path)
    cmd = 'f2py ' + ' -c ' + source +  ' -m ' + extname
    subprocess.call([cmd], shell=True)
    os.chdir(orig_path)

    return


if __name__ == '__main__':
    main()
