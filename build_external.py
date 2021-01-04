import subprocess 
import os

def main():
    #build_script('pinacles/wrf_physics', 'build_p3.sh', 'p3')
    #f2py_file('pinacles/wrf_physics/', 'module_mp_kessler.f95', 'kessler')
    f2py_file('pinacles/wrf_physics/', 'module_mp_kessler_split.f95', 'kessler_split')

    #download_data('pinacles/wrf_physics/') 

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
