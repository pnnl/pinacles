import argparse
import json

def main(casename):
    ''' The function generates input files that are necessary for testing.'''
    if casename == 'colliding_blocks':
        input_dict = colliding_blocks()
    elif casename == 'sullivan_and_patton':
        input_dict = sullivan_and_patton()
    elif casename == 'stable_bubble':
        input_dict = stable_bubble()

    write_file(casename, input_dict)

    return

def write_file(casename, input_dict):
    with open('./' + casename + '.json', 'w') as input_file_out:
        json.dump(input_dict, input_file_out, sort_keys=True, indent=4)
    return

def stable_bubble():

    input_dict = {}

    key = 'meta'
    input_dict[key] = {}
    input_dict[key]['casename'] = 'stable_bubble'
    input_dict[key]['output_directory'] = './'

    key = 'grid'
    input_dict[key] = {}
    #Set the number of grid points in the domain
    input_dict[key]['n'] = [512 ,4, 64]
    #Set the number of halo points in each direct
    input_dict[key]['n_halo'] = [3, 3, 3]
    #Set the domain length, dx will be determined from n and L
    input_dict[key]['l'] = [51200.0, 51200.0, 6400.0]

    key = 'damping'
    input_dict[key] = {}
    input_dict[key]['vars'] = ['u', 'v', 'w', 's']
    input_dict[key]['depth'] = 250.0
    input_dict[key]['timescale'] = 50.0

    key = 'time'
    input_dict[key] = {}
    input_dict[key]['cfl'] = 0.6

    key = 'stats'
    input_dict[key] = {}
    input_dict[key]['frequency'] = 60.0
    input_dict[key]['modules'] = []


    return input_dict

def colliding_blocks():
    ''' data necessary for the simplest testing of the domain '''
    input_dict = {}

    key = 'meta'
    input_dict[key] = {}
    input_dict[key]['casename'] = 'colliding_blocks'

    key = 'grid'
    input_dict[key] = {}
    #Set the number of grid points in the domain
    input_dict[key]['n'] = [256, 256, 5]
    #Set the number of halo points in each direct
    input_dict[key]['n_halo'] = [3, 3, 3]
    #Set the domain length, dx will be determined from n and L
    input_dict[key]['l'] = [1000.0, 1000.0, 1000.0]

    key = 'time'
    input_dict[key] = {}
    input_dict[key]['cfl'] = 0.6

    return input_dict


def sullivan_and_patton():
    input_dict = {}

    key = 'meta'
    input_dict[key] = {}
    input_dict[key]['casename'] = 'sullivan_and_patton'
    input_dict[key]['output_directory'] = './'

    key = 'grid'
    input_dict[key] = {}
    #Set the number of grid points in the domain
    input_dict[key]['n'] = [32, 32, 32]
    #Set the number of halo points in each direct
    input_dict[key]['n_halo'] = [3, 3, 3]
    #Set the domain length, dx will be determined from n and L
    input_dict[key]['l'] = [5120.0, 5120.0, 2048.0]

    key = 'damping'
    input_dict[key] = {}
    input_dict[key]['vars'] = ['u', 'v', 'w', 's']
    input_dict[key]['depth'] = 250.0
    input_dict[key]['timescale'] = 50.0

    key = 'time'
    input_dict[key] = {}
    input_dict[key]['cfl'] = 0.6

    key = 'stats'
    input_dict[key] = {}
    input_dict[key]['frequency'] = 60.0
    input_dict[key]['modules'] = []

    return input_dict

LIST_OF_CASES = ['colliding_blocks',
                 'stable_bubble',
                'sullivan_and_patton']

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Aarg paraser for generating input files.')
    parser.add_argument('casename')
    args = parser.parse_args()

    assert(args.casename in LIST_OF_CASES)

    main(args.casename)
