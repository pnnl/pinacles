import json 

def main(): 
    ''' The function generates input files that are necessary for testing.'''
    with open('./domain_test.json', 'w') as input_file_out: 
        input_dict = domain_test() 
        json.dump(input_dict, input_file_out, sort_keys=True, indent=4)

    return 


def domain_test(): 
    ''' data necessary for the simplest testing of the domain '''
    input_dict = {}
    key = 'grid'
    input_dict[key] = {}
    #Set the number of grid points in the domain
    input_dict[key]['n'] = [16, 16, 16]
    #Set the number of halo points in each direct
    input_dict[key]['n_halo'] = [1, 1, 1]
    #Set the domain length, dx will be determined from n and L
    input_dict[key]['l'] = [3.2, 3.2, 3.2]

    return input_dict 

if __name__ == '__main__': 
    main() 