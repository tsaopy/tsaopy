import sys
import numpy as np

# aux functions
def param_ptype_shape(params,test_ptype):
    if test_ptype == 'x0' or test_ptype == 'v0':
        sys.exit('Error: test ptype is x0 or v0.')
    indexes = []
    for elem in params:
        if elem.ptype == test_ptype:
            indexes.append(elem.index)
    if test_ptype=='c' and not len(indexes)==0:
        arraux = np.array(indexes)
        return max(arraux[:,0]),max(arraux[:,1])
    elif test_ptype=='c' and len(indexes)==0:
        return (0,0)
    elif (test_ptype=='a' or test_ptype=='b' or test_ptype=='f') and not\
        len(indexes)==0:
        return (max(indexes),)
    elif (test_ptype=='a' or test_ptype=='b' or test_ptype=='f') and \
        len(indexes)==0:
        return (0,)
    else:
        sys.exit('Error: test ptype is not a, b, c, or f.')
    pass
        
def params_array_shape(params):
    ptypes_set = []
    for _ in params:
        ptypes_set.append(_.ptype)
    ptypes_set = set(ptypes_set)
    
    if (not 'x0' in ptypes_set) or (not 'v0' in ptypes_set):
        sys.exit('Error: initial conditions are not properly defined.')

    p_shape_array = [(1,),(1,)]
    
    for _ in ['a','b','c']:
        p_shape_array.append(param_ptype_shape(params,_))
        
    p_shape_array.append((3,))
    
    return p_shape_array

def param_names(param):
    ptype,index = param.ptype,param.index
    if ptype == 'c' and len(index) == 2:
        return 'c'+str(index[0])+str(index[1])
    elif ptype == 'a' or ptype == 'b':
        return ptype+str(index)
    elif ptype == 'f' and index==1:
        return 'F'
    elif ptype == 'f' and index==2:
        return 'w'
    elif ptype == 'f' and index==3:
        return 'p'
    elif ptype == 'x0' or ptype == 'v0':
        return ptype
    else:
        sys.exit('Error naming parameters.')
    pass

def param_cindex(param):
    if param.ptype=='x0':
        return 0,0
    elif param.ptype=='v0':
        return 0,1
    elif param.ptype=='a':
        return 1,param.index-1
    elif param.ptype=='b':
        return 2,param.index-1
    elif param.ptype=='c':
        q = param.index
        return 3,(q[0]-1,q[1]-1)
    elif param.ptype=='f':
        return 4,param.index-1

def fitparams_coord_info(fparams):
    indexes,labels = [],[]
    for _ in fparams:
        indexes.append( param_cindex(_) )
        labels.append( param_names(_) )
    return indexes,labels