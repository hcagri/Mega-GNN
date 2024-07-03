import cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef assign_ports(int[:,:] arr, int[:] ports):
    cdef: 
        int j = 0, prev_v = -1
        int counter = 0
        dict mapping = {}
    
    for u, v in arr:
        if v != prev_v:
            counter = 0
            mapping = {}
            mapping[u] = counter
            ports[j] = counter
        else:
            if u in mapping:
                ports[j] = mapping[u]
            else:
                counter += 1
                mapping[u] = counter
                ports[j] = counter
        
        prev_v = v
        j+=1
    
    return ports