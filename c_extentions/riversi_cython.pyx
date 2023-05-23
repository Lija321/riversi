import numpy as np
cimport numpy as np

coefficients=(
    (20,-3,11,8,8,11,-3,20),
    (-3,-7,-4,1,1,-4,-7,-3),
    (11, -4, 2, 2, 2, 2, -4, 11),
    (8, 1, 2, -3, -3, 2, 1, 8),
    (8, 1, 2, -3, -3, 2, 1, 8),
    (11, -4, 2, 2, 2, 2, -4, 11),
    (-3, -7, -4, 1, 1, -4, -7, -3),
    (20, -3, 11, 8, 8, 11, -3, 20)
)

directions = (
    (-1, -1), (0, -1), (1, -1),
    (-1, 0), (1, 0),
    (-1, 1), (0, 1), (1, 1),
)

def heuristics(np.ndarray[np.int8_t, ndim=2] matrix):
    cdef int i,j,k
    cdef double parity=0,corners=0,corner_closeness=0,dynamic=0,frontier=0,score=0
    cdef int b_tiles=0,w_tiles=0,bf_tiles=0,wf_tiles=0
    for i in range(8):
        for j in range(8):
            dynamic+= coefficients[i][j] * matrix[i,j]
            if matrix[i,j]==1: b_tiles+=1
            elif matrix[i,j]==-1: w_tiles+=1
            if matrix[i,j]!=0:
              for k in range(8):
                   x=i+directions[k][0]
                   y=j+directions[k][0]
                   if (x >= 0 and x < 8 and y >= 0 and y < 8 and matrix[x,y] == 0):
                       if matrix[i,j]==1:
                           bf_tiles+=1
                       else:
                           wf_tiles+=1
                       break #nzm zasto

    if b_tiles>w_tiles:
        parity = 100 * b_tiles / (b_tiles + w_tiles)
    elif w_tiles>b_tiles:
        parity = -100 * w_tiles / (b_tiles + w_tiles)
    else:
        parity=0

    frontier = 0
    if bf_tiles+wf_tiles!=0:
        if bf_tiles>wf_tiles:
            frontier = -100 * bf_tiles / (bf_tiles + wf_tiles)
        elif wf_tiles>bf_tiles:
            frontier = 100 * wf_tiles / (bf_tiles + wf_tiles)


    cdef int c1=matrix[0,0]
    cdef int c2=matrix[0,7]
    cdef int c3=matrix[7,0]
    cdef int c4=matrix[7,7]

    corners=25*(c1+c2+c3+c4)
    if c1==0:
        corner_closeness += matrix[0, 1]
        corner_closeness += matrix[1, 1]
        corner_closeness += matrix[1, 0]
    if c2==0:
        corner_closeness += matrix[0, 6]
        corner_closeness += matrix[1, 6]
        corner_closeness += matrix[1, 7]
    if c3==0:
        corner_closeness += matrix[7, 1]
        corner_closeness += matrix[6, 1]
        corner_closeness += matrix[6, 0]
    if c4==0:
        corner_closeness += matrix[6, 7]
        corner_closeness += matrix[6, 6]
        corner_closeness += matrix[7, 6]
    corner_closeness*=-12.5

    score = (10 * parity) + (1000 * corners) + (382.026 * corner_closeness)+ (74.396 * frontier) + (10 * dynamic)
    return np.float32(score)