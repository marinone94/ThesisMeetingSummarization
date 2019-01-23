import numpy as np
import sys

class GraphHelp(object):
    """Help functions for graph propagation"""
    def Preprocess(Luu, Lss, Lus):
        E_11 = GraphHelp.remove_diag(Luu)
        E_22 = GraphHelp.remove_diag(Lss)
        E_12 = Lus
        E_21 = E_12.T
    
        if GraphHelp.check_size(E_11, E_22, E_12):
            sys.stderr.write("Error: The dimensions do not match.\n")
            exit()

        if GraphHelp.check_valid(E_11) or GraphHelp.check_valid(E_22) or GraphHelp.check_valid(E_12):
            sys.stderr.write("Error: The input weights have negative values.\n")
            exit()

        num1 = np.shape(E_11)[0]
        num2 = np.shape(E_22)[0]

        L_11 = GraphHelp.row_normalize(E_11)
        L_22 = GraphHelp.row_normalize(E_22)
        L_12 = GraphHelp.row_normalize(E_21)
        L_21 = GraphHelp.row_normalize(E_12)

        return [L_11, L_22, L_12, L_21, num1, num2]

    def remove_diag( inMtx ):
        num_row, num_col = np.shape(inMtx)
        for i in range(0, num_row):
            inMtx[i][i] = 0
        return inMtx

    def check_valid( inMtx ):
        # check whether there are negative weights
        if np.sum((inMtx<0).astype(np.int)) > 0:
            return 1
        return 0

    def keep_top( inMtx, K ):
        row, col = np.shape(inMtx)
        outMtx = np.zeros((row, col))
        for i in range(0, row):
            sortList = []
            for j in range(0, col):
                sortList.append((j, inMtx[i][j]))
            sortList = sorted(sortList, key=itemgetter(1), reverse=True)
            for k in range(0, K):
                j, s = sortList[k]
                outMtx[i][j] = s
        return outMtx

    def check_size( Mtx1, Mtx2, Mtx12 ):
        row1, col1 = np.shape(Mtx1)
        row2, col2 = np.shape(Mtx2)
        row12, col12 = np.shape(Mtx12)
        if row1 != col1 or row2 != col2 or row1 != row12 or row2 != col12:
            return 1
        return 0

    def row_normalize( inMtx ):
        sum = np.sum(inMtx, 1)
        return (inMtx/sum[:, np.newaxis]).T

    def output_file( outfile, score ):
        fout = open(outfile, "w")
        for i in range(0, len(score)):
            fout.write("%f\n" %score[i])
        fout.close()
        return

    def RemoveEmptyCols(m, squared=True):
        size = np.shape(m)[0]
        if squared:
            new_m = []
            for r in m:
                temp = []
                for el in r:
                    if el:
                        temp.append(el) 
                if temp:
                    new_m.append(temp)
            arr = np.array(new_m)
            return arr
        else:
            new_m = []
            for r in m:
                if np.sum(r): #check if first row is not empty
                    y = 0
                    for el in r: 
                        if el:
                            loc = []
                            for z in range(size):
                                loc.append(m[z][y])
                            new_m.append(loc)
                        y += 1
                    break
            arr = np.array(new_m)
            return np.transpose(arr)


