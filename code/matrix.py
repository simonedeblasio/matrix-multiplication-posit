##PYTHON CODE FOR MATRIX MULTIPLICATION m1 x m2
import softposit as sp
import softfloat as sf
import random as random
import numpy as np
import csv

# Standard size
matrix_size = (2, 2)


def dostuff(size):
    # Float 64
    global matrix_size
    matrix_size = size

    m1float64 = fillWithRandom()
    m2float64 = fillWithRandom()
    resultfloat64 = fillWithZeros("float64")

    # Posit 32
    m1posit32 = fillWithZeros("posit32")
    m2posit32 = fillWithZeros("posit32")
    resultposit32 = fillWithZeros("posit32")

    # Float 32
    m1float32 = fillWithZeros("float32")
    m2float32 = fillWithZeros("float32")
    resultfloat32 = fillWithZeros("float32")

    # Convert from random Float 64 to Posit 32
    convertFromFloat64Matrix(m1float64, m1posit32, "posit32")
    convertFromFloat64Matrix(m2float64, m2posit32, "posit32")

    # Convert from random Float 64 to Float 32
    convertFromFloat64Matrix(m1float64, m1float32, "float32")
    convertFromFloat64Matrix(m2float64, m2float32, "float32")


    # Floating point multiplication
    matrixMultiplication(m1float64, m2float64, resultfloat64, "float64")
    matrixMultiplication(m1posit32, m2posit32, resultposit32, "posit32")
    matrixMultiplication(m1float32, m2float64, resultfloat32, "float32")

    return [sumDiffOfMatrixes(resultposit32, resultfloat64), sumDiffOfMatrixes(resultfloat32, resultfloat64)]


def convertFromFloat64Matrix(randomMatrix, targetMatrix, floatFormat):
    if floatFormat == "posit32":
        for j in range(len(randomMatrix)):
            for i in range(len(randomMatrix[0])):
                targetMatrix[j][i] = sp.posit32(randomMatrix[j][i])
    elif floatFormat == "float32":
        for j in range(len(randomMatrix)):
            for i in range(len(randomMatrix[0])):
                targetMatrix[j][i] = sf.float32(randomMatrix[j][i])


def fillWithZeros(floatFormat):
    if floatFormat == "float64":
        return [[0 for i in range(matrix_size[0])] for j in range(matrix_size[1])]
    elif floatFormat == "posit32":
        return [[sp.posit32(0) for i in range(matrix_size[0])] for j in range(matrix_size[1])] 
    elif floatFormat == "float32":
        return [[sf.float32(0) for i in range(matrix_size[0])] for j in range(matrix_size[1])] 


def fillWithRandom():
    return [[random.random() for i in range(matrix_size[0])] for j in range(matrix_size[1])] 


def matrixMultiplication(m1, m2, result, floatFormat):
    if floatFormat == "float64":
        for i in range(len(m1)):
            # iterate through columns of m2
            for j in range(len(m2[0])):
                # iterate through rows of m2
                for k in range(len(m2)):
                    result[i][j] += m1[i][k] * m2[k][j]

    else:
        for i in range(len(m1)):
            # iterate through columns of m2
            for j in range(len(m2[0])):
                # iterate through rows of m2
                for k in range(len(m2)):
                    result[i][j] = result[i][j].fma(m1[i][k], m2[k][j])


def sumDiffOfMatrixes(m1, m2):
    diff = [[0 for i in range(matrix_size[0])] for j in range(matrix_size[1])]
    for j in range(matrix_size[0]):
        for i in range(matrix_size[1]):
            diff[j][i] = abs(m1[j][i] - m2[j][i])

    return np.sum(diff)


with open("data2x2.csv", "a", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["index", "Posit32 error", "Float32 error"])
    for x in range(100):
        result = dostuff((2, 2))
        writer.writerow([x, result[0], result[1]])
