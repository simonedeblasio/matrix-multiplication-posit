import softposit as sp
import softfloat as sf
import numpy as np
import csv
import sys
import time as time


def main(size, interval):
    # Float 64
    m1float64 = fillWithRandom(interval, size)
    m2float64 = fillWithRandom(interval, size)
    resultfloat64 = fillWithZerosFloat64(size)

    # Posit 32
    m1posit32 = fillWithZerosPosit32(size)
    m2posit32 = fillWithZerosPosit32(size)
    resultposit32 = fillWithZerosPosit32(size)

    # Posit 32 using quire
    m1quire32 = fillWithZerosPosit32(size)
    m2quire32 = fillWithZerosPosit32(size)
    resultquire32 = fillWithZerosPosit32(size)

    # Float 32
    m1float32 = fillWithZerosFloat32(size)
    m2float32 = fillWithZerosFloat32(size)
    resultfloat32 = fillWithZerosFloat32(size)

    # Convert from random Float 64 to Posit 32
    convertFloat64ToPosit32(m1float64, m1posit32)
    convertFloat64ToPosit32(m2float64, m2posit32)

    # Convert from random Float 64 to Posit 32 using quire
    convertFloat64ToPosit32(m1float64, m1quire32)
    convertFloat64ToPosit32(m2float64, m2quire32)

    # Convert from random Float 64 to Float 32
    convertFloat64ToFloat32(m1float64, m1float32)
    convertFloat64ToFloat32(m2float64, m2float32)

    # Floating point multiplication
    matrixMultiplicationFloat64(m1float64, m2float64, resultfloat64)
    matrixMultiplicationPosit32(m1posit32, m2posit32, resultposit32)
    matrixMultiplicationQuire32(m1quire32, m2quire32, resultquire32)
    matrixMultiplicationFloat32(m1float32, m2float64, resultfloat32)

    # Calculate the difference
    posit32Difference = sumDiffOfMatrixes(resultposit32, resultfloat64)
    quire32Difference = sumDiffOfMatrixes(resultquire32, resultfloat64)
    float32Difference = sumDiffOfMatrixes(resultfloat32, resultfloat64)

    # Return the difference in an array
    return [posit32Difference, float32Difference, quire32Difference]
    #return [posit32Difference, float32Difference]


# Returns a matrix filled with Float64 zeros
def fillWithZerosFloat64(size):
    return [[0.0 for i in range(size)] for j in range(size)]


# Returns a matrix filled with Posit32 zeros suing softposit
def fillWithZerosPosit32(size):
    return [[sp.posit32(0.0) for i in range(size)] for j in range(size)]


# Returns a matrix filled with Float32 zeros suing softfloat
def fillWithZerosFloat32(size):
    return [[sf.float32(0.0) for i in range(size)] for j in range(size)]


# Returns a matrix filled with random numbers in the interval in Float64
def fillWithRandom(interval, size):
    return [[np.random.uniform(interval*-1, interval) for i in range(size)]
            for j in range(size)]


# Converts a Float64 matrix to Posit32 matrix using softposit for casting
def convertFloat64ToPosit32(Float64Matrix, Posit32Matrix):
    for j in range(len(Float64Matrix)):
        for i in range(len(Float64Matrix[0])):
            Posit32Matrix[j][i] = sp.posit32(Float64Matrix[j][i])


# Converts a Float64 matrix to Float32 matrix using softfloat for casting
def convertFloat64ToFloat32(Float64Matrix, Float32Matrix):
    for j in range(len(Float64Matrix)):
        for i in range(len(Float64Matrix[0])):
            Float32Matrix[j][i] = sf.float32(Float64Matrix[j][i])


# Matrix multiplies m1 x m2 and puts the result in Float64 the result matrix
def matrixMultiplicationFloat64(m1, m2, result):
    for i in range(len(m1)):
        # iterate through columns of m2
        for j in range(len(m2[0])):
            # iterate through rows of m2
            for k in range(len(m2)):
                result[i][j] += m1[i][k] * m2[k][j]
                

# Matrix multiplies m1 x m2 and puts the result in Posit32 the result matrix
def matrixMultiplicationPosit32(m1, m2, result):
    for i in range(len(m1)):
        # iterate through columns of m2
        for j in range(len(m2[0])):
            # iterate through rows of m2
            for k in range(len(m2)):
                result[i][j] = result[i][j].fma(m1[i][k], m2[k][j])
                #result[i][j] += m1[i][k] * m2[k][j]

# Similar to the function above, but uses a "quire" as an accumulator
# This should be way more exact, but also much slower
def matrixMultiplicationQuire32(m1, m2, result):
    q = sp.quire32()
    for i in range(len(m1)):
        # iterate through columns of m2
        for j in range(len(m2[0])):
            # iterate through rows of m2
            for k in range(len(m2)):
                q.qma(m1[i][k],m2[k][j])
                #result[i][j] += m1[i][k] * m2[k][j]
            result[i][j] = q.toPosit()
            q.clr()

# Matrix multiplies m1 x m2 and puts the result in Float32 the result matrix
def matrixMultiplicationFloat32(m1, m2, result):
    for i in range(len(m1)):
        # iterate through columns of m2
        for j in range(len(m2[0])):
            # iterate through rows of m2
            for k in range(len(m2)):
                result[i][j] = result[i][j].fma(m1[i][k], m2[k][j])
                #result[i][j] += m1[i][k] * m2[k][j]


# Returns the total difference between each element in the matrices m1 & m2
def sumDiffOfMatrixes(m1, m2):
    sum = np.longdouble(0)
    for j in range(len(m1)):
        for i in range(len(m1)):
            sum += abs(float(m1[j][i]-m2[j][i]))
    return sum


# Writes 1000 rows of results from running the main method
# Takes the random interval as input
# Writes to different files depending on the the interval and the size
# Loops through all sizes from 2^0 to 2^8
# (This function is a bit funny looking due to Flake8)
def writeToCsv(interval):
    for i in range(8):
        pow = 2**i
        # This writes return of main to csv files
        with open(
                 str(interval) + "data" + str(pow) + "x" + str(pow) + ".csv",
                 "a",
                 newline=''
                 ) as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                            [
                                "index",
                                "Posit32 error",
                                "Float32 error",
                                "Posit32 with quire error",
                                "Matrix size",
                                "Random interval"
                            ]
                           )
            start_time = time.time()
            for x in range(1000):
                random_interval = float(interval)
                result = main(pow, random_interval)
                writer.writerow(
                                [
                                    x,
                                    result[0],
                                    result[1],
                                    result[2],
                                    pow,
                                    random_interval
                                ]
                               )
            print("--- %s seconds ---" % (time.time() - start_time))


writeToCsv(sys.argv[1])
