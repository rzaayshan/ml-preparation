
def sparse_matrix_multiplication(matrix_a, matrix_b):
    # Write your code here.
    result = []
    if len(matrix_a[0]) != len(matrix_b):
        return [[]]

    row = len(matrix_a)
    col = len(matrix_b[0])

    for i in range(0, row):
        r = []
        for row in range(0, col):
            r.append(0)
        result.append(r)

    for i in range(0, len(matrix_a)):
        for j in range(0, len(matrix_a[0])):
            for l in range(0, len(matrix_b[0])):
                result[i][l] += matrix_a[i][j] * matrix_b[j][l]

    return result

