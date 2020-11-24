/**
 *  Multiplicação de matrizes com OpenCL
 *  @param A Matriz A
 *  @param B Matriz B
 *  @param C Matriz C
 *  @param n No. de linhas de A
 *  @param k No. de colunas de A e de linhas de B
 *  @param m No. de colunas de B
 */
__kernel void mulmat1_opencl(__global int* A, __global int* B, __global int* C, const unsigned int n, const unsigned int k, const unsigned int m) {
    // Índices do linha e coluna
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);
    // Valor da multiplicação
    int sum = 0;
    // Iterando sobre linhas de A e colunas de B
    for(unsigned int x = 0; x < k; x++) {
        // Calculando produto linha por coluna
        sum = sum + (A[i * k + x] * B[x * m + j]);
    }
    // Atribuindo resultado em C
    C[i * m + j] = sum;
}