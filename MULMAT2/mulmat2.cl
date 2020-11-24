/**
 *  Multiplicação de matrizes com OpenCL
 *  @param A Matriz A
 *  @param B Matriz B
 *  @param C Matriz C
 *  @param n No. de linhas de A
 *  @param k No. de colunas de A e de linhas de B
 *  @param m No. de colunas de B
 */
__kernel void mulmat2_opencl(__global int* A, __global int* B, __global int* C, const unsigned int n, const unsigned int k, const unsigned int m) {
    // Sub-matrizes locais
    __local int lA[MY_LOCAL_SIZE][MY_LOCAL_SIZE];
    __local int lB[MY_LOCAL_SIZE][MY_LOCAL_SIZE];
    // Índices da linha e coluna
    unsigned int i = get_global_id(0), li = get_local_id(0);
    unsigned int j = get_global_id(1), lj = get_local_id(1);
    // Valor da multiplicação
    int sum = 0;
    // Iterando nas sub-matrizes
    for(unsigned int x = 0; x < k / MY_LOCAL_SIZE; x++) {
        // Copiando sub-matrizes para memória local
        lA[li][lj] = A[(i * k) + ((x * MY_LOCAL_SIZE) + lj)];
        lB[li][lj] = B[j + ((x * MY_LOCAL_SIZE + lj) * m)];
        // Sincronização dos itens de trabalho
        barrier(CLK_LOCAL_MEM_FENCE);
        // Iterando sobre linhas de A e colunas de B (local)
        for(unsigned int y = 0; y < MY_LOCAL_SIZE; y++) {
            // Calculando produto linha por coluna
            sum = sum + (lA[li][y] * lB[y][lj]);
        }
        // Sincronização dos itens de trabalho
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // Atribuindo resultado em C
    C[i * m + j] = sum;
}