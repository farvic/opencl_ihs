/**
 *  Adição vetorial com OpenCL
 *  @param a Entrada a
 *  @param b Entrada b
 *  @param c Saída c
 */
 __kernel void add_opencl(__global int* a, __global int* b, __global int* c){
     // Índice do vetor
     unsigned int i = get_global_id(0);
     // c = a + b
     c[i] = a[i] + b[i];
 }