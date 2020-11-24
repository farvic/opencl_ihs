// Definições de OpenCL
#include <CL/cl.hpp>
// Asserções em C++
#include <cassert>
// Tempo em C++
#include <chrono>
// E/S em terminal
#include <iostream>
// E/S em arquivo
#include <fstream>
// Container de vetor
#include <vector>
// Importando pacotes
using namespace cl;
using namespace std;
using namespace std::chrono;
// Marcadores de tempo de alta resolução
microseconds cpu, opencl;
// Dispositivo da plaforma
Device device;
// Recursos do dispositivo
Context context;
// Programa do dispositivo
Program program;
// Dimensões das matrizes (64 MiB)
constexpr uint32_t N = (8 << 10) >> 2;
constexpr uint32_t K = (8 << 10) >> 2;
constexpr uint32_t M = (8 << 10) >> 2;
// Matrizes no ambiente de desenvolvimento
vector<int32_t> A(N * K, 5);
vector<int32_t> B(K * M, 8);
vector<int32_t> C_cpu(N * M, 1);
vector<int32_t> C_opencl(N * M, 2);
// Tamanho do item de trabalho na memória local
constexpr uint32_t MY_LOCAL_SIZE = 32;
// Inicialização do dispositivo
void init_device() {
    // Plataformas OpenCL
    vector<Platform> platforms;
    // Dispositivos da plataforma
    vector<Device> devices;
    // Obtendo plataformas disponíveis
    Platform::get(&platforms);
    // Checando se existem plataformas disponíveis
    assert(!platforms.empty());
    // Iterando sobre as plataformas
    for(uint32_t i = 0; i < platforms.size(); i++) {
        // Exibindo informações da plataforma
        cout << "--------------------------------------------------------------------------------" << endl;
        cout << "Platform ID = " << i << endl;
        cout << "--------------------------------------------------------------------------------" << endl;
        cout << "Version: " << platforms[i].getInfo<CL_PLATFORM_VERSION>() << endl;
        cout << "Name:    " << platforms[i].getInfo<CL_PLATFORM_NAME>() << endl;
        cout << "Vendor:  " << platforms[i].getInfo<CL_PLATFORM_VENDOR>() << endl;
    }
    // Selecionando plataforma
    uint32_t pindex = 0;
    cout << "Choose your plaform [0 - " << platforms.size() - 1 << "]: ";
    cin >> pindex;
    // Checando se indíce está nos limites
    assert(pindex >= 0 || pindex < platforms.size());
    // Obtendo os dispositivos da plataforma
    platforms[pindex].getDevices(CL_DEVICE_TYPE_ALL, &devices);
    // Checando se existem dispositivos disponíveis
    assert(!devices.empty());
    // Iterando sobre dispositivos
    for(uint32_t i = 0; i < devices.size(); i++) {
        // Exibindo informações da plataforma
        cout << "--------------------------------------------------------------------------------" << endl;
        cout << "Device ID = " << i << endl;
        cout << "--------------------------------------------------------------------------------" << endl;
        cout << "Platform:                    " << devices[i].getInfo<CL_DEVICE_PLATFORM>() << endl;
        cout << "Version:                     " << devices[i].getInfo<CL_DEVICE_VERSION>() << endl;
        cout << "Name:                        " << devices[i].getInfo<CL_DEVICE_NAME>() << endl;
        cout << "Vendor:                      " << devices[i].getInfo<CL_DEVICE_VENDOR>() << endl;
        cout << "Type:                        " << devices[i].getInfo<CL_DEVICE_TYPE>() << endl;
        cout << "Global memory size:          " << devices[i].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << endl;
        cout << "Local memory size:           " << devices[i].getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << endl;
        cout << "Max compute units:           " << devices[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << endl;
        cout << "Max work item size:          " << "[" << devices[i].getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[0] << ", " << devices[i].getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[1] << ", " << devices[i].getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[2] << "]" << endl;
        cout << "Max work group size:         " << devices[i].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << endl;
    }
    // Selecionando dispositivo
    uint32_t dindex = 0;
    cout << "Choose your device [0 - " << devices.size() - 1 << "]:  ";
    cin >> dindex;
    // Checando se indíce está nos limites
    assert(dindex >= 0 || dindex < devices.size());
    // Setando dispositivo
    device = devices[dindex];
}
// Compilação da função do núcleo
void compile_kernel(string name) {
    // Abertura e leitura do arquivo do núcleo
    ifstream file(name);
    string kernel(istreambuf_iterator<char>(file), (istreambuf_iterator<char>()));
    // Obtendo código fonte do núcleo
    Program::Sources source(1, make_pair(kernel.c_str(), kernel.length()));
    // Obtendo recursos do dispositivo
    context = Context(device);
    // Setando programa do dispositivo
    program = Program(context, source);
    // Compilando programa para o dispositivo
    cl_int status = program.build(string("-DMY_LOCAL_SIZE=" + to_string(MY_LOCAL_SIZE)).c_str());
    // Exibindo informações de compilação
    if(status != CL_BUILD_SUCCESS) cout << "Build error log:             " << status << endl << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
    // Checando se a compilação foi bem sucedida
    assert(status == CL_BUILD_SUCCESS);
}
// Multiplicação de matrizes (CPU)
void mulmat_cpu(int32_t* A, int32_t* B, int32_t* C, const uint32_t n, const uint32_t k, const uint32_t m) {
    // Iterando nas linhas de A
    for(uint32_t i = 0; i < n; i++) {
        // Iterando nas colunas de B
        for(uint32_t j = 0; j < m; j++) {
            // Valor da multiplicação
            int32_t sum = 0;
            // Iterando nas linhas de A e colunas de B
            for(uint32_t x = 0; x < k; x++) {
                // Calculando produto linha por coluna
                sum = sum + (A[i * k + x] * B[x * m + j]);
            }
            // Atribuindo resultado em C
            C[i * m + j] = sum;
        }
    }
}
// Multiplicação de matrizes (OpenCL)
void mulmat1_opencl(int32_t* A, int32_t* B, int32_t* C, const uint32_t n, const uint32_t k, const uint32_t m) {
    // Tempo de início
    high_resolution_clock::time_point delta = high_resolution_clock::now();
    // Compilando núcleo
    compile_kernel("mulmat2.cl");
    // Tempo de compilação do núcleo
    cout << "--------------------------------------------------------------------------------" << endl;
    cout << "Kernel compilation:          " << duration_cast<microseconds>(high_resolution_clock::now() - delta).count() << " ms" << endl;
    // Tempo de início
    delta = high_resolution_clock::now();
    // Alocando a memória no dispositivo
    Buffer buffer_A(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, N * K * sizeof(int32_t), A);
    Buffer buffer_B(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, K * M * sizeof(int32_t), B);
    Buffer buffer_C(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, N * M * sizeof(int32_t));
    // Tempo de alocação dos vetores
    microseconds delay = duration_cast<microseconds>(high_resolution_clock::now() - delta);
    cout << "Buffer allocation:           " << delay.count() << " ms (" << ((((N * K) + (K * M)) * sizeof(int32_t)) / ((double)(delay.count()))) << " MiB/s)" << endl;
    // Tempo de início
    delta = high_resolution_clock::now();
    // Ajustando parâmetros do núcleo
    Kernel kernel(program, "mulmat2_opencl");
    kernel.setArg(0, buffer_A);
    kernel.setArg(1, buffer_B);
    kernel.setArg(2, buffer_C);
    kernel.setArg(3, sizeof(uint32_t), &n);
    kernel.setArg(4, sizeof(uint32_t), &k);
    kernel.setArg(5, sizeof(uint32_t), &m);
    // Tempo de início
    delta = high_resolution_clock::now();
    // Criando fila de comandos do dispositivo
    CommandQueue queue = CommandQueue(context, device);
    // Tempo de instanciação da fila de comandos
    cout << "Queue instantiation:         " << duration_cast<microseconds>(high_resolution_clock::now() - delta).count() << " ms" << endl;
    // Tempo de início
    delta = high_resolution_clock::now();
    // Enfileirando comando com dimensão global N x M e local MY_LOCAL_SIZE x MY_LOCAL_SIZE
    queue.enqueueNDRangeKernel(kernel, NullRange, NDRange(N, M), NDRange(MY_LOCAL_SIZE, MY_LOCAL_SIZE));
    // Tempo de execução do núcleo no dispositivo
    cout << "Kernel execution:            " << duration_cast<microseconds>(high_resolution_clock::now() - delta).count() << " ms" << endl;
    // Tempo de início
    delta = high_resolution_clock::now();
    // Leitura dos resultados do dispositivo para o ambiente de desenvolvimento
    queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, N * M * sizeof(int32_t), C);
    // Tempo de leitura dos resultados do dispositivo
    delay = duration_cast<microseconds>(high_resolution_clock::now() - delta);
    cout << "Buffer read:                 " << delay.count() << " ms (" << ((N * M * sizeof(int32_t)) / ((double)(delay.count()))) << " MiB/s)" << endl;
    // Finalizando a fila de comandos
    queue.finish();
}
// Checando resultados (CPU x OpenCL)
bool check_error() {
    // Status de erro
    bool status = false;
    // Iterando sobre vetores
    for(uint32_t i = 0; i < N * M && !status; i++) {
        // Comparando vetores
        status = C_cpu[i] != C_opencl[i];
    }
    // Retornando status
    return status;
}
// Função principal
int main() {
    // Execução em CPU
    high_resolution_clock::time_point delta = high_resolution_clock::now();
    mulmat_cpu(A.data(), B.data(), C_cpu.data(), N, K, M);
    cpu = duration_cast<microseconds>(high_resolution_clock::now() - delta);
    // Execução em OpenCL
    init_device();
    delta = high_resolution_clock::now();
    mulmat1_opencl(A.data(), B.data(), C_opencl.data(), N, K, M);
    opencl = duration_cast<microseconds>(high_resolution_clock::now() - delta);
    // Checando resultados
    assert(check_error() == false);
    // Exibindo resultados da execução
    cout << "--------------------------------------------------------------------------------" << endl;
    cout << "Performance report" << endl;
    cout << "--------------------------------------------------------------------------------" << endl;
    cout << "CPU:                         " << cpu.count() << " ms" << endl;
    cout << "OpenCL:                      " << opencl.count() << " ms" << endl;
    cout << "CPU x OpenCL:                " << ((double)(opencl.count()) / (double)(cpu.count())) << "x" << endl;
    cout << "OpenCL x CPU:                " << ((double)(cpu.count()) / (double)(opencl.count())) << "x" << endl;
    // Retornando sucesso
    return 0;
}