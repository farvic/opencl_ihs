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
// Tamanho dos vetores (64 MiB)
constexpr uint32_t N = (64 << 20) >> 2;
// Vetores no ambiente de desenvolvimento
vector<int32_t> a(N, 5);
vector<int32_t> b(N, 8);
vector<int32_t> c_cpu(N, 1);
vector<int32_t> c_opencl(N, 2);
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
    cl_int status = program.build();
    // Exibindo informações de compilação
    if(status != CL_BUILD_SUCCESS) cout << "Build error log:             " << status << endl << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
    // Checando se a compilação foi bem sucedida
    assert(status == CL_BUILD_SUCCESS);
}
// Adição de vetores (CPU)
void add_cpu(int32_t* a, int32_t* b, int32_t* c) {
    // Iterando de 0 até N
    for(uint32_t i = 0; i < N; i++) {
        // c = a + b
        c[i] = a[i] + b[i];
    }
}
// Adição de vetores (OpenCL)
void add_opencl(int32_t* a, int32_t* b, int32_t* c) {
    // Tempo de início
    high_resolution_clock::time_point delta = high_resolution_clock::now();
    // Compilando núcleo
    compile_kernel("add.cl");
    // Tempo de compilação do núcleo
    cout << "--------------------------------------------------------------------------------" << endl;
    cout << "Kernel compilation:          " << duration_cast<microseconds>(high_resolution_clock::now() - delta).count() << " ms" << endl;
    // Tempo de início
    delta = high_resolution_clock::now();
    // Alocando a memória no dispositivo
    Buffer buffer_a(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, N * sizeof(int32_t), a);
    Buffer buffer_b(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, N * sizeof(int32_t), b);
    Buffer buffer_c(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, N * sizeof(int32_t));
    // Tempo de alocação dos vetores
    microseconds delay = duration_cast<microseconds>(high_resolution_clock::now() - delta);
    cout << "Buffer allocation:           " << delay.count() << " ms (" << ((2 * N * sizeof(int32_t)) / ((double)(delay.count()))) << " MiB/s)" << endl;
    // Tempo de início
    delta = high_resolution_clock::now();
    // Ajustando parâmetros do núcleo
    Kernel kernel(program, "add_opencl");
    kernel.setArg(0, buffer_a);
    kernel.setArg(1, buffer_b);
    kernel.setArg(2, buffer_c);
    // Tempo de início
    delta = high_resolution_clock::now();
    // Criando fila de comandos do dispositivo
    CommandQueue queue = CommandQueue(context, device);
    // Tempo de instanciação da fila de comandos
    cout << "Queue instantiation:         " << duration_cast<microseconds>(high_resolution_clock::now() - delta).count() << " ms" << endl;
    // Tempo de início
    delta = high_resolution_clock::now();
    // Enfileirando comando com dimensão global N
    queue.enqueueNDRangeKernel(kernel, NullRange, NDRange(N));
    // Tempo de execução do núcleo no dispositivo
    cout << "Kernel execution:            " << duration_cast<microseconds>(high_resolution_clock::now() - delta).count() << " ms" << endl;
    // Tempo de início
    delta = high_resolution_clock::now();
    // Leitura dos resultados do dispositivo para o ambiente de desenvolvimento
    queue.enqueueReadBuffer(buffer_c, CL_TRUE, 0, N * sizeof(int32_t), c);
    // Tempo de leitura dos resultados do dispositivo
    delay = duration_cast<microseconds>(high_resolution_clock::now() - delta);
    cout << "Buffer read:                 " << delay.count() << " ms (" << ((N * sizeof(int32_t)) / ((double)(delay.count()))) << " MiB/s)" << endl;
    // Finalizando a fila de comandos
    queue.finish();
}
// Checando resultados (CPU x OpenCL)
bool check_error() {
    // Status de erro
    bool status = false;
    // Iterando sobre vetores
    for(uint32_t i = 0; i < N && !status; i++) {
        // Comparando vetores
        status = c_cpu[i] != c_opencl[i];
    }
    // Retornando status
    return status;
}
// Função principal
int main() {
    // Execução em CPU
    high_resolution_clock::time_point delta = high_resolution_clock::now();
    add_cpu(a.data(), b.data(), c_cpu.data());
    cpu = duration_cast<microseconds>(high_resolution_clock::now() - delta);
    // Execução em OpenCL
    init_device();
    delta = high_resolution_clock::now();
    add_opencl(a.data(), b.data(), c_opencl.data());
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
