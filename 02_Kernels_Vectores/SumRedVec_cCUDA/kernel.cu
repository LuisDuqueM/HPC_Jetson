
// main.cu
//
// Suma de vectores y reducción (suma total) en:
//   - CPU (versión secuencial)
//   - GPU (versión paralela con CUDA)
//


#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <string>
#include <iomanip>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#else
#include <fstream>
#include <unistd.h>
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ---------------------------------------------------------
// Macro para checar errores de CUDA de forma cómoda
// ---------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::cerr << "CUDA error en " << #call << " : "                    \
                      << cudaGetErrorString(err__) << " (" << err__ << ")\n";  \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------
// Parámetros globales
// ---------------------------------------------------------

// Semilla fija para reproducibilidad
const uint32_t FIXED_SEED = 32u;

// Número de repeticiones para promediar tiempos
const int REPS = 20;

// ---------------------------------------------------------
// Utilidades de memoria (host)
// ---------------------------------------------------------

uint64_t get_available_memory_bytes() {
#ifdef _WIN32
    MEMORYSTATUSEX st;
    st.dwLength = sizeof(st);
    if (GlobalMemoryStatusEx(&st)) {
        return static_cast<uint64_t>(st.ullAvailPhys);
    }
    else {
        return 0;
    }
#else
    std::ifstream f("/proc/meminfo");
    if (!f.is_open()) return 0;
    std::string line;
    uint64_t memAvailableKb = 0;
    while (std::getline(f, line)) {
        if (line.rfind("MemAvailable:", 0) == 0) {
            std::istringstream iss(line);
            std::string key;
            uint64_t value;
            std::string unit;
            iss >> key >> value >> unit;
            memAvailableKb = value;
            break;
        }
    }
    f.close();
    return memAvailableKb * 1024ull;
#endif
}

std::string human_readable_bytes(uint64_t bytes) {
    const char* suf[] = { "B","KB","MB","GB","TB" };
    double val = static_cast<double>(bytes);
    int i = 0;
    while (val >= 1024.0 && i < 4) { val /= 1024.0; ++i; }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << val << " " << suf[i];
    return oss.str();
}

// Inicialización reproducible (host)


void init_vector_random(std::vector<float>& v,
    std::mt19937& rng,
    float               min_val = -1.0f,
    float               max_val = 1.0f) {
    std::uniform_real_distribution<float> dist(min_val, max_val);
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = dist(rng);
    }
}

// CPU: suma de vectores y reducción

void vec_sum_cpu(const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C) {
    size_t n = A.size();
    for (size_t i = 0; i < n; ++i) {
        C[i] = A[i] + B[i];
    }
}

// Reducción por suma en CPU
double reduce_sum_cpu(const std::vector<float>& v) {
    double acc = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        acc += static_cast<double>(v[i]);
    }
    return acc;
}

// Estadística simple para tiempos
double mean(const std::vector<double>& x) {
    double s = 0.0;
    for (double v : x) s += v;
    return s / static_cast<double>(x.size());
}
double stddev(const std::vector<double>& x, double mu) {
    double s = 0.0;
    for (double v : x) {
        double d = v - mu;
        s += d * d;
    }
    return std::sqrt(s / static_cast<double>(x.size()));
}


// GPU: kernels de suma y reducción


// Kernel de suma de vectores:
// Cada hilo procesa un índice: C[idx] = A[idx] + B[idx]
__global__
void vec_sum_kernel(const float* A,
    const float* B,
    float* C,
    size_t       N) {
    // Calculamos índice global del hilo
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Verificamos que esté dentro del rango
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Kernel de reducción por suma:
//   - data: vector de entrada en device
//   - blockSums: un valor por bloque (parcial)
//   - N: tamaño del vector
//
// Cada bloque:
//   1) Carga sus elementos en memoria compartida como double.
//   2) Hace un "árbol" de suma dentro del bloque.
//   3) El hilo 0 del bloque escribe la suma parcial en blockSums[blockIdx.x].
//
// Luego, en el host, terminamos de sumar esos parciales.
__global__
void reduce_sum_kernel(const float* data,
    double* blockSums,
    size_t       N) {
    // Memoria compartida dinámica (el tamaño se especifica al lanzar el kernel)
    extern __shared__ double sdata[];

    // Índices:
    unsigned int tid = threadIdx.x;                             // índice local dentro del bloque
    size_t globalIdx = blockIdx.x * blockDim.x + threadIdx.x;   // índice global

    // Cada hilo carga un elemento de data (o 0 si se pasa de N)
    double x = 0.0;
    if (globalIdx < N) {
        x = static_cast<double>(data[globalIdx]);
    }
    sdata[tid] = x;
    __syncthreads();

    // Reducción en árbol dentro del bloque
    // En cada iteración se reduce a la mitad la cantidad de elementos activos
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Al final, sdata[0] contiene la suma de todos los elementos que procesó el bloque
    if (tid == 0) {
        blockSums[blockIdx.x] = sdata[0];
    }
}



// Host: funciones helper para GPU


// Lanza el kernel de suma de vectores en la GPU y mide su tiempo (solo kernel)
double run_vec_sum_gpu(const float* dA,
    const float* dB,
    float* dC,
    size_t       N,
    int          reps) {
    const int blockSize = 256;
    int gridSize = static_cast<int>((N + blockSize - 1) / blockSize);

    // Creamos eventos CUDA para medir tiempo
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // "Warm-up" (una ejecución sin medir, opcional)
    vec_sum_kernel << <gridSize, blockSize >> > (dA, dB, dC, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Medición
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < reps; ++r) {
        vec_sum_kernel << <gridSize, blockSize >> > (dA, dB, dC, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // El tiempo devuelto lo dividimos entre reps para tener promedio por ejecución
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return static_cast<double>(ms) / static_cast<double>(reps);
}

// Lanza el kernel de reducción en la GPU y devuelve:
//   - tiempo promedio por lanzamiento (solo kernel)
//   - suma final (terminando la reducción en CPU sobre los parciales)
double run_reduce_sum_gpu(const float* dData,
    size_t       N,
    int          reps,
    double& result_out) {
    const int blockSize = 256;
    int gridSize = static_cast<int>((N + blockSize - 1) / blockSize);

    // Reservamos memoria en device para las sumas parciales (un double por bloque)
    double* dBlockSums = nullptr;
    CUDA_CHECK(cudaMalloc(&dBlockSums, gridSize * sizeof(double)));

    // Memoria en host para copiar esas sumas parciales
    std::vector<double> hBlockSums(gridSize);

    // Tamaño de memoria compartida (un double por hilo)
    size_t sharedBytes = blockSize * sizeof(double);

    // Eventos para medir tiempo
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up
    reduce_sum_kernel << <gridSize, blockSize, sharedBytes >> > (dData, dBlockSums, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Medición
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < reps; ++r) {
        reduce_sum_kernel << <gridSize, blockSize, sharedBytes >> > (dData, dBlockSums, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double avg_ms = static_cast<double>(ms) / static_cast<double>(reps);

    // Ahora copiamos las sumas parciales al host
    CUDA_CHECK(cudaMemcpy(hBlockSums.data(),
        dBlockSums,
        gridSize * sizeof(double),
        cudaMemcpyDeviceToHost));

    // Terminamos la reducción en CPU (suma de los parciales)
    double total = 0.0;
    for (int i = 0; i < gridSize; ++i) total += hBlockSums[i];
    result_out = total;

    CUDA_CHECK(cudaFree(dBlockSums));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return avg_ms;
}

// main: orquesta CPU + GPU

int main() {
    std::cout << "=== Suma y Reducción: CPU vs GPU (CUDA) ===\n\n";

    // Información de memoria disponible aproximada
    uint64_t avail = get_available_memory_bytes();
    if (avail > 0) {
        std::cout << "Memoria disponible aproximada en host: "
            << human_readable_bytes(avail) << "\n\n";
    }

    // Lista de tamaños N a evaluar
    std::vector<size_t> tamaños = {
        10000ul,        // 1e4
        100000ul,       // 1e5
        1000000ul,      // 1e6
        10000000ul,     // 1e7
        100000000ul     // 1e8  (ajusta según tu GPU / RAM)
    };

    // Recorremos cada N
    for (size_t N : tamaños) {
        std::cout << "--------------------------------------------------\n";
        std::cout << "N = " << N << " elementos\n";

        // Memoria requerida en host para A, B, C
        uint64_t bytes_needed_host = static_cast<uint64_t>(N) * sizeof(float) * 3ull;
        std::cout << "Memoria host requerida (3 vectores float): "
            << human_readable_bytes(bytes_needed_host) << "\n";

        if (avail > 0 && bytes_needed_host > avail) {
            std::cout << "ADVERTENCIA: La memoria requerida puede exceder la disponible en host.\n";
        }

        // Reservamos vectores en host
        std::vector<float> hA(N), hB(N), hC(N);

        // Inicialización reproducible
        std::mt19937 rng(FIXED_SEED);
        init_vector_random(hA, rng);
        init_vector_random(hB, rng);

        // ---------------- CPU: suma ----------------

        // Warm-up CPU
        vec_sum_cpu(hA, hB, hC);

        std::vector<double> tiempos_cpu_sum;
        tiempos_cpu_sum.reserve(REPS);

        for (int r = 0; r < REPS; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            vec_sum_cpu(hA, hB, hC);
            auto t1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> dt = t1 - t0;
            tiempos_cpu_sum.push_back(dt.count());
        }

        double mu_cpu_sum = mean(tiempos_cpu_sum);
        double sd_cpu_sum = stddev(tiempos_cpu_sum, mu_cpu_sum);

        std::cout << "[CPU] Suma -> Promedio: " << mu_cpu_sum
            << " ms, Desv. estándar: " << sd_cpu_sum << " ms\n";

        //  CPU: reducción 

        std::vector<double> tiempos_cpu_red;
        tiempos_cpu_red.reserve(REPS);
        double sumaC_cpu_ultima = 0.0;

        for (int r = 0; r < REPS; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            double sumaC = reduce_sum_cpu(hC);
            auto t1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> dt = t1 - t0;
            tiempos_cpu_red.push_back(dt.count());
            sumaC_cpu_ultima = sumaC;
        }

        double mu_cpu_red = mean(tiempos_cpu_red);
        double sd_cpu_red = stddev(tiempos_cpu_red, mu_cpu_red);

        std::cout << "[CPU] Reducción C -> Promedio: " << mu_cpu_red
            << " ms, Desv. estándar: " << sd_cpu_red
            << " ms (sumaC = " << sumaC_cpu_ultima << ")\n";

        
        // GPU: reserva de memoria y copia de datos
    

        float* dA = nullptr, * dB = nullptr, * dC = nullptr;

        // Memoria en device para A, B, C
        uint64_t bytes_each = static_cast<uint64_t>(N) * sizeof(float);
        std::cout << "Memoria device requerida (3 vectores float): "
            << human_readable_bytes(bytes_each * 3ull) << "\n";

        CUDA_CHECK(cudaMalloc(&dA, bytes_each));
        CUDA_CHECK(cudaMalloc(&dB, bytes_each));
        CUDA_CHECK(cudaMalloc(&dC, bytes_each));

        // Copiamos A y B a device
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes_each, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes_each, cudaMemcpyHostToDevice));

        
        // GPU: suma de vectores
        

        double avg_ms_gpu_sum = run_vec_sum_gpu(dA, dB, dC, N, REPS);

        std::cout << "[GPU] Suma (kernel only) -> Promedio: "
            << avg_ms_gpu_sum << " ms\n";

        // Si queremos comparar resultados numéricos, copiamos C de device a host
        CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes_each, cudaMemcpyDeviceToHost));

        
        // GPU: reducción de C
        

        double sumaC_gpu = 0.0;
        double avg_ms_gpu_red = run_reduce_sum_gpu(dC, N, REPS, sumaC_gpu);

        std::cout << "[GPU] Reducción C (kernel only) -> Promedio: "
            << avg_ms_gpu_red << " ms, sumaC = " << sumaC_gpu << "\n";

        
        // Comparación CPU vs GPU (numérica)
        

        double sumaA_cpu = reduce_sum_cpu(hA);
        double sumaB_cpu = reduce_sum_cpu(hB);
        double sumaC_cpu = reduce_sum_cpu(hC);

        double diff_C = sumaC_cpu - (sumaA_cpu + sumaB_cpu);
        double diff_gpu_cpu = sumaC_gpu - sumaC_cpu;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Verificación CPU:\n";
        std::cout << "  sum(A)_cpu = " << sumaA_cpu << "\n";
        std::cout << "  sum(B)_cpu = " << sumaB_cpu << "\n";
        std::cout << "  sum(C)_cpu = " << sumaC_cpu << "\n";
        std::cout << "  sum(C)_cpu - (sum(A)_cpu + sum(B)_cpu) = "
            << diff_C << "  (debería ser cercano a 0)\n";

        std::cout << "Comparación GPU vs CPU (reducción C):\n";
        std::cout << "  sum(C)_gpu = " << sumaC_gpu << "\n";
        std::cout << "  sum(C)_gpu - sum(C)_cpu = "
            << diff_gpu_cpu << "  (debería ser cercano a 0)\n\n";

        
        // Liberar memoria en device
        
        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }

    std::cout << "Experimento finalizado.\n";
    return 0;
}

