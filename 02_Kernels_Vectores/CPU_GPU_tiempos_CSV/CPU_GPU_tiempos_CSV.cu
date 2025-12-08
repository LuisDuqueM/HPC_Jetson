// main.cu
//
// Suma de vectores y reducción (suma total de las componentes) en:
//   - CPU (versión secuencial)
//   - GPU (CUDA, en paralelo)
//
// El código mide tiempos por etapa:
//   - CPU: suma y reducción.
//   - GPU: H->D, kernel suma, kernel reducción, D->H, total aproximado.
// Además calcula speedups y escribe un CSV para graficar.

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <string>
#include <iomanip>
#include <sstream>
#include <fstream>   

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

const uint32_t FIXED_SEED = 32u;  // semilla fija para reproducibilidad
const int REPS = 30;              // repeticiones para promediar tiempos

// Estima memoria física disponible en host en bytes y avisa si el N es muy grande, entonces va a exceder la memoria RAM disponible
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

// Convierte una cantidad en bytes a una cadena con formato legible(KB, MB, GB).
std::string human_readable_bytes(uint64_t bytes) {
    const char* suf[] = { "B","KB","MB","GB","TB" };
    double val = static_cast<double>(bytes);
    int i = 0;
    while (val >= 1024.0 && i < 4) { val /= 1024.0; ++i; }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << val << " " << suf[i];
    return oss.str();
}

//Host
// Llenado de vector con valores aleatorios en rango [min_val, max_val] 

void init_vector_random(std::vector<float>& v,
    std::mt19937& rng,
    float               min_val = -1.0f,
    float               max_val = 1.0f) {
    std::uniform_real_distribution<float> dist(min_val, max_val);
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = dist(rng);
    }
}

// Función de suma de vectores (CPU)

void vec_sum_cpu(const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C) {
    size_t n = A.size();
    for (size_t i = 0; i < n; ++i) {
        C[i] = A[i] + B[i];
    }
}

// Reducción por suma de las componentes del vector C (CPU)
double reduce_sum_cpu(const std::vector<float>& v) {
    double acc = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        acc += static_cast<double>(v[i]);
    }
    return acc;
}

// Creación de funciones estadísticas para cálculo de media y desviación estándar de las repeticiones
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

// GPU
// Kernel de suma de vectores: C[idx] = A[idx] + B[idx]
__global__
void vec_sum_kernel(const float* A,
    const float* B,
    float* C,
    size_t       N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Kernel de reducción (suma) por bloques.
// data: vector de entrada en device
// blockSums: 1 valor por bloque (parcial)
// N: tamaño total
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
    extern __shared__ double sdata[];   // memoria compartida dinámica

    // Índices:
    unsigned int tid = threadIdx.x;                           // índice local dentro del bloque
    size_t globalIdx = blockIdx.x * blockDim.x + threadIdx.x; // índice global

    // Cargar elemento (o 0 si nos pasamos de N)
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

// Host
// Mide tiempo promedio del kernel de suma en GPU (solo kernel, no copias)
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

    // Warm-up
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

// Mide tiempo promedio del kernel de reducción en GPU
// y devuelve la suma total (sumando los parciales en CPU)
double run_reduce_sum_gpu(const float* dData,
    size_t       N,
    int          reps,
    double& result_out) {
    const int blockSize = 256;
    int gridSize = static_cast<int>((N + blockSize - 1) / blockSize);

    // Memoria en device para sumas parciales
    double* dBlockSums = nullptr;
    CUDA_CHECK(cudaMalloc(&dBlockSums, gridSize * sizeof(double)));

    // Memoria en host para copiar los parciales
    std::vector<double> hBlockSums(gridSize);

    size_t sharedBytes = blockSize * sizeof(double);

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

    // Copiamos sumas parciales al host
    CUDA_CHECK(cudaMemcpy(hBlockSums.data(),
        dBlockSums,
        gridSize * sizeof(double),
        cudaMemcpyDeviceToHost));

    // Sumamos parciales en CPU
    double total = 0.0;
    for (int i = 0; i < gridSize; ++i) total += hBlockSums[i];
    result_out = total;

    CUDA_CHECK(cudaFree(dBlockSums));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return avg_ms;
}

// main
int main() {
    std::cout << "=== Suma y Reducción: CPU vs GPU (CUDA) ===\n\n";
    // std::cout << "La cantidad de repeticiones a hacer son:" REPS "\n\n";

    uint64_t avail = get_available_memory_bytes();
    if (avail > 0) {
        std::cout << "Memoria disponible aproximada en host: "
            << human_readable_bytes(avail) << "\n\n";
    }

    // Abrimos archivo CSV para guardar resultados
    std::ofstream fout("resultados_tiempos.csv");
    fout << "N,"
        "Sequential addition,Secuential reduction,"
        "Host to Device,CUDA-based parallel addition,CUDA-based parallel reduction,Device to Host,Total CUDA time,"
        "Speedup addition,Speedup reduction,Overall speedup\n";

    // Lista de tamaños a evaluar
    std::vector<size_t> tamaños = {
        10000ul,        // 1e4
        100000ul,       // 1e5
        1000000ul,      // 1e6
        10000000ul,     // 1e7
        100000000ul     // 1e8 (ajustar si la RAM o VRAM son insuficientes)
    };

    // Loop principal
    for (size_t N : tamaños) {
        std::cout << "--------------------------------------------------\n";
        std::cout << "N = " << N << " elementos\n";

        // Memoria host requerida para A, B, C
        uint64_t bytes_needed_host = static_cast<uint64_t>(N) * sizeof(float) * 3ull;
        std::cout << "Memoria host requerida (3 vectores float): "
            << human_readable_bytes(bytes_needed_host) << "\n";

        if (avail > 0 && bytes_needed_host > avail) {
            std::cout << "ADVERTENCIA: La memoria requerida puede exceder la disponible en host.\n";
        }

        // Vectores en host
        std::vector<float> hA(N), hB(N), hC(N);

        // Inicialización reproducible
        std::mt19937 rng(FIXED_SEED);
        init_vector_random(hA, rng);
        init_vector_random(hB, rng);

        // ---------------- CPU: suma ----------------
        vec_sum_cpu(hA, hB, hC);  // warm-up

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

        // ---------------- CPU: reducción ----------------
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

        // GPU: reserva de memoria y copias H->D (medidas)
        float* dA = nullptr, * dB = nullptr, * dC = nullptr;

        uint64_t bytes_each = static_cast<uint64_t>(N) * sizeof(float);
        std::cout << "Memoria device requerida (3 vectores float): "
            << human_readable_bytes(bytes_each * 3ull) << "\n";

        CUDA_CHECK(cudaMalloc(&dA, bytes_each));
        CUDA_CHECK(cudaMalloc(&dB, bytes_each));
        CUDA_CHECK(cudaMalloc(&dC, bytes_each));

        // Eventos para medir H->D y D->H
        cudaEvent_t evt_h2d_start, evt_h2d_stop;
        cudaEvent_t evt_d2h_start, evt_d2h_stop;
        CUDA_CHECK(cudaEventCreate(&evt_h2d_start));
        CUDA_CHECK(cudaEventCreate(&evt_h2d_stop));
        CUDA_CHECK(cudaEventCreate(&evt_d2h_start));
        CUDA_CHECK(cudaEventCreate(&evt_d2h_stop));

        // H -> D
        CUDA_CHECK(cudaEventRecord(evt_h2d_start));

        CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes_each, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes_each, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaEventRecord(evt_h2d_stop));
        CUDA_CHECK(cudaEventSynchronize(evt_h2d_stop));

        float h2d_ms_f = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&h2d_ms_f, evt_h2d_start, evt_h2d_stop));
        double h2d_ms = static_cast<double>(h2d_ms_f);

        std::cout << "[GPU] H->D (A,B): " << h2d_ms << " ms\n";

        // GPU: suma de vectores (kernel)
        double avg_ms_gpu_sum = run_vec_sum_gpu(dA, dB, dC, N, REPS);
        std::cout << "[GPU] Suma (kernel only) -> Promedio: "
            << avg_ms_gpu_sum << " ms\n";

        // GPU: reducción del vetor C (kernel)
        double sumaC_gpu = 0.0;
        double avg_ms_gpu_red = run_reduce_sum_gpu(dC, N, REPS, sumaC_gpu);
        std::cout << "[GPU] Reducción C (kernel only) -> Promedio: "
            << avg_ms_gpu_red << " ms, sumaC = " << sumaC_gpu << "\n";

        // GPU: copia D->H 
        CUDA_CHECK(cudaEventRecord(evt_d2h_start));

        CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes_each, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaEventRecord(evt_d2h_stop));
        CUDA_CHECK(cudaEventSynchronize(evt_d2h_stop));

        float d2h_ms_f = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&d2h_ms_f, evt_d2h_start, evt_d2h_stop));
        double d2h_ms = static_cast<double>(d2h_ms_f);

        std::cout << "[GPU] D->H (C): " << d2h_ms << " ms\n";

        // Destruimos eventos
        CUDA_CHECK(cudaEventDestroy(evt_h2d_start));
        CUDA_CHECK(cudaEventDestroy(evt_h2d_stop));
        CUDA_CHECK(cudaEventDestroy(evt_d2h_start));
        CUDA_CHECK(cudaEventDestroy(evt_d2h_stop));

        // Comparación CPU vs GPU 
        double sumaA_cpu = reduce_sum_cpu(hA);
        double sumaB_cpu = reduce_sum_cpu(hB);
        double sumaC_cpu = reduce_sum_cpu(hC);  // C ya viene de GPU, pero debería ser igual a CPU

        double diff_C = sumaC_cpu - (sumaA_cpu + sumaB_cpu);
        double diff_gpu_cpu = sumaC_gpu - sumaC_cpu;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Verificación CPU:\n";
        std::cout << "  sum(A)_cpu = " << sumaA_cpu << "\n";
        std::cout << "  sum(B)_cpu = " << sumaB_cpu << "\n";
        std::cout << "  sum(C)_cpu = " << sumaC_cpu << "\n";
        std::cout << "  sum(C)_cpu - (sum(A)_cpu + sum(B)_cpu) = "
            << diff_C << " (debería ser cercano a 0)\n";
        std::cout << "Comparación GPU vs CPU (reducción C):\n";
        std::cout << "  sum(C)_gpu = " << sumaC_gpu << "\n";
        std::cout << "  sum(C)_gpu - sum(C)_cpu = "
            << diff_gpu_cpu << " (debería ser cercano a 0)\n";

        // Speedups y total GPU
        double cpu_total_ms = mu_cpu_sum + mu_cpu_red;
        double total_gpu_ms = h2d_ms + avg_ms_gpu_sum + avg_ms_gpu_red + d2h_ms;

        double speedup_sum_kernel = mu_cpu_sum / avg_ms_gpu_sum;
        double speedup_red_kernel = mu_cpu_red / avg_ms_gpu_red;
        double speedup_total_gpu = cpu_total_ms / total_gpu_ms;

        std::cout << "\nResumen tiempos (ms):\n";
        std::cout << "  CPU suma      : " << mu_cpu_sum << "\n";
        std::cout << "  CPU reducción : " << mu_cpu_red << "\n";
        std::cout << "  GPU H->D      : " << h2d_ms << "\n";
        std::cout << "  GPU suma kern : " << avg_ms_gpu_sum << "\n";
        std::cout << "  GPU red kern  : " << avg_ms_gpu_red << "\n";
        std::cout << "  GPU D->H      : " << d2h_ms << "\n";
        std::cout << "  GPU total     : " << total_gpu_ms << "\n";
        std::cout << "Speedups:\n";
        std::cout << "  speedup suma (kernel)      = " << speedup_sum_kernel << "\n";
        std::cout << "  speedup reducción (kernel) = " << speedup_red_kernel << "\n";
        std::cout << "  speedup total GPU          = " << speedup_total_gpu << "\n\n";

        // Escribir los resiltados en un CSV
        fout << std::fixed << std::setprecision(6);
        fout << N << ","
            << mu_cpu_sum << ","
            << mu_cpu_red << ","
            << h2d_ms << ","
            << avg_ms_gpu_sum << ","
            << avg_ms_gpu_red << ","
            << d2h_ms << ","
            << total_gpu_ms << ","
            << speedup_sum_kernel << ","
            << speedup_red_kernel << ","
            << speedup_total_gpu << "\n";

        // Liberar memoria device
        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }

    fout.close();
    std::cout << "Experimento finalizado. Resultados en 'resultados_tiempos.csv'.\n";
    return 0;
}
