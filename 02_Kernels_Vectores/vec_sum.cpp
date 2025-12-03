#include <iostream>
#include <chrono>
#include <vector>
#include "../include/init_data.h" 

// Semilla constante para reproducibilidad
const int FIXED_SEED = 32; 

// Función secuencial para sumar dos vectores
// Usamos long long para evitar desbordamiento en índices si escalamos mucho
void vec_sum(float * A, float * B, float *C, long long n){
    for (long long i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    std::cout << "=== Benchmark Secuencial (CPU) ===" << std::endl;

    // Lista de tamaños a evaluar (de 10^4 a 10^8)
    // 10^8 floats = ~400MB por vector. Total ~1.2GB RAM. Seguro para la Jetson.
    std::vector<long long> tamaños = {
        10000,       // 1e4
        100000,      // 1e5
        1000000,     // 1e6
        10000000,    // 1e7
        100000000    // 1e8
    };

    // Iteramos sobre cada tamaño N
    for (long long N : tamaños) {
        
        std::cout << "\nProcesando N = " << N << " elementos..." << std::endl;

        // 1. Asignación de memoria dinámica
        float *A = new float[N];
        float *B = new float[N];
        float *C = new float[N];

        // 2. Inicialización de datos
        // Reiniciamos la semilla en cada iteración para consistencia
        srand(FIXED_SEED);
        
        // Llenamos los vectores usando la función compartida
        i_vector(A, N); 
        i_vector(B, N); 

        // 3. Medición del tiempo de cómputo
        auto start = std::chrono::high_resolution_clock::now();

        vec_sum(A, B, C, N);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;

        // 4. Reporte de resultados
        std::cout << "Tiempo CPU: " << duration.count() << " ms" << std::endl;

        // 5. Liberación de memoria (Crítico para no saturar la RAM en el bucle)
        delete[] A;
        delete[] B;
        delete[] C;
    }

    std::cout << "\nBenchmark finalizado." << std::endl;
    return 0;
}




