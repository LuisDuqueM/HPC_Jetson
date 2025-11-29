#include <iostream>
#include <chrono>
#include "../include/init_data.h" //Importamos la función que creamos para crear los vectores


// Constantes 
const int N = 10000000; // Tamaño del vector
const int FIXED_SEED = 32; //Semilla para garantizar reproducibilidad de resultados.



// Función secuencial para sumar dos vectores
void vec_sum(float * A, float * B, float *C, int n){
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}


int main() {
    std::cout << "Comenzando suma de vectores secuencialmente..." << std::endl;

    // Asignando memoria para los vectores
    float *A = new float[N];
    float *B = new float[N];
    float *C = new float[N];

    // Inicializando la semilla para generación de números aleatorios
    srand(FIXED_SEED);

    // Ahora iniciamos los vectores A y B con los datos de la función del .h
    i_vector(A, N); //Vector A
    i_vector(B, N); //Vector B

    // Medimos el tiempo de ejecución de la suma de vectores
    auto start = std::chrono::high_resolution_clock::now();

    // Comenzamos la suma de vectores
    vec_sum(A, B, C, N);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Mostramos el tiempo de ejecución
    std::cout << " Elementos " << N << std::endl;
    std::cout << " Tiempo de ejecución: " << duration.count() << " ms" << std::endl;

    //Verificamos algunos resultados
    std::cout << "C[0] = " << C[0] << std::endl;
    std::cout << "C[N-1] = " << C[N-1] << std::endl;

    // Liberamos la memoria
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;



}




