#ifndef INIT_DATA_H     // <-- 1. Guarda de encabezado (Inicio)
#define INIT_DATA_H     // <-- 2. Definición de la guarda

#include <cstdlib>
#include <ctime>   
#include <iostream> 

// Usamos inline para evitar el error de vinculación (linker error)
// NOTA: Eliminamos la semilla 'seed' de los argumentos
inline void i_vector(float *vector, int N) { 
    for (int i=0 ; i < N; i++){
        // Usamos static_cast para asegurar la división de punto flotante
        vector[i] = static_cast <float>(rand()) / RAND_MAX;
    }
}

#endif
