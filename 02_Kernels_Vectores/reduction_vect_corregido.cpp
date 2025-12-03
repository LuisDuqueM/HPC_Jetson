#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include "../include/init_data.h"

//Semilla constante de reproducibilidad
const int FIXED_SEED=32;

//Función secuencial para reducción simple de vectores (suma de sus elementos.
float reduction_sum(float *A, long long n){
	float sum = 0.0f;
	for (long long i=0;i<n;i++){
		sum += A[i];
	}
	return sum;
}



int main(){
	std::cout<<"=== Benchmark Secuencial (CPU) ==="<<std::endl;

	//Lista de tamaños a evaluar (de 10^4 a 10^8)
	//10^8 floats = ~4000MB por vector. Total ~1.2GB RAM. Seguro para la Jetson.
	std::vector<long long> tamanos={
		10000,		//10e4
		100000,		//10e5
		1000000,	//10e6
		10000000,	//10e7
		100000000,	//10e8
	};

	//Iteramos sobre cada tamaño de N
	for(long long N:tamanos){
		std::cout<<"\nProcesando N = "<< N <<"elementos..."<<std::endl;
		//1.Asignación de memoria dinámica
		float *A=new float[N];
		//2.Inicialización de datos
		//Reiniciamos la semilla en cada iteración para consistencia
		srand(FIXED_SEED);
		//3.Medición del tiempo de computo
		auto start = std::chrono::high_resolution_clock::now();
		float suma = reduction_sum(A,N);
		auto end=std::chrono::high_resolution_clock::now();
		std::chrono::duration<double,std::milli>duration=end-start;
		//4.Reporte de resultados
		std::cout<<"Tiempo CPU: "<<duration.count()<<"ms"<<std::endl;
		//5.Liberación de la memoria
		delete[] A;
	}
	std::cout<<"\nBenchmark finalizado. "<<std::endl;
	return 0;
}


