#include <iostream>

int main()
{
    // Definicja wymiaru jądra
    size_t threshhold = 1;
    size_t R = 5;
    R += threshhold;
    
    size_t r = R;
    if (r < threshhold + 2) r = 0;
    else r -= threshhold + 2;
    size_t dim = 2 * R + 1;
    
    // Definicja buffora 2 wymiarowego
    bool **buffer = (bool **)calloc(dim, sizeof(bool *));
    for (size_t i = 0; i < dim; i++) {
        buffer[i] = (bool *)calloc(dim, sizeof(bool));
    }

    // Wypełnienie kernela
    for (size_t i = 0; i < dim; i++){
        for (size_t j = 0; j < dim; j++) {
            size_t distance = (i - dim / 2)*(i - dim / 2) + (j - dim / 2) * (j - dim / 2);
            if (distance <= R * R && distance > r * r) buffer[i][j] = 1;
        }
    }
    if (r == 0 && R > r) buffer[dim/2][dim/2] = 1;
    
    // Wypisanie elementów buffora
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            std::cout << buffer[i][j]<<" ";
        }
        std::cout << std::endl;
    }
    
    // Zwolnienie wewnętrznych bufforów
    for (size_t i = 0; i < dim; i++) {
        free(buffer[i]);
    }
    
    // Zwolnienie zewnętrznego buffora
    free(buffer);

    return 0;
}
