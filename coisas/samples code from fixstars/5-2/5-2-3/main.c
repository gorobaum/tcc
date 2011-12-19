#include <stdio.h>
#include <stdlib.h>

/* Read Stock data */
int stock_array1[] = {
    #include "stock_array1.txt"
};

/* Define width for the moving average */
#define WINDOW_SIZE (13)

int main(int argc, char *argv[])
{

    float *result;

    int data_num = sizeof(stock_array1) / sizeof(stock_array1[0]);
    int window_num = (int)WINDOW_SIZE;

    int i;
        
    /* Allocate space for the result */
    result = (float *)malloc(data_num*sizeof(float));

    /* Call the moving average function */
    moving_average(stock_array1,
                   result,
                   data_num,
                   window_num);

    /* Print result */
    for(i=0; i<data_num; i++) {
        printf( "result[%d] = %f\n", i, result[i] );
    }

    /* Deallocate memory */
    free(result);

    return 0;
}

