void moving_average_float(float *values,
                          float *average,
                          int length,
                          int width)
{
    int i,j;
    float add_value;

    /* Insert zeros to 0th ~ (width-2)th elements */
    for( i = 0; i < width-1; i++ ) {
        average[i] = 0.0f;
    }

    /* Compute average of (width-1) ~ (length-1) elements */
    for( i = width-1; i < length; i++ ) {
        add_value = 0.0f;
        for( j = 0; j < width; j++ ) {
            add_value += values[i - j];
        }
        average[i] = add_value / (float)width;
    }
}

