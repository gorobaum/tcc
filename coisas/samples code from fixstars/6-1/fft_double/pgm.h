#ifndef _PGM_H_
#define _PGM_H_

#include <math.h>
#include <string.h>

#define PGM_MAGIC "P5"

typedef struct _pgm_t {
    int width;
    int height;
    unsigned char *buf;
} pgm_t;

int readPGM(pgm_t* pgm, const char* filename)
{
    char *token, *pc, *saveptr;
    char *buf;
    char *del = " \t\n";
    unsigned char *dot;

    long begin, end;
    int  filesize;    
    int i, w, h, luma, pixs;


    FILE* fp;
    if ((fp = fopen(filename, "rb"))==NULL) {
        fprintf(stderr, "Failed to open file\n");
        return -1;
    }

    fseek(fp, 0, SEEK_SET);
    begin = ftell(fp);
    fseek(fp, 0, SEEK_END);
    end = ftell(fp);
    filesize = (int)(end - begin);
    
    buf = (char*)malloc(filesize * sizeof(char));
    fseek(fp, 0, SEEK_SET);
    fread(buf, filesize * sizeof(char), 1, fp);
    
    fclose(fp);

    token = (char *)strtok_r(buf, del, &saveptr);
    if (strncmp(token, PGM_MAGIC, 2) != 0) {
        return -1;
    }
    
    token = (char *)strtok_r(NULL, del, &saveptr);
    if (token[0] == '#' ) {
        token = (char *)strtok_r(NULL, "\n", &saveptr);
        token = (char *)strtok_r(NULL, del, &saveptr);
    }
   
    w = strtoul(token, &pc, 10);
    token = (char *)strtok_r(NULL, del, &saveptr);
    h = strtoul(token, &pc, 10);
    token = (char *)strtok_r(NULL, del, &saveptr);
    luma = strtoul(token, &pc, 10);
    
    token = pc + 1;
    pixs = w * h; 
   
    pgm->buf = (unsigned char *)malloc(pixs * sizeof(unsigned char));
    
    dot = pgm->buf;

    for (i=0; i< pixs; i++, dot++) {
        *dot = *token++;
    }

    pgm->width = w;
    pgm->height = h;

    return 0;
}

int writePGM(pgm_t* pgm, const char* filename)
{
    int i, w, h, pixs;
    FILE* fp;
    unsigned char* dot;
    
    w = pgm->width;
    h = pgm->height;
    pixs = w * h;

    if ((fp = fopen(filename, "wb+")) ==NULL) {
        fprintf(stderr, "Failed to open file\n");
        return -1;
    }

    fprintf (fp, "%s\n%d %d\n255\n", PGM_MAGIC, w, h);

    dot = pgm->buf;
    
    for (i=0; i<pixs; i++, dot++) {
        putc((unsigned char)*dot, fp);
    }
    
    fclose(fp);

    return 0;
}

int normalizePGM(pgm_t* pgm, double* x)
{
    int i, j, w, h;
    
    w = pgm->width;
    h = pgm->height;
  
    pgm->buf = (unsigned char*)malloc(w * h * sizeof(unsigned char));
       
    double min = 0;
    double max = 0;
    for (i=0; i<h; i++) {
        for (j=0; j<w; j++) {
            if (max < x[i*w+j])
                max = x[i*w+j];
            if (min > x[i*w+j])
                min = x[i*w+j];
        }
    }

    for (i=0; i<h; i++) {
        for (j=0; j<w; j++) {
            if((max-min)!=0)
                pgm->buf[i*w+j] = (unsigned char)(255*(x[i*w+j]-min)/(max-min));
            else
                pgm->buf[i*w+j]= 0;
        }
    }

    return 0;
}

int destroyPGM(pgm_t* pgm)
{
    if (pgm->buf) {
        free(pgm->buf);
    }

    return 0;
}

#endif /* _PGM_H_ */
