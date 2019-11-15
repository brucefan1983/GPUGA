/*
    Purpose:
        Converting DFT data to train.in for GPUGA
    Compile:
        g++ -O3 main.cpp
    Run:
        ./a.out < input.txt
    Author:
        Zheyong Fan <brucenju(at)gmail.com>
*/


#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>


#define PRINT_ERROR(count, n)                              \
{                                                          \
    if (count != n)                                        \
    {                                                      \
        fprintf(stderr, "Reading error:\n");               \
        fprintf(stderr, "    File: %s\n", __FILE__);       \
        fprintf(stderr, "    Line: %d\n", __LINE__);       \
        exit(1);                                           \
    }                                                      \
}


FILE *my_fopen(const char *filename, const char *mode)
{
    FILE *fid = fopen(filename, mode);
    if (fid == NULL) 
    {
        printf ("Failed to open %s!\n", filename);
        printf ("%s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    return fid;
}


// modify these before compiling:
const int MAX_NC = 1000;          // Do you have more than 1000 configurations?
const int CELL_SIZE = 4;          // cell size of energy/virial configurations
const int NA = 4;                 // number of cells in direction a
const int NB = 4;                 // number of cells in direction b
const int NC = 3;                 // number of cells in direction c
const int NABC = NA * NB * NC;    // total number of cells
const bool STRESS_IN_KBAR = true; // true for stress in kbar from VASP
const float DELTA_E = 1.31684;    // energy shift (eV/atom)

FILE *fid_train;
char dft_files[MAX_NC][200];
int Nc_force, Nc_energy, Nc;
int Na[MAX_NC];
float box[9];

void get_Nc(void);
void get_files(void);
void get_Na(void);
void get_force(void);
void get_energy_and_virial(void);


int main(int argc, char *argv[])
{
    fid_train = my_fopen("train.in", "w");

    get_Nc();
    get_files();
    get_Na();
    get_force();
    get_energy_and_virial();

    fclose(fid_train);
    return EXIT_SUCCESS;
}


void get_Nc(void)
{
    int count = scanf("%d%d", &Nc_force, &Nc_energy);
    PRINT_ERROR(count, 2);
    Nc = Nc_force + Nc_energy;
    fprintf(fid_train, "%d %d\n", Nc, Nc_force);
}


void get_files(void)
{
    for (int n = 0; n < Nc; ++n)
    {
        int count = scanf("%s", dft_files[n]);
        PRINT_ERROR(count, 1);
    }
}


void get_Na(void)
{
    for (int n = 0; n < Nc; ++n)
    {
        FILE *fid_in = my_fopen(dft_files[n], "r");
        int count = fscanf(fid_in, "%d", &Na[n]);
        PRINT_ERROR(count, 1);
        if (n >= Nc_force) Na[n] *= NABC;
        fprintf(fid_train, "%d\n", Na[n]);
        fclose(fid_in);
    }
}


void get_force(void)
{
    for (int n = 0; n < Nc_force; ++n)
    {
        FILE *fid_in = my_fopen(dft_files[n], "r");
        int count = fscanf(fid_in, "%d", &Na[n]);
        for (int k = 0; k < 9; ++k)
        {
            count = fscanf(fid_in, "%f", &box[k]);
            PRINT_ERROR(count, 1);
            fprintf(fid_train, "%g ", box[k]);
        }
        fprintf(fid_train, "\n");

        int type;
        float r[3], f[3];
        for (int na = 0; na < Na[n]; ++na)
        {
            count = fscanf
            (
                fid_in, "%d%f%f%f%f%f%f", &type, &r[0], &r[1], &r[2],
                &f[0], &f[1], &f[2]
            );
            PRINT_ERROR(count, 7);
            fprintf
            (
                fid_train, "%d %g %g %g %g %g %g\n", type, r[0], r[1], r[2], 
                f[0], f[1], f[2]
            );
        }

        fclose(fid_in);
    }
}


void get_energy_and_virial(void)
{
    for (int n = Nc_force; n < Nc; ++n)
    {
        FILE *fid_in = my_fopen(dft_files[n], "r");
        int count = fscanf(fid_in, "%d", &Na[n]);

        float energy, virial[6];
        count = fscanf
        (
            fid_in, "%f%f%f%f%f%f%f", &energy, &virial[0], &virial[1], 
            &virial[2], &virial[3], &virial[4], &virial[5]
        );
        PRINT_ERROR(count, 7);

        for (int k = 0; k < 9; ++k)
        {
            count = fscanf(fid_in, "%f", &box[k]);
            PRINT_ERROR(count, 1);
        }

        energy = ((energy / Na[n]) + DELTA_E) * Na[n] * NABC;
        
        for (int d = 0; d < 6; ++d) virial[d] *= NABC;
        if (STRESS_IN_KBAR)
        {
            float vol = box[0] * (box[4] * box[8] - box[5] * box[7])
                      + box[1] * (box[5] * box[6] - box[3] * box[8])
                      + box[2] * (box[3] * box[7] - box[4] * box[6]);
            for (int d = 0; d < 6; ++d) virial[d] *= vol / 1602.18;
        }

        fprintf
        (
            fid_train, "%g %g %g %g %g %g %g\n", energy, virial[0], virial[1], 
            virial[2], virial[3], virial[4], virial[5]
        );
        for (int k = 0; k < 3; ++k) fprintf(fid_train, "%g ", box[k] * NA);
        for (int k = 3; k < 6; ++k) fprintf(fid_train, "%g ", box[k] * NB);
        for (int k = 6; k < 9; ++k) fprintf(fid_train, "%g ", box[k] * NC);
        fprintf(fid_train, "\n");

        int type[CELL_SIZE];
        float r[CELL_SIZE][3];
        for (int m = 0; m < Na[n]; ++m)
        {
            count = fscanf
            (fid_in, "%d%f%f%f", &type[m], &r[m][0], &r[m][1], &r[m][2]);
            PRINT_ERROR(count, 4);
        }
        for (int na = 0; na < NA; ++na)
        for (int nb = 0; nb < NB; ++nb)
        for (int nc = 0; nc < NC; ++nc)
        {
            float dx = box[0] * na + box[3] * nb + box[6] * nc;
            float dy = box[1] * na + box[4] * nb + box[7] * nc;
            float dz = box[2] * na + box[5] * nb + box[8] * nc;
            for (int m = 0; m < Na[n]; ++m)
            {
                fprintf
                (
                    fid_train, "%d %g %g %g\n", type[m], 
                    r[m][0] + dx, r[m][1] + dy, r[m][2] + dz
                );
            }
        }

        fclose(fid_in);
    }
}


