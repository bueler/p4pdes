#ifndef QUADRATURE_H_
#define QUADRATURE_H_

//STARTONEDIM
#define MAXPTS 3

typedef struct {
    int    n;          // number of quadrature points for this rule
    double xi[MAXPTS], // locations in [-1,1]
           w[MAXPTS];  // weights (sum to 2)
} Quad1D;

static const Quad1D gausslegendre[3]
    = {  {1,
          {0.0,                NAN,               NAN},
          {2.0,                NAN,               NAN}},
         {2,
          {-0.577350269189626, 0.577350269189626, NAN},
          {1.0,                1.0,               NAN}},
         {3,
          {-0.774596669241483, 0.0,               0.774596669241483},
          {0.555555555555556,  0.888888888888889, 0.555555555555556}} };
//ENDONEDIM

//STARTTRIANGLE
#define MAXPTS 4

typedef struct {
    int    n;           // number of quadrature points for this rule
    double xi[MAXPTS],  // locations: (xi,eta) in reference triangle
           eta[MAXPTS], //   with vertices (0,0), (1,0), (0,1)
           w[MAXPTS];   // weights (sum to 0.5)
} Quad2DTri;

static const Quad2DTri symmgauss[3]
    = {  {1,
          {1.0/3.0,    NAN,       NAN,       NAN},
          {1.0/3.0,    NAN,       NAN,       NAN},
          {1.0/2.0,    NAN,       NAN,       NAN}},
         {3,
          {1.0/6.0,    2.0/3.0,   1.0/6.0,   NAN},
          {1.0/6.0,    1.0/6.0,   2.0/3.0,   NAN},
          {1.0/6.0,    1.0/6.0,   1.0/6.0,   NAN}},
         {4,
          {1.0/3.0,    1.0/5.0,   3.0/5.0,   1.0/5.0},
          {1.0/3.0,    1.0/5.0,   1.0/5.0,   3.0/5.0},
          {-27.0/96.0, 25.0/96.0, 25.0/96.0, 25.0/96.0}}  };
//ENDTRIANGLE

#endif

