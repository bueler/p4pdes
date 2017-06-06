#ifndef QUADRATURE_H_
#define QUADRATURE_H_

#define MAXPTS1D 3

typedef struct {
    int    n;           // number of quadrature points for this rule
    double z[MAXPTS1D], // locations in [-1,1]
           w[MAXPTS1D]; // weights (sum to 2)
} Quad1D;

static const Quad1D gausslegendre[3]
    = {  {1,  // degree 1 rule has one point
          {0.0,                NAN,               NAN},
          {2.0,                NAN,               NAN}},
         {2,  // degree 2 rule has 2 points
          {-0.577350269189626, 0.577350269189626, NAN},
          {1.0,                1.0,               NAN}},
         {3,  // degree 3 rule has 3 points
          {-0.774596669241483, 0.0,               0.774596669241483},
          {0.555555555555556,  0.888888888888889, 0.555555555555556}} };

#define MAXPTS2DT 4

typedef struct {
    int    n;              // number of quadrature points for this rule
    double xi[MAXPTS2DT],  // locations: (xi,eta) in reference triangle
           eta[MAXPTS2DT], //   with vertices (0,0), (1,0), (0,1)
           w[MAXPTS2DT];   // weights (sum to 0.5)
} Quad2DTri;

static const Quad2DTri symmgauss[3]
    = {  {1,  // degree 1 rule has one point
          {1.0/3.0,    NAN,       NAN,       NAN},
          {1.0/3.0,    NAN,       NAN,       NAN},
          {1.0/2.0,    NAN,       NAN,       NAN}},
         {3,  // degree 2 rule has 3 points
          {1.0/6.0,    2.0/3.0,   1.0/6.0,   NAN},
          {1.0/6.0,    1.0/6.0,   2.0/3.0,   NAN},
          {1.0/6.0,    1.0/6.0,   1.0/6.0,   NAN}},
         {4,  // degree 3 rule had 4 points
          {1.0/3.0,    1.0/5.0,   3.0/5.0,   1.0/5.0},
          {1.0/3.0,    1.0/5.0,   1.0/5.0,   3.0/5.0},
          {-27.0/96.0, 25.0/96.0, 25.0/96.0, 25.0/96.0}}  };

#endif

