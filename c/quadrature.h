#ifndef QUADRATURE_H_
#define QUADRATURE_H_

#define MAXPTS1D 3

typedef struct {
    int    n;           // number of quadrature points for this rule
    double w[MAXPTS1D], // weights (sum to 2)
           z[MAXPTS1D]; // locations in [-1,1]
} Quad1D;

static Quad1D gausslegendre[3] // first three rules
    = {{1,
       {0.0,NAN,NAN},
       {2.0,NAN,NAN}},
      {2,
       {-0.577350269189626,0.577350269189626,NAN},
       {1.0,1.0,NAN}},
      {3,
       {-0.774596669241483,0.0,0.774596669241483},
       {0.555555555555556,0.888888888888889,0.555555555555556}} };

#define MAXPTS2DT 4

typedef struct {
    int    n;              // number of quadrature points for this rule
    double w[MAXPTS2DT],   // weights (sum to 0.5)
           xi[MAXPTS2DT],  // locations: (xi,eta) in reference triangle
           eta[MAXPTS2DT]; //   with vertices (0,0), (1,0), (0,1)
} Quad2DTriangle;

FIXME

// quadrature points and weights
const int    Q[3] = {1, 3, 4};  // number of quadrature points
const double w[3][4]   = {{1.0/2.0,    NAN,       NAN,       NAN},
                          {1.0/6.0,    1.0/6.0,   1.0/6.0,   NAN},
                          {-27.0/96.0, 25.0/96.0, 25.0/96.0, 25.0/96.0}},
             xi[3][4]  = {{1.0/3.0,    NAN,       NAN,       NAN},
                          {1.0/6.0,    2.0/3.0,   1.0/6.0,   NAN},
                          {1.0/3.0,    1.0/5.0,   3.0/5.0,   1.0/5.0}},
             eta[3][4] = {{1.0/3.0,    NAN,       NAN,       NAN},
                          {1.0/6.0,    1.0/6.0,   2.0/3.0,   NAN},
                          {1.0/3.0,    1.0/5.0,   1.0/5.0,   3.0/5.0}};

#endif

