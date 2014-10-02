#!/bin/bash

# Copyright (C) 2014 Ed Bueler

make c2triangle &> /dev/null
make c2poisson &> /dev/null

triangle -pqa0.1024 square |'grep' "Mesh vertices"
./c2triangle -f square.1 &> /dev/null
./c2poisson -f square.1 -ksp_rtol 1.0e-10 |'grep' "check II"

triangle -rpqa0.0256 square.1 |'grep' "Mesh vertices"
./c2triangle -f square.2 &> /dev/null
./c2poisson -f square.2 -ksp_rtol 1.0e-10 |'grep' "check II"

triangle -rpqa0.0064 square.2 |'grep' "Mesh vertices"
./c2triangle -f square.3 &> /dev/null
./c2poisson -f square.3 -ksp_rtol 1.0e-10 |'grep' "check II"

triangle -rpqa0.0016 square.3 |'grep' "Mesh vertices"
./c2triangle -f square.4 &> /dev/null
./c2poisson -f square.4 -ksp_rtol 1.0e-10 |'grep' "check II"

triangle -rpqa0.0004 square.4 |'grep' "Mesh vertices"
./c2triangle -f square.5 &> /dev/null
./c2poisson -f square.5 -ksp_rtol 1.0e-10 |'grep' "check II"

triangle -rpqa0.0001 square.5 |'grep' "Mesh vertices"
./c2triangle -f square.6 &> /dev/null
./c2poisson -f square.6 -ksp_rtol 1.0e-10 |'grep' "check II"

triangle -rpqa0.000025 square.6 |'grep' "Mesh vertices"
./c2triangle -f square.7 &> /dev/null
./c2poisson -f square.7 -ksp_rtol 1.0e-10 |'grep' "check II"

#triangle -rpqa0.000006 square.7 |'grep' "Mesh vertices"
#./c2triangle -f square.8 &> /dev/null
#./c2poisson -f square.8 -ksp_rtol 1.0e-10 |'grep' "check II"

