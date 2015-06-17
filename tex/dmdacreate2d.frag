//START
  DM  da;
  DMDACreate2d(PETSC_COMM_WORLD,
               DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
               -9,-9,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,
               &da);
  DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,-1.0,-1.0);
//STOP

