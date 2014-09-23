# set PETSC for
# use
#   $ source setmypetsc.sh

machine=$(uname -n)
if [[ "$machine" == 'bueler-leopard' ]]; then
   echo "on my workstation ${machine}:"
   export PETSC_DIR=~/petsc-3.5.2/
   export PETSC_ARCH=linux-gnu-opt
   # mpiexec is global
elif [[ "$machine" == 'bueler-gazelle' ]]; then
   echo "on my laptop ${machine}:"
   export PETSC_DIR=~/petsc-3.5.2/
   export PETSC_ARCH=linux-gnu-opt
   export PATH=${PETSC_DIR}${PETSC_ARCH}/bin:$PATH  # find mpiexec
else
   echo "'uname -n' returns: ${machine}"
   echo "... huh?  no change to PETSC_DIR/ARCH because unknown machine ..."
   return
fi

echo "  PETSC_DIR  = ${PETSC_DIR}"
echo "  PETSC_ARCH = ${PETSC_ARCH}"
mympiexec=$(which mpiexec)
echo "  'which mpiexec' returns: ${mympiexec}"
