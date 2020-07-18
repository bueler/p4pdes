#!/usr/bin/env python3

# solution to exercise 14.16
# regarding "Also confirm that eddies disappear around 146^o", I can find eddies at 130^o but not at 140^o or higher

# for example, using P4xP3 elements and a direct solver:
#   $ ./angle.py -angle 130 -cornerrefine 2000 wedge.geo
#   $ gmsh -2 wedge.geo
#   $ ../stokes.py -mesh wedge.msh -o wedge.pvd -s_ksp_converged_reason -dm_view -s_pc_type lu -s_mat_type aij -s_pc_factor_shift_type inblocks -s_ksp_type preonly -udegree 4 -pdegree 3 -refine 2

from argparse import ArgumentParser, RawTextHelpFormatter
from datetime import datetime
import numpy as np
import sys, platform

parser = ArgumentParser(description="""
Generate .geo file for a triangular domain, a wedge, with controllable base angle.
Compare lidbox.py.""",
                    formatter_class=RawTextHelpFormatter)
parser.add_argument('outname', type=str, default='', metavar='OUTNAME',
                    help='output file name ending with .geo')
parser.add_argument('-angle', type=float, default=28.1, metavar='CL',
                    help='angle in degrees at base of wedge (default=28.1)')
parser.add_argument('-cl', type=float, default=0.2, metavar='CL',
                    help='characteristic length for top (default=0.2)')
parser.add_argument('-cornerrefine', type=float, default=200, metavar='X',
                    help='ratio of refinement in corners (default=200)')
parser.add_argument('-quiet', action='store_true', default=False,
                    help='suppress all stdout')
parser.add_argument('-usenames', action='store_true', default=False,
                    help='put names "dirichlet","neumann","interior" in PhysicalNames() ... used only for running through c/ch10/vis/petsc2tikz.py')
args = parser.parse_args()

if not args.quiet:
    print('writing wedge domain geometry with base angle %.1f to file %s ...' \
          % (args.angle,args.outname))
geo = open(args.outname, 'w')

firstline = '// angle (triangle) domain geometry for lid-driven cavity example\n'

meat = '''
Point(1) = {0.0,topy,0,cl};     // left top
Point(2) = {0.5,0.0,0,cleddy};  // bottom
Point(3) = {1.0,topy,0,cl};     // right top

Line(10) = {1,2};
Line(11) = {2,3};
Line(12) = {3,1};

Line Loop(20) = {10,11,12};
Plane Surface(30) = {20};\n'''

physnums = '''
Physical Line(40) = {12};  // lid
Physical Line(41) = {10,11};  // other

Physical Surface(50) = {30};  // interior\n'''

physnames = '''
Physical Line("dirichlet") = {10,11,12};
Physical Line("neumann") = {};

Physical Surface("interior") = {30};\n'''

geo.write(firstline)
now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
commandline = " ".join(sys.argv[:])
geo.write('// created %s by %s using command\n//   %s\n'
          % (now,platform.node(),commandline) )  # header records creation info
theta = 0.5 * (np.pi/180.0) * args.angle  # half angle at base of wedge
topy = 0.5 / np.tan(theta)                # note top width always 1.0
geo.write('topy = %f;  // characteristic length\n' % topy)
geo.write('cl = %f;   // characteristic length\n' % args.cl)
geo.write('cleddy = %f;  // characteristic length for corner (%g times smaller)\n' \
          % (args.cl/args.cornerrefine,args.cornerrefine))
geo.write(meat)
if args.usenames:
    geo.write(physnames)
else:
    geo.write(physnums)
geo.close()

