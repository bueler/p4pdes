#!/usr/bin/env python3

from argparse import ArgumentParser, RawTextHelpFormatter
from datetime import datetime
import sys, platform

parser = ArgumentParser(description="""
Generate .geo file for Gmsh to generate a mesh on the unit square.
The settings are suitable for solving the Stokes flow problem in a
lid-driven cavity, with refinement in the lower corners so that we
can find Moffatt eddies.""",
                    formatter_class=RawTextHelpFormatter)
parser.add_argument('outname', type=str, default='', metavar='OUTNAME',
                    help='output file name ending with .geo')
parser.add_argument('-cl', type=float, default=0.1, metavar='CL',
                    help='characteristic length for most of boundary (default=0.1)')
parser.add_argument('-cornerrefine', type=float, default=20, metavar='X',
                    help='ratio of refinement in corners (default=20)')
parser.add_argument('-quiet', action='store_true', default=False,
                    help='suppress all stdout')
args = parser.parse_args()

if not args.quiet:
    print('writing lidbox domain geometry to file %s ...' % args.outname)
geo = open(args.outname, 'w')

firstline = '// box domain geometry for lid-driven cavity example\n'
usagemessage = '''// usage to generate lidbox.msh for input in stokes.py:
//   $ gmsh -2 %s\n\n''' % args.outname
meat = '''
Point(1) = {0.0,1.0,0,cl};
Point(2) = {0.0,trans,0,cl};
Point(3) = {0.0,0.0,0,cleddy};
Point(4) = {trans,0.0,0,cl};
Point(5) = {1.0-trans,0.0,0,cl};
Point(6) = {1.0,0.0,0,cleddy};
Point(7) = {1.0,trans,0,cl};
Point(8) = {1.0,1.0,0,cl};

Line(10) = {1,2};
Line(11) = {2,3};
Line(12) = {3,4};
Line(13) = {4,5};
Line(14) = {5,6};
Line(15) = {6,7};
Line(16) = {7,8};
Line(17) = {8,1};

Line Loop(20) = {10,11,12,13,14,15,16,17};
Plane Surface(30) = {20};

Physical Line(40) = {17};  // lid
Physical Line(41) = {10,11,12,13,14,15,16};  // other

Physical Surface(50) = {30};  // interior\n'''

geo.write(firstline)
now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
commandline = " ".join(sys.argv[:])
geo.write('// created %s by %s using command\n//   %s\n'
          % (now,platform.node(),commandline) )  # header records creation info
geo.write(usagemessage)
geo.write('cl = %f;  // characteristic length\n' % args.cl)
geo.write('cleddy = %f;  // characteristic length for corners (%g times smaller)\n' \
          % (args.cl/args.cornerrefine,args.cornerrefine))
geo.write('trans = 0.4;  // location of transition\n')
geo.write(meat)

geo.close()

