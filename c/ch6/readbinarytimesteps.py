#!/usr/bin/env python

# reading output from  -ts_monitor binary:foo.dat  (branch  barry/feature-ts-monitor-binary)

# example:
#   $ ./readbinarytime -h
#   $ ./ode -ts_monitor binary:foo.dat
#   $ ./readbinarytime foo.dat

import sys
import argparse
import struct

parser = argparse.ArgumentParser(description='Example showing read from -ts_monitor binary output.')
parser.add_argument('f',metavar='FILE',
                    help='input file from -ts_monitor binary:foo.dat')
args = parser.parse_args()

try:
    f = open(args.f,'r')
except:
    print 'cannot open %s for reading ... stopping' % args.f

while True:
    try:
        bytes = f.read(8)
    except:
        print "f.read() failed"
        sys.exit(1)
    if len(bytes) > 0:
        print struct.unpack('>d',bytes)[0]
    else:
        break

f.close()
    

