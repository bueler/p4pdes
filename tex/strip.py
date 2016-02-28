#!/usr/bin/env python
#
# (C) 2016 Ed Bueler
#
# This script puts only the "meat" of codes into new files in cstrip/, for
# inclusion into the book.
#
# We remove "PetscErrorCode ierr" lines and "//STRIP" lines and
# characters "ierr = " and "CHKERRQ(ierr);"

import os
import re

# filenames should be distinct
files = {2 : "vecmatksp.c tri.c",
         3 : "poisson.c",
         4 : "expcircle.c ecjacobian.c reaction.c",
         5 : "plap.c",
         6 : "ode.c heat.c",
         7 : "ad3.c",
         10: "readmesh.c poissontools.c poissonfem.c",
         13: "obstacle.c"}

if not os.path.exists("cstrip/"):
    os.makedirs("cstrip/")
else:
    print "WARNING:  cstrip/ exists"

#FIXME  no actual need for .tmp names; we are already copying

# use for loop to read all values and indexes
csfiles = []
for chapt, chfiles in files.items():
    flist = chfiles.split()
    for fname in flist:
        src = '../c/ch' + str(chapt) + '/' + fname
        destroot = 'cstrip/' + fname
        csfiles.append(destroot)
        dest = destroot + '.tmp'
        print '  copying  %s to  %s' % (src.ljust(30),dest.ljust(30))
        os.system('cp %s %s' % (src,dest))

for name in csfiles:
    f = open(name + '.tmp','r')
    newf = open(name,'w')
    for line in f:
        if re.search('\/\/STRIP',line) or re.search('PetscErrorCode ierr',line):
            continue
        line = re.sub('ierr = ','', line).rstrip()
        line = re.sub('CHKERRQ\(ierr\);','', line).rstrip()
        newf.write(line + '\n')
    f.close()
    newf.close()

