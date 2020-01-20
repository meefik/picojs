#!/usr/bin/python3
#
# sudo apt install python3-numpy python3-pil
#

import sys
import random
import struct
import argparse
import numpy
from PIL import Image
from PIL import ImageOps

parser = argparse.ArgumentParser()
parser.add_argument('labels', help='A file with a list of labels')
parser.add_argument('--plot', action='store_true', help='Preview images')
args = parser.parse_args()
plot = args.plot

if plot:
    import matplotlib.pyplot
    import matplotlib.image
    import matplotlib.cm


def write_rid(im):
    h = im.shape[0]
    w = im.shape[1]

    hw = struct.pack('ii', h, w)

    tmp = [None]*w*h
    for y in range(0, h):
        for x in range(0, w):
            tmp[y*w + x] = im[y, x]

    pixels = struct.pack('%sB' % w*h, *tmp)

    sys.stdout.buffer.write(hw)
    sys.stdout.buffer.write(pixels)

def export(im, c, r, s):
    nrows = im.shape[0]
    ncols = im.shape[1]

    # crop
    r0 = max(int(r - 0.75*s), 0); r1 = min(int(r + 0.75*s), nrows)
    c0 = max(int(c - 0.75*s), 0); c1 = min(int(c + 0.75*s), ncols)

    im = im[r0:r1, c0:c1]

    nrows = im.shape[0]
    ncols = im.shape[1]

    r = r - r0
    c = c - c0

    # resize, if needed
    maxwsize = 192.0
    wsize = max(nrows, ncols)

    ratio = maxwsize/wsize

    if ratio<1.0:
        im = numpy.asarray( Image.fromarray(im).resize((int(ratio*ncols), int(ratio*nrows))) )

        r = ratio*r
        c = ratio*c
        s = ratio*s

    nrands = 7;

    lst = []

    for i in range(0, nrands):
        stmp = s*random.uniform(0.9, 1.1)

        rtmp = r + s*random.uniform(-0.05, 0.05)
        ctmp = c + s*random.uniform(-0.05, 0.05)

        if plot:
            matplotlib.pyplot.cla()

            matplotlib.pyplot.plot([ctmp-stmp/2, ctmp+stmp/2], [rtmp-stmp/2, rtmp-stmp/2], 'b', linewidth=3)
            matplotlib.pyplot.plot([ctmp+stmp/2, ctmp+stmp/2], [rtmp-stmp/2, rtmp+stmp/2], 'b', linewidth=3)
            matplotlib.pyplot.plot([ctmp+stmp/2, ctmp-stmp/2], [rtmp+stmp/2, rtmp+stmp/2], 'b', linewidth=3)
            matplotlib.pyplot.plot([ctmp-stmp/2, ctmp-stmp/2], [rtmp+stmp/2, rtmp-stmp/2], 'b', linewidth=3)

            matplotlib.pyplot.imshow(im, cmap=matplotlib.cm.Greys_r)

            matplotlib.pyplot.show()

        lst.append( (int(rtmp), int(ctmp), int(stmp)) )

    write_rid(im)

    sys.stdout.buffer.write( struct.pack('i', nrands) )

    for i in range(0, nrands):
        sys.stdout.buffer.write( struct.pack('iii', lst[i][0], lst[i][1], lst[i][2]) )

def mirror_and_export(im, c, r, s):
    #
    # exploit mirror symmetry of the face
    #

    # flip image
    im = numpy.asarray(ImageOps.mirror(Image.fromarray(im)))

    # flip column coordinate of the object
    c = im.shape[1] - c

    # export
    export(im, c, r, s)

def prepare():
    # object sample is specified by three coordinates (row, column and size; all in pixels)
    cs = [int(line.split()[0]) for line in open(args.labels, 'r').readlines()]
    rs = [int(line.split()[1]) for line in open(args.labels, 'r').readlines()]
    ss = [int(line.split()[2]) for line in open(args.labels, 'r').readlines()]
    fn = [line.split()[3].strip() for line in open(args.labels, 'r').readlines()]

    n = 0

    for i in range(0, len(fn)):
        n = n + 1
        c = cs[i]
        r = rs[i]
        s = ss[i]
        f = fn[i]

        try:
            im = numpy.asarray(Image.open(f).convert('L'))
        except:
            continue

        sys.stderr.write(str(c) + '\t' + str(r) + '\t' + str(s) + '\t' + f + '\n')

        if (c > 0 and r > 0 and s > 0):
            export(im, c, r, s)
            mirror_and_export(im, c, r, s)
        else:
            write_rid(im)
            sys.stdout.buffer.write( struct.pack('i', 0) )

prepare()
