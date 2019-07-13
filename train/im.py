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
parser.add_argument('src', help='Source folder of images')
parser.add_argument('--backgrounds', default='backgrounds.txt', help='A file with a list of background images')
parser.add_argument('--faces', default='faces.txt', help='A file with a list of face images')
parser.add_argument('--labels', default='labels.txt', help='A file with a list of labels')
parser.add_argument('--plot', action='store_true', help='Preview images')
args = parser.parse_args()
plot = args.plot

if plot:
	import matplotlib.pyplot
	import matplotlib.image
	import matplotlib.cm


def write_rid(im):
	#
	# raw intensity data
	#

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

def export(im, r, c, s):
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

def mirror_and_export(im, r, c, s):
	#
	# exploit mirror symmetry of the face
	#

	# flip image
	im = numpy.asarray(ImageOps.mirror(Image.fromarray(im)))

	# flip column coordinate of the object
	c = im.shape[1] - c

	# export
	export(im, r, c, s)

def prepare_faces():
    # faces list
    imlist = open(args.faces, 'r').readlines()

    # object sample is specified by three coordinates (row, column and size; all in pixels)
    rs = [float(line.split()[1]) for line in open(args.labels, 'r').readlines()]
    cs = [float(line.split()[0]) for line in open(args.labels, 'r').readlines()]
    ss = [float(line.split()[2]) for line in open(args.labels, 'r').readlines()]

    n = 0

    for i in range(0, len(imlist)):
        # construct full image path
        path = args.src + '/' + imlist[i].strip()

        n = n + 1
        sys.stderr.write(str(n) + '\t' + path + '\n')

        r = rs[i]
        c = cs[i]
        s = ss[i]

        try:
            im = Image.open(path).convert('L')
        except:
            continue

        im = numpy.asarray(im)

        export(im, r, c, s)

        # faces are symmetric and we exploit this here
        mirror_and_export(im, r, c, s)

def prepare_backgrounds():
    # backgrounds list
    imlist = open(args.backgrounds, 'r').readlines()

    n = 0

    for i in range(0, len(imlist)):
        # construct full image path
        path = args.src + '/' + imlist[i].strip()

        n = n + 1
        sys.stderr.write(str(n) + '\t' + path + '\n')

        try:
            im = numpy.asarray(Image.open(path).convert('L'))
        except:
            continue

        write_rid(im)
        sys.stdout.buffer.write( struct.pack('i', 0) )

prepare_faces()
prepare_backgrounds()
