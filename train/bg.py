#!/usr/bin/python3

import sys
import struct
import argparse
import numpy
import os
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('src', help='Source folder of images')
args = parser.parse_args()


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


n = 0

for dirpath, dirnames, filenames in os.walk(args.src):
	for filename in filenames:
		path = dirpath + '/' + filename

		n = n + 1
		sys.stderr.write(str(n) + '\t' + path + '\n')

		try:
			im = numpy.asarray(Image.open(path).convert('L'))
		except:
			continue

		write_rid(im)
		sys.stdout.buffer.write( struct.pack('i', 0) )

