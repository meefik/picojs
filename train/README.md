# PICO face classifier

This tutorial will guide you through the process of learning your own face classifier.

## How to build

Build OpenCV with modules:

```sh
git clone https://github.com/opencv/opencv.git opencv
git clone https://github.com/opencv/opencv_contrib.git opencv_contrib

cmake -S "./opencv" -B "./opencv_build" \
 -DCMAKE_BUILD_TYPE=Release \
 -DBUILD_opencv_face=ON \
 -DOPENCV_EXTRA_MODULES_PATH="./opencv_contrib/modules"

(
  cd ./opencv_build
  make -j$(grep -c ^processor /proc/cpuinfo)
)
```

Build `detector`:

```sh
cmake -S "./detector" -B "./detector_build"

(
  cd ./detector_build
  make all
)
```

Build `picolrn`:

```sh
(
  cd ./picolrn
  make
)
```

Download DNN data:

```sh
wget https://github.com/opencv/opencv_extra/raw/965a410b36d3ee1cd58cd57626feb5637a5306b8/testdata/dnn/opencv_face_detector.pbtxt
wget https://github.com/opencv/opencv_3rdparty/raw/8033c2bc31b3256f0d461c919ecc01c2428ca03b/opencv_face_detector_uint8.pb
```

## How to train classifier

Remove low quality images if necessary (less than 8k and not equal to 640x480):

```sh
find /path/to/images/ -type f -size -8k -exec rm {} \;
find /path/to/images/ -type f | while read f; do identify $f | grep -q 'JPEG 640x480' || rm $f; done
```

Make labels.csv:

```sh
./detector_build/detector /path/to/images/ 0.87 1>faces.csv
./detector_build/detector /path/to/images/ 0.13 2>backgrounds.csv
cat faces.csv backgrounds.csv >labels.csv
```

You can use the image annotation tool: http://localhost:3000/train/annotation.html

Prepare positive training samples and background images:

```sh
./im.py labels.csv >trdata
```

The file `trdata` can now be processed with `picolrn`.

Start `picolrn` with default parameters:

```sh
./picolrn/picolrn trdata cascade.dat
```

After the learning is finished (~2 days on a modern machine with 4 CPU cores), you should find a classification cascade file `cascade.dat` in the folder.
