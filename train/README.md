This tutorial will guide you through the process of learning your own face detector.

## 1. Get images with annotated faces:

* Download the GENKI dataset from <https://github.com/watersink/GENKI/raw/master/GENKI-R2009a.zip>
* Extract the contents of the archive to some folder: `GENKI-R2009a`

## 2. Get images that do not contain faces (background):

* Download `background.tar` from <http://www.vision.caltech.edu/Image_Datasets/background/background.tar>
* Extract the contents to some folder: `background`

The negative training samples are data mined from these images during the training process.

## 3. Generate the training data

* Prepare positive training samples: `./im.py --images=GENKI-R2009a/Subsets/GENKI-SZSL/GENKI-SZSL_Images.txt --labels=GENKI-R2009a/Subsets/GENKI-SZSL/GENKI-SZSL_labels.txt GENKI-R2009a/files >trdata`
* Prepare background images: `./bg.py background >>trdata`

The file `trdata` can now be processed with `picolrn`.

## 4. Start the training process

Start with default parameters:

	$ ./picolrn trdata out

After the learning is finished (~2 days on a modern machine with 4 CPU cores), you should find a classification cascade file `out` in the folder.

