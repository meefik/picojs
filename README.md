# PICO Face Detection

A library for detecting faces using the [PICO](https://arxiv.org/abs/1305.4537)
algorithm with the face rotation invariant implementation.

## Usage

An example of using the basic functions of the library:
```js
// create PICO instance with options
var pico = new PICO({
  shiftfactor: 0.1, // move the detection window by 10% of its size
  scalefactor: 1.1, // resize the detection window by 10% when moving to the higher scale
  initialsize: 0.1, // minimum size of a face (10% of image area)
  rotation: [0, 30, 60, 90, 270, 300, 330], // rotation angles in degrees
  threshold: 0.2, // overlap threshold
  memory: 3 // number of images in the memory
});
// load cascade
pico.loadCascade('./cascade/facefinder').then(function() {
  // image = ImageData
  var dets = pico.detect(image);
  // dets = [{ r: rows, c: cols, s: size, q: quality, a: angle }]
});
```

## Run demo

Run the dev server:
```
npm install
npm start
```
And open the link in your browser http://localhost:8080

See this video: https://youtu.be/9WiGC08_ZFY

## Related projects

- PICO: https://github.com/nenadmarkus/pico
- picojs: https://github.com/tehnokv/picojs
- tracking.js: https://github.com/eduardolundgren/tracking.js
