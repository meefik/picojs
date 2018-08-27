/**
 * Converts a color from a color-space based on an RGB color model to a
 * grayscale representation of its luminance. The coefficients represent the
 * measured intensity perception of typical trichromat humans, in
 * particular, human vision is most sensitive to green and least sensitive
 * to blue.
 *
 * @param {Uint8Array|Uint8ClampedArray|Array} pixels The pixels in a linear [r,g,b,a,...] array.
 * @param {number} width The image width.
 * @param {number} height The image height.
 * @param {boolean} fillRGBA If the result should fill all RGBA values with the gray scale
 *  values, instead of returning a single value per pixel.
 * @return {Uint8ClampedArray} The grayscale pixels in a linear array ([p,p,p,a,...] if fillRGBA
 *  is true and [p1, p2, p3, ...] if fillRGBA is false).
 */
export function grayscale(pixels, width, height, fillRGBA) {
  /*
    Performance result (rough EST. - image size, CPU arch. will affect):
    https://jsperf.com/tracking-new-image-to-grayscale
    Firefox v.60b:
          fillRGBA  Gray only
    Old      11       551     OPs/sec
    New    3548      6487     OPs/sec
    ---------------------------------
            322.5x     11.8x  faster
    Chrome v.67b:
          fillRGBA  Gray only
    Old     291       489     OPs/sec
    New    6975      6635     OPs/sec
    ---------------------------------
            24.0x      13.6x  faster
    - Ken Nilsen / epistemex
   */

  var len = pixels.length >> 2;
  var gray = fillRGBA ? new Uint32Array(len) : new Uint8Array(len);
  var data32 = new Uint32Array(pixels.buffer || new Uint8Array(pixels).buffer);
  var i = 0;
  var c = 0;
  var luma = 0;

  // unrolled loops to not have to check fillRGBA each iteration
  if (fillRGBA) {
    while (i < len) {
      // Entire pixel in little-endian order (ABGR)
      c = data32[i];

      // Using the more up-to-date REC/BT.709 approx. weights for luma instead: [0.2126, 0.7152, 0.0722].
      //   luma = ((c>>>16 & 0xff) * 0.2126 + (c>>>8 & 0xff) * 0.7152 + (c & 0xff) * 0.0722 + 0.5)|0;
      // But I'm using scaled integers here for speed (x 0xffff). This can be improved more using 2^n
      //   close to the factors allowing for shift-ops (i.e. 4732 -> 4096 => .. (c&0xff) << 12 .. etc.)
      //   if "accuracy" is not important (luma is anyway an visual approx.):
      luma = ((c >>> 16 & 0xff) * 13933 + (c >>> 8 & 0xff) * 46871 + (c & 0xff) * 4732) >>> 16;
      gray[i++] = luma * 0x10101 | c & 0xff000000;
    }
  } else {
    while (i < len) {
      c = data32[i];
      luma = ((c >>> 16 & 0xff) * 13933 + (c >>> 8 & 0xff) * 46871 + (c & 0xff) * 4732) >>> 16;
      // ideally, alpha should affect value here: value * (alpha/255) or with shift-ops for the above version
      gray[i++] = luma;
    }
  }

  // Consolidate array view to byte component format independent of source view
  return new Uint8ClampedArray(gray.buffer);
}

/**
 * Extract data from the cascade binary.
 *
 * @param {Int8Array} bytes Cascade binary data.
 * @return {function}
 */
export function unpackCascade(bytes) {
  var dview = new DataView(new ArrayBuffer(4));
  // we skip the first 8 bytes of the cascade file
  // (cascade version number and some data used during the learning process)
  var p = 8;
  // read the depth (size) of each tree first: a 32-bit signed integer
  dview.setUint8(0, bytes[p + 0]), dview.setUint8(1, bytes[p + 1]), dview.setUint8(2, bytes[p + 2]), dview.setUint8(3, bytes[p + 3]);
  var tdepth = dview.getInt32(0, true);
  var pow2tdepth = Math.pow(2, tdepth) >> 0; // '>>0' transforms this number to int
  p = p + 4;
  // next, read the number of trees in the cascade: another 32-bit signed integer
  dview.setUint8(0, bytes[p + 0]), dview.setUint8(1, bytes[p + 1]), dview.setUint8(2, bytes[p + 2]), dview.setUint8(3, bytes[p + 3]);
  var ntrees = dview.getInt32(0, true);
  p = p + 4;
  // read the actual trees and cascade thresholds
  var tcodes = [];
  var tpreds = [];
  var thresh = [];
  for (var t = 0; t < ntrees; ++t) {
    var i;
    // read the binary tests placed in internal tree nodes
    Array.prototype.push.apply(tcodes, [0, 0, 0, 0]);
    Array.prototype.push.apply(tcodes, bytes.slice(p, p + 4 * Math.pow(2, tdepth) - 4));
    p = p + 4 * Math.pow(2, tdepth) - 4;
    // read the prediction in the leaf nodes of the tree
    for (i = 0; i < Math.pow(2, tdepth); ++i) {
      dview.setUint8(0, bytes[p + 0]), dview.setUint8(1, bytes[p + 1]), dview.setUint8(2, bytes[p + 2]), dview.setUint8(3, bytes[p + 3]);
      tpreds.push(dview.getFloat32(0, true));
      p = p + 4;
    }
    // read the threshold
    dview.setUint8(0, bytes[p + 0]), dview.setUint8(1, bytes[p + 1]), dview.setUint8(2, bytes[p + 2]), dview.setUint8(3, bytes[p + 3]);
    thresh.push(dview.getFloat32(0, true));
    p = p + 4;
  }
  // cosinus and sinus tables with step 1 degrees
  var qcostable = [];
  var qsintable = [];
  for (let i = 0; i < 360; i++) {
    let a = i * Math.PI / 180;
    qcostable[i] = Math.cos(a) * 256;
    qsintable[i] = Math.sin(a) * 256;
  }

  // construct the classification function from the read data
  function classifyRegion(r, c, a, s, pixels, ldim) {
    r = r << 16; // r * 65536
    c = c << 16; // c * 65536
    a = a | 0;
    var root = 0;
    var o = 0.0;
    var qsin = s * qsintable[a]; //s*(int)(256.0f*sinf(2*M_PI*a));
    var qcos = s * qcostable[a]; //s*(int)(256.0f*cosf(2*M_PI*a));
    for (let i = 0; i < ntrees; ++i) {
      let idx = 1;
      for (let j = 0; j < tdepth; ++j) {
        let n = root + 4 * idx;
        let t0 = tcodes[n + 0];
        let t1 = tcodes[n + 1];
        let t2 = tcodes[n + 2];
        let t3 = tcodes[n + 3];
        // let r1 = (r + tcodes[n + 0] * s) >> 8; // (r+tcodes[4*idx+0]*s)/256
        let r1 = (r + qcos * t0 - qsin * t1) >> 16; // (r + qcos*tcodes[4*idx+0] - qsin*tcodes[4*idx+1])/65536
        // let c1 = (c + tcodes[n + 1] * s) >> 8; // (c+tcodes[4*idx+1]*s)/256
        let c1 = (c + qsin * t0 + qcos * t1) >> 16; // (c + qsin*tcodes[4*idx+0] + qcos*tcodes[4*idx+1])/65536
        // let r2 = (r + tcodes[n + 2] * s) >> 8; // (r+tcodes[4*idx+2]*s)/256
        let r2 = (r + qcos * t2 - qsin * t3) >> 16; // (r + qcos*tcodes[4*idx+2] - qsin*tcodes[4*idx+3])/65536
        // let c2 = (c + tcodes[n + 3] * s) >> 8; // (c+tcodes[4*idx+3]*s)/256
        let c2 = (c + qsin * t2 + qcos * t3) >> 16; // (c + qsin*tcodes[4*idx+2] + qcos*tcodes[4*idx+3])/65536
        idx = 2 * idx + (pixels[r1 * ldim + c1] <= pixels[r2 * ldim + c2]);
      }
      o = o + tpreds[pow2tdepth * i + idx - pow2tdepth];
      if (o <= thresh[i]) return -1;
      root += 4 * pow2tdepth;
    }
    return o - thresh[ntrees - 1];
  }
  return classifyRegion;
}

/**
 * Run cascade and get detections.
 *
 * @param {number[]} pixels The grayscale pixels in a linear array.
 * @param {number} width The image width.
 * @param {number} height The image height.
 * @param {function} clfn The classification function from unpackCascade().
 * @param {number} shiftfactor How much to rescale the window during the multiscale detection process.
 * @param {number} scalefactor How much to move the window between neighboring detections.
 * @param {number} initialsize Minimum face size relative to the area of the image.
 * @param {number|number[]} rotation Angles of rotation in degrees.
 * @return {number[][]} Data of detections.
 */
export function runCascade(pixels, width, height, clfn, shiftfactor, scalefactor, initialsize, rotation) {
  rotation = rotation ? [].concat(rotation) : [0]; // index from qcostable/qsintable
  var maxsize = Math.max(width, height);
  var detections = [];
  while (initialsize <= maxsize) {
    let step = (shiftfactor * initialsize + 1) >> 0;
    let offset = (initialsize / 2 + 1) >> 0;
    for (let r = offset; r <= height - offset; r += step) {
      for (let c = offset; c <= width - offset; c += step) {
        for (let i = 0; i < rotation.length; i++) {
          let a = rotation[i];
          let q = clfn(r, c, a, initialsize, pixels, width);
          if (q > 0.0) {
            detections.push([r, c, initialsize, q, a]);
          }
        }
      }
    }
    initialsize *= scalefactor;
  }
  return detections;
}

/**
 * Calculates the intersection over union for two detections.
 *
 * @param {number[]} det1
 * @param {number[]} det2
 */
export function calcOverlap(det1, det2) {
  // unpack the position and size of each detection
  var r1 = det1[0];
  var c1 = det1[1];
  var s1 = det1[2];
  var r2 = det2[0];
  var c2 = det2[1];
  var s2 = det2[2];
  // calculate detection overlap in each dimension
  var or = Math.max(0, Math.min(r1 + s1 / 2, r2 + s2 / 2) - Math.max(r1 - s1 / 2, r2 - s2 / 2));
  var oc = Math.max(0, Math.min(c1 + s1 / 2, c2 + s2 / 2) - Math.max(c1 - s1 / 2, c2 - s2 / 2));
  // minimum size
  var ms = Math.min(s1, s2);
  // calculate and return overlap
  return (or * oc) / (ms * ms);
}

/**
 * Clustering the array of detection.
 *
 * @param {number[]} det
 * @param {number} threshold
 */
export function clusterDetections(dets, threshold) {
  // sort detections by their quality
  dets.sort(function (a, b) {
    return b[3] - a[3];
  });
  // do clustering through non-maximum suppression
  var assignments = [];
  var clusters = [];
  for (let i = 0; i < dets.length; i++) {
    if (assignments[i]) continue;
    // now we make a cluster out of it and see whether some other detections belong to it
    let r = dets[i][0];
    let c = dets[i][1];
    let s = dets[i][2];
    let q = dets[i][3];
    let a = dets[i][4];
    let n = 1;
    for (let j = i + 1; j < dets.length; j++) {
      if (assignments[j]) continue;
      // check overlap with other
      if (calcOverlap(dets[i], dets[j]) > threshold) {
        assignments[j] = true;
        r += dets[j][0];
        c += dets[j][1];
        s += dets[j][2];
        q += dets[j][3];
        n++;
      }
    }
    // make a cluster representative
    clusters.push({ r: r / n, c: c / n, s: s / n, q: q, a: a });
  }
  return clusters;
}

/**
 * Get function for updating images in the memory.
 *
 * @param {number} size Size of the memory.
 * @return {function}
 */
export function getMemoryUpdater(size) {
  // initialize a circular buffer of `size` elements
  var n = 0;
  var memory = [];
  for (let i = 0; i < size; ++i) {
    memory.push([]);
  }
  // build a function that:
  // (1) inserts the current frame's detections into the buffer;
  // (2) merges all detections from the last `size` frames and returns them
  function updateMemory(dets) {
    memory[n] = dets;
    n = (n + 1) % memory.length;
    dets = [];
    for (let i = 0; i < memory.length; ++i) {
      dets = dets.concat(memory[i]);
    }
    return dets;
  }
  return updateMemory;
}
