/**
 * Face detection using the PICO algorithm with the face rotation
 * invariant implementation.
 *
 * Paper: https://arxiv.org/abs/1305.4537
 * Source code: https://github.com/meefik/picojs
 */

/**
 * PICO constructor.
 *
 * @param {number[]} cascade Classification cascade.
 * @param {Object} [options] Algorithm options.
 * @param {number} [options.shiftfactor=0.1] Move the detection window by 10% of its size.
 * @param {number} [options.scalefactor=1.1] For multiscale processing: resize the detection window by 10% when moving to the higher scale.
 * @param {number} [options.initialsize=0.1] Minimum size of a face (10% of image area).
 * @param {number} [options.threshold=0.2] Overlap threshold.
 * @param {number} [options.memory=1] Number of images in the memory.
 * @param {number[]} [options.rotation=[0]] Angles of rotation in degrees.
 */
function PICO(cascade, options) {
  const {
    shiftfactor = 0.1,
    scalefactor = 1.1,
    initialsize = 0.1,
    threshold = 0.2,
    rotation = [0],
    memory = 1,
  } = options;

  const classifier = unpackCascade(new Int8Array(cascade));
  const memoryUpdater = getMemoryUpdater(memory);

  /**
   * Detect face in the image.
   *
   * @param {ImageData} image Data of image.
   * @return {Object[]}
   * r - row, c - col, s - size, q - quality, a - angle in degrees
   */
  function detect(image) {
    const { data, width, height } = image;
    const pixels = grayscale(data);
    const dets = runCascade(
      pixels,
      width,
      height,
      classifier,
      shiftfactor,
      scalefactor,
      initialsize,
      rotation,
    );
    return clusterDetections(memoryUpdater(dets), threshold);
  }

  /**
   * Extract data from the cascade binary.
   *
   * @param {Int8Array} bytes Cascade binary data.
   * @return {function}
   */
  function unpackCascade(bytes) {
    // use a DataView directly over the Int8Array buffer for fast reads
    const dview = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    // skip the first 8 bytes of the cascade file
    // (cascade version number and some data used during the learning process)
    let p = 8;
    const tdepth = dview.getInt32(p, true);
    p += 4;
    const pow2tdepth = 1 << tdepth;
    const ntrees = dview.getInt32(p, true);
    p += 4;
    // allocate typed arrays sized ahead of time to reduce allocations
    const tcodes = new Int8Array(ntrees * 4 * pow2tdepth);
    const tpreds = new Float32Array(ntrees * pow2tdepth);
    const thresh = new Float32Array(ntrees);
    for (let t = 0; t < ntrees; ++t) {
      const base = t * 4 * pow2tdepth;
      // first 4 values are zeros (kept for historical layout)
      tcodes[base + 0] = 0;
      tcodes[base + 1] = 0;
      tcodes[base + 2] = 0;
      tcodes[base + 3] = 0;
      // copy rest of tcodes for this tree
      const sliceLen = 4 * pow2tdepth - 4;
      tcodes.set(bytes.subarray(p, p + sliceLen), base + 4);
      p += sliceLen;
      // read predictions (float32)
      for (let i = 0; i < pow2tdepth; ++i) {
        tpreds[t * pow2tdepth + i] = dview.getFloat32(p, true);
        p += 4;
      }
      // read threshold
      thresh[t] = dview.getFloat32(p, true);
      p += 4;
    }
    // cosinus and sinus tables (int) - using Int32Array is faster than pushing into JS arrays
    const qcostable = new Int32Array(360);
    const qsintable = new Int32Array(360);
    for (let i = 0; i < 360; ++i) {
      const a = (i * Math.PI) / 180;
      qcostable[i] = (Math.cos(a) * 256) | 0;
      qsintable[i] = (Math.sin(a) * 256) | 0;
    }
    // construct the classification function from the read data
    function classifyRegion(r, c, a, s, pixels, ldim) {
      r = r << 16; // r * 65536
      c = c << 16; // c * 65536
      a = a | 0;
      const qsin = s * qsintable[a]; // s*(int)(256.0f*sinf(2*M_PI*a));
      const qcos = s * qcostable[a]; // s*(int)(256.0f*cosf(2*M_PI*a));
      let o = 0.0;
      for (let i = 0; i < ntrees; ++i) {
        let idx = 1;
        for (let j = 0; j < tdepth; ++j) {
          const n = 4 * (pow2tdepth * i + idx);
          const t0 = tcodes[n + 0];
          const t1 = tcodes[n + 1];
          const t2 = tcodes[n + 2];
          const t3 = tcodes[n + 3];
          // let r1 = (r + tcodes[n + 0] * s) >> 8; // (r+tcodes[4*idx+0]*s)/256
          const r1 = (r + qcos * t0 - qsin * t1) >> 16; // (r + qcos*tcodes[4*idx+0] - qsin*tcodes[4*idx+1])/65536
          // let c1 = (c + tcodes[n + 1] * s) >> 8; // (c+tcodes[4*idx+1]*s)/256
          const c1 = (c + qsin * t0 + qcos * t1) >> 16; // (c + qsin*tcodes[4*idx+0] + qcos*tcodes[4*idx+1])/65536
          // let r2 = (r + tcodes[n + 2] * s) >> 8; // (r+tcodes[4*idx+2]*s)/256
          const r2 = (r + qcos * t2 - qsin * t3) >> 16; // (r + qcos*tcodes[4*idx+2] - qsin*tcodes[4*idx+3])/65536
          // let c2 = (c + tcodes[n + 3] * s) >> 8; // (c+tcodes[4*idx+3]*s)/256
          const c2 = (c + qsin * t2 + qcos * t3) >> 16; // (c + qsin*tcodes[4*idx+2] + qcos*tcodes[4*idx+3])/65536
          const p1 = pixels[r1 * ldim + c1] | 0;
          const p2 = pixels[r2 * ldim + c2] | 0;
          idx = 2 * idx + (p1 <= p2 ? 1 : 0);
        }
        o += tpreds[pow2tdepth * (i - 1) + idx];
        if (o <= thresh[i]) return -1;
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
  function runCascade(
    pixels,
    width,
    height,
    classifier,
    shiftfactor,
    scalefactor,
    initialsize,
    rotation,
  ) {
    const detections = [];
    const rl = rotation.length;
    const minsize = (initialsize * Math.sqrt(width * height)) | 0;
    let s = Math.min(width, height) | 0;
    while (s >= minsize) {
      const step = (shiftfactor * s + 1) | 0;
      const offset = (s / 2 + 1) | 0;
      for (let r = offset; r < height - offset; r += step) {
        for (let c = offset; c < width - offset; c += step) {
          for (let i = 0; i < rl; i++) {
            const a = rotation[i];
            const q = classifier(r, c, a, s, pixels, width);
            if (q > 0) {
              detections.push([r, c, s, q, a]);
            }
          }
        }
      }
      s = (s / scalefactor) | 0;
    }
    return detections;
  }

  /**
   * Calculates the intersection over union for two detections.
   *
   * @param {number[]} det1
   * @param {number[]} det2
   */
  function calcOverlap(det1, det2) {
    // unpack the position and size of each detection
    const [r1, c1, s1] = det1;
    const [r2, c2, s2] = det2;
    // calculate detection overlap in each dimension
    const or = Math.max(
      0,
      Math.min(r1 + s1 / 2, r2 + s2 / 2) - Math.max(r1 - s1 / 2, r2 - s2 / 2),
    );
    const oc = Math.max(
      0,
      Math.min(c1 + s1 / 2, c2 + s2 / 2) - Math.max(c1 - s1 / 2, c2 - s2 / 2),
    );
    // minimum size
    const ms = Math.min(s1, s2);
    // calculate and return overlap
    return (or * oc) / (ms * ms);
  }

  /**
   * Clustering the array of detection.
   *
   * @param {number[]} det
   * @param {number} threshold
   */
  function clusterDetections(dets, threshold) {
    // sort detections by their quality
    dets.sort((a, b) => b[3] - a[3]);
    // do clustering through non-maximum suppression
    const assignments = new Uint8Array(dets.length);
    const clusters = [];
    for (let i = 0; i < dets.length; i++) {
      if (assignments[i]) continue;
      // now we make a cluster out of it and see whether some other detections belong to it
      let r = dets[i][0];
      let c = dets[i][1];
      let s = dets[i][2];
      let q = dets[i][3];
      const a = dets[i][4];
      let n = 1;
      for (let j = i + 1; j < dets.length; j++) {
        if (assignments[j]) continue;
        // check overlap with other
        if (calcOverlap(dets[i], dets[j]) > threshold) {
          assignments[j] = 1;
          r += dets[j][0];
          c += dets[j][1];
          s += dets[j][2];
          q += dets[j][3];
          n++;
        }
      }
      // make a cluster representative
      clusters.push({
        r: (r / n) | 0,
        c: (c / n) | 0,
        s: (s / n) | 0,
        q,
        a,
      });
    }
    return clusters;
  }

  /**
   * Get function for updating images in the memory.
   *
   * @param {number} size Size of the memory.
   * @return {function}
   */
  function getMemoryUpdater(size) {
    // initialize a circular buffer of `size` elements
    let n = 0;
    const memory = [];
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

  /**
   * Converts a color from a color-space based on an RGB color model to a
   * grayscale representation of its luminance. The coefficients represent the
   * measured intensity perception of typical trichromat humans, in
   * particular, human vision is most sensitive to green and least sensitive
   * to blue.
   * The source code from tracking.js: https://github.com/eduardolundgren/tracking.js
   *
   * @param {Uint8Array|Uint8ClampedArray|Array} pixels The pixels in a linear [r,g,b,a,...] array.
   * @param {boolean} fillRGBA If the result should fill all RGBA values with the gray scale
   *  values, instead of returning a single value per pixel.
   * @return {Uint8ClampedArray} The grayscale pixels in a linear array ([p,p,p,a,...] if fillRGBA
   *  is true and [p1, p2, p3, ...] if fillRGBA is false).
   */
  function grayscale(pixels, fillRGBA) {
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

    const len = pixels.length >> 2;
    const gray = fillRGBA ? new Uint32Array(len) : new Uint8Array(len);
    const data32 = new Uint32Array(
      pixels.buffer || new Uint8Array(pixels).buffer,
    );
    let i = 0;
    let c = 0;
    let luma = 0;

    while (i < len) {
      // Entire pixel in little-endian order (ABGR)
      c = data32[i];

      // Using the more up-to-date REC/BT.709 approx. weights for luma instead: [0.2126, 0.7152, 0.0722].
      //   luma = ((c>>>16 & 0xff) * 0.2126 + (c>>>8 & 0xff) * 0.7152 + (c & 0xff) * 0.0722 + 0.5)|0;
      // But I'm using scaled integers here for speed (x 0xffff). This can be improved more using 2^n
      //   close to the factors allowing for shift-ops (i.e. 4732 -> 4096 => .. (c&0xff) << 12 .. etc.)
      //   if "accuracy" is not important (luma is anyway an visual approx.):
      luma = (((c >>> 16) & 0xff) * 13933
        + ((c >>> 8) & 0xff) * 46871
        + (c & 0xff) * 4732) >>> 16;
      gray[i++] = fillRGBA ? (luma * 0x10101) | (c & 0xff000000) : luma;
    }

    // Consolidate array view to byte component format independent of source view
    return new Uint8ClampedArray(gray.buffer);
  }

  return detect;
}

/**
 * Create detector function.
 *
 * @param {number[]} cascade Classification cascade.
 * @param {Object} [options] Algorithm options.
 * @return {function} Detector function.
 */
export default function (cascade, options) {
  // create worker
  const fnString = `(function(){${PICO.toString()};var detect=${
    PICO.name
  }([${new Int8Array(cascade).toString()}],${JSON.stringify(
    options || {},
  )});onmessage=function(e){postMessage(detect(e.data));};})();`;
  const workerBlob = new Blob([fnString], {
    type: 'application/javascript',
  });
  const workerBlobURL = URL.createObjectURL(workerBlob);
  const worker = new Worker(workerBlobURL);
  // send data
  return function (data) {
    return new Promise((resolve) => {
      worker.onmessage = e => resolve(e.data);
      worker.postMessage(data);
    });
  };
}
