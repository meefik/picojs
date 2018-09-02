import { grayscale, getMemoryUpdater, unpackCascade, runCascade, clusterDetections } from './utils';

export default class PICO {
  /**
   * PICO constructor.
   *
   * @param {Object} [options] Algorithm options.
   * @param {number} [options.shiftfactor=0.1] Move the detection window by 10% of its size.
   * @param {number} [options.scalefactor=1.1] Resize the detection window by 10% when moving to the higher scale.
   * @param {number} [options.initialsize=0.1] Minimum size of a face (10% of image area).
   * @param {number[]} [options.rotation=0] Angles of rotation in degrees.
   * @param {number} [options.threshold=0.2] Overlap threshold.
   * @param {number} [options.memory=1] Number of images in the memory.
   */
  constructor(options = {}) {
    this.options = Object.assign({
      shiftfactor: 0.1,
      scalefactor: 1.1,
      initialsize: 0.1,
      rotation: 0,
      threshold: 0.2,
      memory: 1
    }, options);
    this._clfn = function () { return -1.0; };
    this._mu = getMemoryUpdater(this.options.memory);
  }
  /**
   * Load the cascade from URL.
   *
   * @param {string} url URL of the cascade file.
   * @return {Promise}
   */
  loadCascade(url) {
    return fetch(url).then((response) => {
      return response.arrayBuffer().then((buffer) => {
        var bytes = new Int8Array(buffer);
        this._clfn = unpackCascade(bytes);
      });
    });
  }
  /**
   * Detect face in the image.
   *
   * @param {ImageData} image Data of image.
   * @return {Object[]}
   * r - row, c - col, s - size, q - quality, a - angle in degrees
   */
  detect(image) {
    var { data, width, height } = image;
    var pixels = grayscale(data, width, height);
    var { shiftfactor, scalefactor, initialsize, rotation, threshold } = this.options;
    var dets = runCascade(pixels, width, height, this._clfn,
      shiftfactor, scalefactor, initialsize, rotation);
    dets = clusterDetections(this._mu(dets), threshold);
    return dets;
  }
}
