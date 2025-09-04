import PICO from '../src/index.js';

let detect, rects, time;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const video = document.createElement('video');
// detection loop
function loop() {
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) return requestAnimationFrame(loop);
  canvas.width = width;
  canvas.height = height;
  ctx.drawImage(video, 0, 0);
  const image = ctx.getImageData(0, 0, width, height);
  drawRects(rects, time);
  const d = Date.now();
  detect(image).then(function (dets) {
    rects = dets;
    time = Date.now() - d;
    requestAnimationFrame(loop);
  });
}
// helper function to draw rects
function drawRects(rects, time) {
  if (!rects) return;
  // overlay time
  ctx.fillStyle = 'rgba(127,127,127,0.5)';
  ctx.fillRect(0, 0, 120, 28);
  ctx.fillStyle = 'white';
  ctx.font = '12px monospace';
  ctx.fillText(`${~~(1000 / time)} fps, ${time} ms`, 10, 18);
  // draw detection rects
  for (let i = 0; i < rects.length; ++i) {
    const rect = rects[i];
    const x = ~~(rect.c - rect.s / 2);
    const y = ~~(rect.r - rect.s / 2);
    const { s: size, a: angle, q: quality } = rect;
    // check the detection score
    // if it's above the threshold, draw it
    // the constant 30.0 is empirical: it depends on PICO options
    const threshold = 30;
    let color = quality - threshold;
    if (color < 0) color = 0;
    if (color > 100) color = 100;
    if (quality > threshold) {
    // first save the untranslated/unrotated ctx
      ctx.save();
      ctx.beginPath();
      // move the rotation point to the center of the rect
      ctx.translate(x + size / 2, y + size / 2);
      // rotate the rect
      ctx.rotate((-angle * Math.PI) / 180);
      // draw the rect on the transformed ctx
      // Note: after transforming [0,0] is visually [x,y]
      //       so the rect needs to be offset accordingly when drawn
      ctx.rect(-size / 2, -size / 2, size, size);
      ctx.strokeStyle = 'hsl(' + color + ',50%,50%)';
      ctx.lineWidth = 3;
      ctx.stroke();
      // draw top line
      ctx.beginPath();
      ctx.moveTo(-size / 2, -size / 2);
      ctx.lineTo(size / 2, -size / 2);
      ctx.lineWidth = 3;
      ctx.strokeStyle = 'blue';
      ctx.stroke();
      // restore the ctx to its untranslated/unrotated state
      ctx.restore();
      // draw labels
      ctx.fillStyle = 'rgba(127,127,127,0.5)';
      ctx.fillRect(x + size + 5, y, 60, 85);
      ctx.fillStyle = 'white';
      ctx.font = '12px monospace';
      ctx.fillText('x=' + x, x + size + 10, y + 15);
      ctx.fillText('y=' + y, x + size + 10, y + 30);
      ctx.fillText('s=' + size, x + size + 10, y + 45);
      ctx.fillText('a=' + angle, x + size + 10, y + 60);
      ctx.fillText('q=' + ~~(quality), x + size + 10, y + 75);
    }
  }
}
// load cascade
fetch('/data/classifier.dat')
  .then(function (response) {
    if (!response.ok) throw Error(response.statusText || 'Request error');
    return response.arrayBuffer();
  })
  .then(function (cascade) {
    // create PICO instance without options
    detect = PICO(cascade, { rotation: [0, 30, 330], memory: 3 });
    // capture video from webcam
    return navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: { ideal: 'user' },
      },
      audio: false,
    });
  })
  .then(function (stream) {
    video.srcObject = stream;
    video.onplaying = function () {
      requestAnimationFrame(loop);
    };
    video.play();
  })
  .catch(function (err) {
    alert(err.toString());
  });
