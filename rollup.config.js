import terser from '@rollup/plugin-terser';

export default {
  input: 'src/pico.js',
  output: [{
    file: 'dist/pico.min.js',
    format: 'umd',
    name: 'PICO'
  }],
  plugins: [terser()]
};
