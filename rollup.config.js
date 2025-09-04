import terser from '@rollup/plugin-terser';

const { NODE_ENV = 'production' } = process.env;

export default {
  input: 'src/index.js',
  output: [{
    file: 'dist/pico.umd.js',
    format: 'umd',
    name: 'PICO',
  }, {
    file: 'dist/pico.esm.js',
    format: 'esm',
  }],
  plugins: NODE_ENV === 'production' ? [terser()] : [],
};
