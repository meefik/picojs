const path = require('path');

export default (env = {}) => ({
  mode: env.production ? 'production' : 'development',
  devtool: env.production ? false : 'source-map',
  entry: './src/pico.js',
  output: {
    path: path.resolve(__dirname, './dist'),
    filename: 'pico.js',
    libraryTarget: 'umd',
    libraryExport: 'default',
    library: 'PICO',
    umdNamedDefine: true,
    globalObject: 'typeof self !== "undefined" ? self : this'
  },
  module: {
    rules: [
      {
        test: /\.(js)$/,
        exclude: /(node_modules)/,
        use: 'babel-loader'
      }
    ]
  }
});
