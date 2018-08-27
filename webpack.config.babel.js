const path = require('path');

export default (env = {}) => ({
  mode: env.production ? 'production' : 'development',
  devtool: env.production ? false : 'source-map',
  entry: './src/pico.js',
  output: {
    path: path.resolve(__dirname, './lib'),
    filename: 'pico.js',
    libraryTarget: 'this',
    libraryExport: 'default',
    library: 'PICO'
  },
  module: {
    rules: [{
      test: /\.(js)$/,
      exclude: /(node_modules)/,
      use: 'babel-loader'
    }]
  }
});
