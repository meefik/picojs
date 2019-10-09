var fs = require("fs");

var filename = process.argv[2];
var buffer = [];

var stream = fs.createReadStream(filename);
stream.on("data", function (chunk) {
  for (var i = 0; i < chunk.length; i++) {
    buffer.push(chunk[i]);
  }
});

stream.on("close", function () {
  console.log(JSON.stringify(buffer));
});
