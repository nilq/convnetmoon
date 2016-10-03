root = "convnetmoon/"
modules = {
  "util",
  "vol",
  "net",
  "trainer",
  "layers/input",
  "layers/dotproduct",
}

export json = require root .. "extern/json"

for _, v in pairs modules
  require root .. v
