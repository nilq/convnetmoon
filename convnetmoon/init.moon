root = "convnetmoon/"
modules = {
  "util",
  "vol",
  "net",
  "trainer",
  "layers/input",
}

export json = require root .. "extern/json"

for _, v in pairs modules
  require root .. v
