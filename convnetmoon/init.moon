root = "convnetmoon/"
modules = {
  "util",
  "vol",
  "net",
  "trainer",
}

export json = require root .. "extern/json"

for _, v in pairs modules
  require root .. v
