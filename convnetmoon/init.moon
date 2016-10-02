root = "convnetmoon/"
modules = {
  "util",
}

export json = require root .. "extern/json"

for _, v in pairs modules
  require root .. v
