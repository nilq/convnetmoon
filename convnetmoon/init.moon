root = "convnetmoon/"
modules = {
  "extern/json",
  "util",
}

for _, v in pairs modules
  require root .. v
