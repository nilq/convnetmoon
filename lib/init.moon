require "lib/util"
require "lib/vol"
require "lib/vol_util"

require "lib/net"

n = Net!

n\make_layers {
  {type: "input", out_sx: 1, out_sy: 1, out_sz: 1}
  {type: "fc", num_neurons: 6, activation: "tanh"},
  {type: "softmax", num_classes: 2},
}
