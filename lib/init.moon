require "lib/util"
require "lib/vol"
require "lib/vol_util"

require "lib/net"
require "lib/trainer"

require "lib/dot_layer"

math.sign = (n) ->
  if n < 0
    return -1
  elseif n > 0
    return 1
  0

n = Net!

cl = ConvLayer {sx: 2, filters: 1, in_sz: 3, in_sx: 1, in_sy: 2}

n\make_layers {
  {type: "input", out_sx: 1, out_sy: 1, out_sz: 1},
  {type: "fc", num_neurons: 6, activation: "tanh"},
  {type: "softmax", num_classes: 2},
}

t = Trainer n, {}
