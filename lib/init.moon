require "lib/util"
require "lib/vol"
require "lib/vol_util"

require "lib/net"
require "lib/trainer"

require "lib/dot_layer"
require "lib/input_layer"
require "lib/dropout_layer"
require "lib/nonlinear_layer"
require "lib/normalization_layer"
require "lib/pool_layer"

require "lib/layer_loss"

math.sign = (n) ->
  if n < 0
    return -1
  elseif n > 0
    return 1
  0
