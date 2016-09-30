bit = require "bit"

export *

class ConvLayer
  new: (info) =>
    info = info or {}

    -- required
    @out_sz = info.filters
    @sx     = info.sx
    @in_sz  = info.in_sz
    @in_sx  = info.in_sx
    @in_sy  = info.in_sy

    -- optional
    @sy = info.sy or @sx
    @stride = info.stride or 1
    @pad = info.pad or 0
    @l1_decay_mul = info.l1_decay_mul or 0
    @l2_decay_mul = info.l1_decay_mul or 1

    -- computed
    @out_sx = math.floor((@in_sx + @pad * 2 - @sx) / @stride + 1)
    @out_sy = math.floor((@in_sy + @pad * 2 - @sx) / @stride + 1)
    @layer_type = "conv"

    -- init
    bias = info.bias_pref or 0

    @filters = {}
    for i = 1, @out_sz
      @filters[#@filters + 1] = Vol @sx, @sy, @in_sz

    @biases = Vol 1, 1, @out_sz, bias

  forward: (vol, train) =>
    @in_act = vol
    A = Vol bit.bor(@out_sx, 0), bit.bor(@out_sy, 0), bit.bor(@out_sy, 0), bit.bor(@out_sz, 0), 0

    vol_sx = bit.bor vol.sx, 0
    vol_sy = bit.bor vol.sy, 0
    xy_stride = bit.bor @stride, 0

    for d = 1, @out_sz
      f = @filters[d]
      x = bit.bor -@pad, 0
      y = bit.bor -@pad, 0
      for ay = 1, @out_sy
        x = bit.bor -@pad, 0
        for ax = 1, @out_sx
          a = 0
          for fy = 1, f.sy
            oy = y + dy
            for fx = 1, f.sx
              ox = x + fx
              if oy >= 0 and oy < vol_sy and ox >= 0 and ox < vol_sx
                for fd = 0, f.z
                  a += f.w[((f.sx * fy) + fx) * f.z + fd] * vol.w[((vol_sx * oy) * vol.z + fd)]
          a += @biases.w[d]
          A\set ax, ay, d, a
    @out_act = A
    @out_act
