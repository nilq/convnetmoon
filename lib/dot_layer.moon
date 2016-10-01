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
              if oy >= 1 and oy < vol_sy and ox >= 1 and ox < vol_sx
                for fd = 0, f.z
                  a += f.w[((f.sx * fy) + fx) * f.z + fd] * vol.w[((vol_sx * oy) * vol.z + fd)]
          a += @biases.w[d]
          A\set ax, ay, d, a
    @out_act = A
    @out_act

  backward: =>
    V = @in_act

    V_sx = bit.bor V.sx, 0
    V_sy = bit.bor V.sy, 0

    xy_stride = bit.bor @stride, 0

    for d = 1, @out_sz
      f = @filter
      x = bit.bor -@pad, 0
      y = bit.bor -@pad, 0

      for ay = 1, @out_sy
        x = bit.bor -@pad
        for ax = 1, @out_sx
          chain_grad = @out_act\get_grad ax, ay, d
          for fy = 1, f.sy
            oy = y + fy
            for fx = 1, f.sx
              ox = x + fx
              if oy >= 1 and oy < vol_sy and ox >= 1 and ox < vol_sx
                for fd = 1, f.sz
                  ix1 = ((V_sx * oy) + ox) * V.sz + fd
                  ix2 = ((f.sx * fy) + fx) * f.sz + fd

                  f.dw[ix2] += V.w[ix1] * chain_grad
                  V.dw[ix1] += f.w[ix2] * chain_grad
          @biases.dw[d] += chain_grad

  get_params_and_grads: =>
    response = {}
    for i = 1, @out_sz
      response[#response + 1] = {
        params: @filters[i].w,
        grads: @filters[i].dw,
        l2_decay_mul: @l2_decay_mul,
        l1_decay_mul: @l1_decay_mul,
      }
    response[#response + 1] = {
      params: @biases.w,
      grads: @biases.dw,
      l2_decay_mul: 0,
      l1_decay_mul: 0,
    }
    response

class FullyConnLayer
  new: (info) =>
    info = info or {}

    @out_sz = info.num_neurons or info.filters

    @l1_decay_mul = info.l1_decay_mul or 0
    @l2_decay_mul = info.l2_decay_mul or 1

    @num_inputs = info.in_sx * info.in_sy * info.in_sz
    @out_sx = 1
    @out_sy = 1
    @layer_type = "fc"

    bias = info.bias_pref or 0
    @filters = {}
    for i = 1, @out_sz
      @filters[#@filters + 1] = Vol 1, 1, @num_inputs
    @biases = Vol 1, 1, @out_sz, bias

  forward: (vol, train) =>
    @in_act = vol
    A = Vol 1, 1, @out_sz, 0

    for i = 1, @out_sz
      a = 0
      wi = @filters[i].w
      for d = 1, @num_inputs
        a += vol.w[d] * wi[d]
      a += @biases.w[i]
      A.w[i] = a
    @out_act = A
    @out_act

  backward: =>
    V = @in_act
    V.dw = list_zeros #V.w -- zero out gradients

    for i = 1, @out_sz
      tfi = @filters[i]
      chain_grad = @out_act.dw[i]
      for d = 1, @num_inputs
        V.dw[d] += tfi.w[d] * chain_grad
        tfi.dw[d] += V.w[d] * chain_grad
      @biases.dw[i] += chain_grad

  get_params_and_grads: =>
    response = {}
    for i = 1, @out_sz
      response[#response + 1] = {
        params: @filters[i].w,
        grads: @filters[i].dw,
        l2_decay_mul: @l2_decay_mul,
        l1_decay_mul: @l1_decay_mul,
      }
    response[#response + 1] = {
      params: @biases.w,
      grads: @biases.dw,
      l2_decay_mul: 0,
      l1_decay_mul: 0,
    }
    response
