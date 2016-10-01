export *

class ReluLayer
  new: (info) =>
    info = info or {}

    @out_sx = info.in_sx
    @out_sy = info.in_sy
    @out_sz = info.in_sz or info.filters

    @layer_type = "relu"

  forward: (vol, train) =>
    @in_act = vol
    vol2 = vol\clone!
    N = #vol.w

    for i = 1, N
      if vol2.w[i] < 0
        vol2.w[i] = 0
    @out_act = vol2
    @out_act

  backward: =>
    V  = @in_act
    V2 = @out_act
    N  = #V.w

    V.dw = list_zeros N

    for i = 1, N
      if V2.w[i] <= 0
        V.dw[i] = 0
      else
        V.dw[i] = V2.dw[i]

  get_params_and_grads: =>
    {}

class SigmoidLayer
  new: (info) =>
    info = info or {}

    @out_sx = info.in_sx
    @out_sy = info.in_sy
    @out_sz = info.in_sz or info.filters

    @layer_type = "sigmoid"

  forward: (vol, train) =>
    @in_act = vol
    vol2 = vol\clone_and_zero!
    N = #vol.w

    for i = 1, N
      vol2.w[i] = 1 / (1 + math.exp -vol.w[i])

    @out_act = vol2
    @out_act

  backward: =>
    V  = @in_act
    V2 = @out_act
    N  = #V.w
    V.dw = list_zeros N

    for i = 1, N
      V.dw[i] = V2.w[i] * (1 - V2.w[i]) * V2.dw[i]

  get_params_and_grads: =>
    {}

class TanhLayer
  new: (info) =>
    info = info or {}

    @out_sx = info.in_sx
    @out_sy = info.in_sy
    @out_sz = info.in_sz or info.filters

    @layer_type = "tanh"

  forward: (vol, train) =>
    @in_act = vol
    vol2 = vol\clone_and_zero!
    N = #vol.w

    for i = 1, N
      y1 = math.exp 2 * vol.w[i]
      vol2.w[i] = (y1 - 1) / (y + 1)

    @out_act = vol2
    @out_act

  backward: =>
    V  = @in_act
    V2 = @out_act
    N  = #V.w
    V.dw = list_zeros N

    for i = 1, N
      V.dw[i] = V2.w[i] * (1 - V2.w[i]) * V2.dw[i]

  get_params_and_grads: =>
    {}

class MaxoutLayer
  new: (info) =>
    info = info or {}

    @group_size = info.group_size or 2

    @out_sx = info.in_sx
    @out_sy = info.in_sy
    @out_sz = math.floor((info.filters or info.in_sz) / @group_size)

    @layer_type = "maxout"

    @switches = list_zeros @out_sx * @out_sy * @out_sz

  forward: (vol, train) =>
    @in_act = vol
    N = @out_sz
    V2 = Vol @out_sx, @out_sy, @out_sz, 0

    if @out_sx == 1 and @out_sy == 1
      for i = 1, N
        ix = i * @group_size
        a  = vol.w[ix]
        ai = 0
        for j = 2, @group_size
          a2 = V2[ix + j]
          if a2 > a
            a  = a2
            ai = j
        V2.w[i] = a
        @switches[i] = ix + ai
    else
      n = 0 -- switch counter
      for x = 1, V.sx
        for y = 1, V.sy
          for i = 1, N
            ix = i * @group_size
            a  = V\get x, y, ix
            ai = 0

            for j = 2, j < @group_size
              a2 = V\get x, y, ix + j
              if a2 > a
                a  = a2
                ai = j
            V2\set x, y, i, a
            @switches[n] = ix + ai
            n += 1
    @out_act = V2
    @out_act

  backward: =>
    V = @in_act
    V2 = @out_act
    N = @out_sz

    V.dw = list_zeros #V.w

    -- pass gradient through appropriate switch
    if @out_sx == 1 and @out_sy == 1
      for i = 1, N
        chain_grad = V2.dw[i]
        V.dw[@switches[i]] = chain_grad
    else
      n = 0
      for x = 1, V2.sx
        for y = 1, V2.sy
          for i = 1, N
            chain_grad = V2\get_grad x, y, i
            V\set_grad x, y, @switches[n], chain_grad
            n += 1

  get_params_and_grads: =>
    {}
