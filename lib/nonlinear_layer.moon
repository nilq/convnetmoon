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
