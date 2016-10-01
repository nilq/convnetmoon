export *

----------------------------------
-- An ineffecient dropout layer.
-- This is not the most effecient implementation
-- since the layer before computed all these activations
-- and we now are just going to drop them - same
-- goes for backward pass.
----------------------------------
class DropoutLayer
  new: (info) =>
    info = info or {}

    @out_sx = info.in_sx
    @out_sy = info.in_sy
    @out_sz = info.in_sz or info.filters

    @drop_prob = info.drop_prob or 0.5
    @dropped = list_zeros @out_sx * @out_sy * @out_sz

  forward: (vol, train) =>
    @in_act = vol

    vol2 = table.deepcopy vol
    N = #vol.w

    if train
      for i = 1, N
        if math.random! < @drop_prob
          vol2.w[i] = 0
          @dropped[i] = true
        else
          @dropped[i] = false
    else
      for i = 1, N
        vol2.w[i] *= @drop_prob
    @out_act = vol2
    @out_act

  backward: =>
    chain_grad = @out_act
    N = #@in_act.w
    @in_act.dw = list_zeros N

    for i = 1, N
      if not @dropped[i]
        @in_act.dw[i] = chain_grad.dw[i]

  get_params_and_grads: =>
    {}
