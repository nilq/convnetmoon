export class DropoutLayer
  ----------------------------------
  -- Randomly omits half the feature detectors on each training case.
  -- Each neuron tends to learn something 'useful'.
  -- "http://arxiv.org/pdf/1207.0580.pdf"
  ----------------------------------
  new: (opt) =>
    @out_sx = opt["in_sx"]
    @out_sy = opt["in_sy"]
    @out_depth = opt["in_depth"]

    @layer_type = "dropout"

    @drop_prob = util.get_opt opt, "drop_prob", 0.5
    @dropped = util.zeros @out_sx * @out_sy * @out_depth

  forward: (V, is_training) =>
    @in_act = V
    V2 = V\clone!
    N  = #V.w

    if is_training
      -- do dropout
      for i = 1, N
        if math.random! < @drop_prob
          V2.w[i] = 0
          @dropped[i] = true
        else
          @dropped = false
    else
      -- scale the activations during prediction
      for i = 1, N
        V2.w[i] *= @drop_prob

    @out_act = V2
    @out_act

  backward: =>
    V = @in_act
    chain_grad = @out_act

    N = #V.w
    V.dw = util.zeros N -- zero out gradient ...

    for i = 1, N
      if not @dropped[i]
        V.dw[i] = chain_grad.dw[i] -- copy over the gradient

  to_JSON: =>
    {
      ["out_sx"]: @out_sx,
      ["out_sy"]: @out_sy,
      ["out_depth"]: @out_depth,
      ["layer_type"]: @layer_type,
      ["drop_prob"]: @drop_prob,
    }

  from_JSON: (json) =>
    @out_sx = json["out_sx"]
    @out_sy = json["out_sy"]
    @out_depth = json["out_depth"]
    @layer_type = json["layer_type"]
    @drop_prob = json["drop_prob"]
