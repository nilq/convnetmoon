export class ReluLayer
  ----------------------------------
  -- Implements ReLU nonlinear elementwise.
  -- x -> max(0, x)
  -- the output is in [0, inf)
  ----------------------------------
  new: (opt) =>
    @out_sx = opt["in_sx"]
    @out_sy = opt["in_sy"]
    @out_depth = opt["in_depth"]

    @layer_type = "relu"

  forward: (V, is_training) =>
    @in_act = V
    V2 = V\clone!
    N = #V.w
    V2w = V2.w

    @out_act = V2
    @out_act

  backward: =>
    V = @in_act
    V2 = @out_act
    N = #V.w
    V.dw = util.zeros N -- zero out gradient w.r.t. data

    for i = 1, N
      if V2.w[i] <= 0
        V.dw[i] = 0
      else
        V.dw[i] = V2.dw[i]

  get_params_and_grads: =>
    {}

  to_JSON: =>
    {
      ["out_sx"]: @out_sx,
      ["out_sy"]: @out_sy,
      ["out_depth"]: @out_depth,
      ["layer_type"]: @layer_type,
    }

  from_JSON: (json) =>
    @out_sx = json["out_sx"]
    @out_sy = json["out_sy"]
    @out_depth = json["out_depth"]
