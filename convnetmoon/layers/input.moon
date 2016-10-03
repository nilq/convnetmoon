export class InputLayer
  new: (opt) =>
    @out_sx = opt["out_sx"]
    @out_sy = opt["out_sy"]
    @out_depth = opt["out_depth"]

    @layer_type = "input"

  forward: (V, is_training) =>
    @in_act = V
    @out_act = V
    @out_act

  backward: =>
  get_params_and_grads: =>

  to_JSON: =>
    {
      ["out_depth"]: @out_depth,
      ["out_sx"]: @sx,
      ["out_sy"]: @sy,
      ["layer_type"]: @layer_type,
    }

  from_JSON: (json) =>
    @out_depth = json["out_depth"]
    @out_sx = json["out_sx"]
    @out_sy = json["out_sy"]
    @layer_type = json["layer_type"]
