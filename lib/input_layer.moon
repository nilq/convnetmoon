export *

class InputLayer
  new: (info) =>
    info = info or {}

    @out_sz = get_opt info, {"out_sz", "sz", "filters"}, 0
    @out_sx = get_opt info, {"out_sx", "sx", "width"}, 1
    @out_sy = get_opt info, {"out_sy", "sy", "height"}, 1

    @layer_type = "input"

  forward: (vol, train) =>
    @in_act = vol
    @out_act = vol
    @out_act

  backward: =>

  get_params_and_grads: =>
    {}
