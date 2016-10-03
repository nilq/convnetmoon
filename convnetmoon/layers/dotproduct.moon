export *

class FullyConnectedLayer
  ----------------------------------
  -- Fully connected dot products, ie. in multi-layer perceptron
  -- networks. Building blocks of most networks.
  ----------------------------------
  new: (opt) =>
    @out_depth = opt["num_neurons"]

    @l2_decay_mul = util.get_opt opt, "l2_decay_mul", 1
    @l1_decay_mul = util.get_opt opt, "l1_decay_mul", 0

    @num_inputs = opt["in_sx"] * opt["in_sy"] * opt["in_depth"]
    @out_sx = 1
    @out_sy = 1

    @layer_type = "fc"

    bias = util.get_opt opt, "bias_pref", 0

    @filters = {}

    for i = 1, #@out_depth
      @filters[#@filters + 1] = Vol 1, 1, @num_inputs
    @biases = Vol 1, 1, @out_depth, bias

  forward: (V) =>
    @in_act = V

    A = Vol 1, 1, @out_depth, 0
    Vw = V.w

    -- dot(W, x) + b
    for i = 1, #@out_depth
      sum_a = 0
      fiw = @filters[i].w
      for d = 1, #@num_inputs
        sum_a += Vw[d] * fiw[d]
      sum_a += @biases.w[i]
      A.w[i] = sum_a

    @out_act = A
    @out_act

  backward: =>
    V = @in_act
    V.dw = util.zeros #@V.w

    ----------------------------------
    -- Compute gradient w.r.t. weights and data
    ----------------------------------
    for i = 1, #@out_depth
      fi = @filters[i]
      chain_grad = @out_act.dw[i]

      for d = 1, @num_inputs
        V.dw[d]  += fi.w[d] * chain_grad -- gradient w.r.t. input data
        fi.dw[d] += V.w[d] * chain_grad  -- gradient w.r.t. parameters

      @biases.dw[i] += chain_grad

  get_params_and_grads: =>
    response = {}
    for d = 1, #@out_depth
      response[#response + 1] = {
        ["params"]: @filters[d].w,
        ["grads"]: @filters[d].dw,
        ["l2_decay_mul"]: @l2_decay_mul,
        ["l1_decay_mul"]: @l1_decay_mul,
      }
    response[#response + 1] = {
      ["params"]: @biases.w,
      ["grads"]: @biases.dw,
      ["l2_decay_mul"]: 0,
      ["l1_decay_mul"]: 0,
    }
    response

  to_JSON: =>
    {
      ["out_depth"]: @out_depth,
      ["out_sx"]: @out_sx,
      ["out_sy"]: @out_sy,
      ["layer_type"]: @layer_type,
      ["num_inputs"]: @num_inputs,
      ["l2_decay_mul"]: @l2_decay_mul,
      ["l1_decay_mul"]: @l1_decay_mul,
      ["filters"]: {v\to_JSON! for _, v in pairs @filters},
      ["biases"]: @biases\to_JSON!,
    }

  from_JSON: (json) =>
    @out_depth = json["out_depth"]
    @out_sx = json["out_sx"]
    @out_sy = json["out_sy"]
    @layer_type = json["layer_type"]
    @num_inputs = json["num_inputs"]
    @l2_decay_mul = json["l2_decay_mul"]
    @l1_decay_mul = json["l1_decay_mul"]
    @filters = [(Vol(0, 0, 0, 0)\from_JSON f) for f in json["filters"]]
    @biases  = Vol(0, 0, 0, 0)\from_JSON json["biases"]
