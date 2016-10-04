export *

class ConvLayer
  ----------------------------------
  -- Performs convolutions: spatial weight sharing.
  ----------------------------------
  new: (opt) =>
    @out_depth = opt["filters"]
    @sx = opt["sx"] -- filter size: should be odd

    @in_depth = opt["in_depth"]
    @in_sx = opt["in_sx"]
    @in_sy = opt["in_sy"]

    -- optional
    @sy = util.get_opt opt, "sy", @sx
    @stride = util.get_opt opt, "stride", 1 -- stride at which we apply filters
    @pad = util.get_opt opt, "stride", 0

    @l2_decay_mul = util.get_opt opt, "l2_decay_mul", 1
    @l1_decay_mul = util.get_opt opt, "l1_decay_mul", 0

    ----------------------------------
    -- NOTE: we are doing floor, so if the strided convolution of the filter doesn't
    -- fit into the input volume exactly, the output will be trimmed and not contain the
    -- (incomplete) computed final application.
    ----------------------------------

    @out_sx = (math.floor @in_sx - @sx + 2 * @pad) / @stride + 1
    @out_sy = (math.floor @in_sy - @sy + 2 * @pad) / @stride + 1

    @layer_type = "conv"

    bias = util.get_opt opt, "bias_pref", 0

    for i = 1, #@out_depth
      @filters[#@filters + 1] = Vol 1, 1, @num_inputs
    @biases = Vol 1, 1, @out_depth, bias

  forward: (V, is_training) =>
    @in_act = V
    A = Vol @out_sx, @out_sy, @out_depth, 0

    v_sx = V.sx
    v_sy = V.sy
    xy_stride = @stride

    for d = 1, @out_depth
      f = @filters[d]
      x = -@pad
      y = -@pad

      for ay = 1, @out_sy
        x = -@pad
        for ax = 1, @out_sx
          -- convolve centered at this particular location
          sum_a = 0
          for fy = 1, f.sy
            off_y = y + fy
            for fx = 1, f.sx
              -- coordinates in the original input array coordinates
              if off_y >= 0 and off_y < V.sy and off_x >= 0 and off_x < V.sx
                for fd = 1, f.depth
                  sum_a += f.w[((f.sx * fy) + fx) * f.depth + fd] * V.w[((v_sx * off_y) + off_x) * V.depth + fd]

          sum_a += @biases.w[d]
          A\set ax, ay, d, sum_a

          x += xy_stride
        y += xy_stride
    @out_act = A
    @out_act

  backward: =>
    -- compute gradient w.r.t. weights, biases and input data
    V = @in_act
    V.dw = util.zeros #V.w

    v_sx = V.sx
    v_sy = V.sy

    xy_stride = @stride

    for d = 1, @out_depth
      f = @filters[d]
      x = -@pad
      y = -@pad

      for ay = 1, @out_sy
        x = -@pad
        for ax = 1, @out_sx
          -- convolve and add up the gradients
          chain_grad = @out_act\get_grad ax, ay, d -- gradient from above
          for fy = 1, f.sy
            off_y = y + fy
            for fx = 1, f.sx
              off_x = x + fx
              if off_y >= 0 and off_y < V.sy and off_x >= 0 and off_x < V.sx
                ix1 = ((v_sx * off_y) + off_y) * V.depth + fd
                ix2 = ((f.sx * fy) + fx) * f.depth + fd

                f.dw[ix2] += V.w[ix1] * chain_grad
                V.dw[ix1] += f.w[ix2] * chain_grad
          @biases.dw[d] += chain_grad
          x += xy_stride
        y += xy_stride

  get_params_and_grads: =>
    response = {}
    for d = 1, @out_depth
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
      ["sx"]: @sx,
      ["sy"]: @sy,
      ["stride"]: @stride,
      ["in_depth"]: @in_depth,
      ["out_depth"]: @out_depth,
      ["out_sx"]: @out_sx,
      ["out_sy"]: @out_sy,
      ["layer_type"]: @layer_type,
      ["l2_decay_mul"]: @l2_decay_mul,
      ["l1_decay_mul"]: @l1_decay_mul,
      ["pad"]: @pad,
      ["filters"]: {f\to_JSON! for _, f in pairs @filters},
      ["biases"]: @biases\to_JSON!,
    }

  from_JSON: (json) =>
    @sx = json["sx"]
    @sy = json["sy"]
    @stride = json["stride"]y
    @in_depth = json["in_depth"]
    @out_depth = json["out_depth"]
    @out_sx = json["out_sx"]
    @out_sy = json["out_sy"]
    @layer_type = json["layer_type"]
    @l2_decay_mul = json["l2_decay_mul"]
    @l1_decay_mul = json["l1_decay_mul"]
    @pad = json["pad"]
    @filters = {(Vol 0, 0, 0, 0)\from_JSON f for _, f in pairs json["filters"]}
    @biases = (Vol 0, 0, 0, 0)\from_JSON json["biases"]

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
