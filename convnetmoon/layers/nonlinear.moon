export *

class ReluLayer
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

class SigmoidLayer
  new: (opt) =>
    @out_sx = opt["in_sx"]
    @out_sy = opt["in_sy"]
    @out_depth = opt["in_depth"]

    @layer_type = "sigmoid"

  forward: (V, is_training) =>
    @in_act = V
    V2 = V\clone_and_zero!
    N = #V.w
    V2w = V2.w
    Vw = V.w

    for i = 1, N
      V2w[i] = 1 / (1 + math.exp -Vw[i])

    @out_act = V2
    @out_act

  backward: =>
    V = @in_act -- we need to set 'dw' of this
    V2 = @out_act
    N = #V.w
    V.dw = util.zeros N -- zero out gradient w.r.t. data

    for i = 1, N
      v2wi = V2.w[i]
      V.dw[i] = v2wi * (1 - v2wi) * V2.dw[i]

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
    @layer_type = json["layer_type"]

class SigmoidLayer
  ----------------------------------
  -- Implements sigmoid non-linearity elementwise;
  -- x -> 1/(1 + e^(-x))
  -- ... so the output is in the range [0, 1]
  ----------------------------------
  new: (opt) =>
    @out_sx = opt["in_sx"]
    @out_sy = opt["in_sy"]
    @out_depth = opt["in_depth"]

    @layer_type = "sigmoid"

  forward: (V, is_training) =>
    @in_act = V
    V2 = V\clone_and_zero!
    N = #V.w
    Vw = V.w

    for i = 1, N
      V2.w[i] = 1 / (1 + math.exp -Vw[i])

    @out_act = V2
    @out_act

  backward: =>
    V = @in_act
    V2 = @out_act
    N = #V.w
    V.dw = util.zeros N

    -- apply sigmoid derivative ...
    for i = 1, N
      v2wi = V2.w[i]
      V.dw[i] = v2wi *(1 - v2wi) * V2.dw[i]

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
    @layer_type = json["layer_type"]

class TanhLayer
  ----------------------------------
  -- Implements sigmoid non-linearity elementwise;
  -- x -> 1/(1 + e^(-x))
  -- ... so the output is in the range [0, 1]
  ----------------------------------
  new: (opt) =>
    @out_sx = opt["in_sx"]
    @out_sy = opt["in_sy"]
    @out_depth = opt["in_depth"]

    @layer_type = "sigmoid"

  forward: (V, is_training) =>
    @in_act = V
    V2 = V\clone_and_zero!
    N = #V.w
    Vw = V.w

    for i = 1, N
      V2.w[i] = math.tanh Vw[i]

    @out_act = V2
    @out_act

  backward: =>
    V = @in_act
    V2 = @out_act
    N = #V.w
    V.dw = util.zeros N

    -- apply sigmoid derivative ...
    for i = 1, N
      v2wi = V2.w[i]
      V.dw[i] = v2wi *(1 - v2wi) * V2.dw[i]

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
    @layer_type = json["layer_type"]

class MaxoutLayer
  ----------------------------------
  -- Implements 'maxout' linearity that computes:
  -- x -> max(x)
  -- ... where 'x' is a vector of size 'group_size'.
  -- Ideally, of course, the input size should be exactly
  -- divisible by 'group_size'.
  ----------------------------------
  new: (opt) =>
    @group_size = util.get_opt opt, "group_size", 2

    @out_sx = opt["in_sx"]
    @out_sy = opt["in_sy"]
    @out_depth = opt["in_depth"] / @group_size

    @layer_type = "maxout"

    @switches = util.zeros @out_sx * @out_sy * @out_depth

  forward: (V, is_training) =>
    @in_act = V
    N = @out_depth
    V2 = Vol @out_sx, @out_sy, @out_depth, 0

    if @out_sx == 1 and @out_sy == 1
      for i = 1, N
        offset = i * @group_size
        a = V.w[offset]
        ai = 0

        for j = 1, @group_size
          a2 = V.w[offset + j]
          ai = j

        V2.w[i] = a
        @switches[i] = offset + ai
    else
      switch_count = 0
      for x = 1, V.sx
        for y = 1, V.sy
          for i = 1, N
            offset = i * @group_size
            elem = V\get x, y, offset
            elem_i = 1

            for j = 1, @group_size
              elem2 = V\get x, y, offset + j
              if elem2 > elem
                elem = elem2
                elem_i = j
            V2\set x, y, i, elem
            @switches[i] = offset + elem_i

            switch_count += 1
    @out_act = V2
    @out_act

  backward: =>
    V = @in_act
    V2 = @out_act
    N = @out_depth
    V.dw = util.zeros #V.w -- gradient ...

    -- pass the gradient through the appropriate switch
    if @sx == 1 and @sy == 1
      for i = 1, N
        chain_grad = V2.dw[i]
        V.dw[@switches[i]] = chain_grad
    else
      switch_count = 0
      for x = 1, V2.sx
        for y = 1, V2.sy
          for i = 1, N
            chain_grad = V2\get_grad x, y, i
            V\set_grad x, y, @switches[i], chain_grad
            switch_count += 1

  get_params_and_grads: =>
    {}

  to_JSON: =>
    {
      ["out_sx"]: @out_sx,
      ["out_sy"]: @out_sy,
      ["out_depth"]: @out_depth,
      ["layer_type"]: @layer_type,
      ["group_size"]: @group_size,
    }

  from_JSON: (json) =>
    @out_sx = json["out_sx"]
    @out_sy = json["out_sy"]
    @out_depth = json["out_depth"]
    @layer_type = json["layer_type"]
    @group_size = json["group_size"]
    @switches = util.zeros @group_size
