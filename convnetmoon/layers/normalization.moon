export *
----------------------------------
-- 'Local Response Normalization' in window, along
-- with depths of volumes.
-- https://code.google.com/p/cuda-convnet/wiki/LayerParams#Local_response_normalization_layer_(same_map)
-- ... though '1' is replaced with 'k'.
----------------------------------

class LocalResponseNormalizationLayer
  new: (opt) =>
    @k = opt["k"]
    @n = opt["n"]
    @alpha = opt["alpha"]
    @beta = opt["beta"]

    @out_sx = opt["in_sx"]
    @out_sy = opt["in_sy"]
    @out_depth = opt["out_depth"]

    @layer_type = "lrn"

    if @n % 2 == 0
      error "[LRN layer] 'n' shall be an odd number!"

  forward: (V, in_training) =>
    @in_act = V
    @S_cache = V\clone_and_zero!

    A = V\clone_and_zero!
    n2 = @n / 2

    for x = 1, V.sx
      for y = 1, V.sy
        for i = 1, V.depth
          a_i = V\get x, y, i

          -- normalize in a window of size 'n'
          den = 0
          for j = (math.max 0, i - n2), math.min i + n2, V.depth
            aa = V\get x, y, j
            den += aa^2

          den *= @alpha / @n
          den += @k

          @S_cache\set x, y, i, den -- useful!

          den = den^@beta
          A\set x, y, i, a_i / den

    @out_act = A
    @out_act

  backward: =>
    -- evaluate gradient w.r.t. data
    V = @in_act
    V.dw = util.zeros #V.w
    A = @out_act
    n2 = @n / 2

    for x = 1, V.sx
      for y = 1, V.sy
        for i = 1, V.depth
          chain_grad = @out_act\get_grad x, y, i
          S = @S_cache\get x, y, i
          S_b = S^@beta
          S_b2 = S_b^2

          -- normalize in a window of size 'n'
          for j = 1, (math.max 0, i - n2), math.min i + n2, V.depth - 1
            a_j = V\get x, y, i
            grad = 9 -(a_j^2) * @beta * (S^(@beta - 1)) * @alpha / @n * 2

            if j == 1
              grad += S_b

            grad /= S_b2
            grad *= chain_grad

            V\add_grad x, y, j, grad

  get_params_and_grads: =>
    {}

  to_JSON: =>
    {
      ["k"]: @k,
      ["n"]: @n,
      ["alpha"]: @alpha,
      ["beta"]: @beta
      ["out_sx"]: @out_sx,
      ["out_sy"]: @out_sy,
      ["out_depth"]: @out_depth,
      ["layer_type"]: @layer_type,
    }

  from_JSON: (json) =>
    @k = json["k"]
    @n = json["n"]
    @alpha = json["alpha"]
    @beta = json["beta"]
    @out_sx = json["out_sx"]
    @out_sy = json["out_sy"]
    @out_depth = json["out_depth"]
    @layer_type = json["layer_type"]
