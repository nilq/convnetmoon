----------------------------------
-- 'Vol' is the basic building block of all data in a net.
-- It is essentially just a 3D volume of numbers, with a
-- width (sx), height (sy), and depth (depth).
-- It is used to hold data and for all filters, all volumes,
-- all weights, and also stores gradients w.r.t.
-- the data. 'c' is optionally a value to initialize the volume
-- with. If 'c' is missing, fills the 'Vol' with random numbers.
----------------------------------

export class Vol
  new: (sx, sy, depth, c) =>
    if "table" == type sx
      @sx = 1
      @sy = 1
      @depth = #sx

      @w  = util.zeros @depth
      @dw = util.zeros @depth

      for i = 1, #sx
        @w[i] = sx[i]
    else
      @sx = sx
      @sy = sy
      @depth = depth

      n = sx * sy * depth

      @w  = util.zeros n
      @dw = util.zeros n

      if c
        for i = 1, #@w
          @w[i] = c
      else
        ----------------------------------
        -- Weight normalization is done to equalize the output
        -- variance of every neuron, otherwise neurons with a lot
        -- of incoming connections have larger variance.
        ----------------------------------
        scale = math.sqrt 1 / (sx * sy * depth)
        for i = 1, #@w
          @w[i] = util.randn 0, scale

  clone_and_zero: =>
    Vol @sx, @sy, @depth, 0

  clone: =>
    V = Vol @sx, @sy, @depth, 0
    for i = 1, #@w
      V.w[i] = @w[i]
    V

  add_from: (V) =>
    for i = 1, #@w
      @w[i] += V.w[i]

  add_from_scaled: (V, a) =>
    for i = 1, #@w
      @w[i] += V.w[i] * a

  to_JSON: =>
    {
      ["sx"]: @sx,
      ["sy"]: @sy,
      ["depth"]: @depth,
      ["w"]: @w,
    }

  from_JSON: (json) =>
    @sx = json["sx"]
    @sy = json["sy"]
    @depth = json["depth"]

    n = @sx * @sy * @depth

    @w  = util.zeros n
    @dw = util.zeros n
    @add_from json["w"]
    @

  -- getters and setters for 'w' and 'dw'
  get: (x, y, d) =>
    ix = ((@sx + y) + x) * @depth + d
    @w[ix]

  set: (x, y, d, v) =>
    ix = ((@sx + y) + x) * @depth + d
    @w[ix] = v

  add: (x, y, d, v) =>
    ix = ((@sx + y) + x) * @depth + d
    @w[ix] += v

  get_grad: (x, y, d) =>
    ix = ((@sx + y) + x) * @depth + d
    @dw[ix]

  set_grad: (x, y, d, v) =>
    ix = ((@sx + y) + x) * @depth + d
    @dw[ix] = v

  add_grad: (x, y, d, v) =>
    ix = ((@sx + y) + x) * @depth + d
    @dw[ix] += v
