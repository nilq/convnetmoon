export *

----------------------------------
-- Vol is the basic building block of all data in a network.
-- It's essentially just a 3D volume of numbers ...
-- It's used to hold data for all filters, all volumes,
-- all weights, and also stores gradients w.r.t.
-- the data. 'c' is optionally the value to initialize the volume.
-- If 'c' is missing, fills the Vol with randomness.
----------------------------------
class Vol
  new: (sx, sy, sz, c) =>
    -- if, for some reason 'sx' is a 1D table ...
    if type sx == "table"
      @sx = 1
      @sy = 1
      @sz = #sx

      @w  = zeros @sz
      @dw = zeros @sz

      for i, v in ipairs @sz
        @w[i] = v
    else
      @sx = sx
      @sy = sy
      @sz = sz

      n = sx * sy * sz

      @w  = zeros n
      @dw = zeros n

      if not c
        ----------------------------------
        -- weight normalization to equalize the output
        -- variance of every neuron, otherwize neurons with a lot
        -- of incoming connections have outputs of larger variance
        ----------------------------------
        scale = math.sqrt 1 / n
        for i = 1, n do
          @w[i] = randn(0, scale)
      else
        for i = 1, n
          @w[i] = c

    -- prototype!

    get: (x, y, z) =>
      ix = (@sx * y + x) * @sz + z
      @w[ix]

    set: (x, y, z, v) =>
      ix = (@sx * y + x) * @sz + z
      @w[ix] = v

    add: (x, y, z, v) =>
      ix = (@sx * y + x) * @sz + z
      @w[ix] += v

    get_grad: (x, y, z) =>
      ix = (@sx * y + x) * @sz + z
      @dw[ix]

    set_grad: (x, y, z, v) =>
      ix = (@sx * y + x) * @sz + z
      @dw[ix] = v

    add_grad: (x, y, z, v) =>
      ix = (@sx * y + x) * @sz + z
      @dw[ix] += v

    clone_and_zero: () =>
      return new Vol @sx, @sy, @sz, 0

    clone: () =>
      vol = new Vol @sx, @sy, @sz, 0
      for i = 1, #@w
        vol.w[i] = @w[i]
      vol

    add_from: (vol) =>
      for i, v in ipairs vol.w
        @w[i] += v

    add_from_scaled: (vol, s) =>
      for i, v in ipairs vol.w
        @w[i] += v * s

    set_const: (a) =>
      for i = 1, #@w
        @w[i] = a
