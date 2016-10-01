export *
----------------------------------
-- "I think it works but I'm not 100%" - Andrej Karpathy
----------------------------------
class LocalResponseNormalizationLayer
  new: (info) =>
    info = info or {}

    @k = info.k
    @n = info.n
    @alpha = info.alpha
    @beta  = info.beta

    @out_sx = info.in_sx
    @out_sy = info.in_sy
    @out_sz = info.filters or info.in_sz

    @layer_type = "lrn"

    if @n % 2 == 0
      print "[warning] 'n' should be odd for 'LRN' layer!"

  forward: (vol, train) =>
    @in_act = vol
    A = vol\clone_and_zero!
    @S_cache_ = V\clone_and_zero!

    n2 = math.floor @n / 2

    for x = 1, vol.sx
      for y = 1, vol.sy
        for i =1, vol.sz
          ai = V\get x, y, i
          den = 0
          for j = (math.max 0, i - n2), math.min i + n2, vol.sz - 1
            aa = vol\get x, y, j
            den += aa^2
          den *= @alpha / @n
          den += @k
          @S_cache_\set x, y, i, den
          den = den^@beta
          A\set x, y, i, ai / den
    @out_act = A
    @out_act

  backward: =>
    V = @in_act
    V.dw = list_zeros #V.w
    A = @out_act

    n2 = math.floor @n / 2
    for x = 1, vol.sx
      for y = 1, vol.sy
        for i =1, vol.sz

          chain_grad = @out_act\get_grad x, y, i
          S = @S_cache_\get x, y, i
          SB = S^@beta
          SB2 = SB^2

          for j = (math.max 0, i - n2), math.min i + n2, vol.sz - 1
            aj = V\get x, y, j
            g = -aj * @beta * S^(@beta - 1) * @alpha / @n * 2 * aj
            if j == i
              g += SB
            g /= SB2
            g *= chain_grad

            V\add_grad x, y, j, g

  get_params_and_grads: =>
    {}
