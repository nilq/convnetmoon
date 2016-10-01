export *

class PoolLayer
  new: (info) =>
    info = info or {}

    @sx = info.sx

    @in_sx = info.in_sx
    @in_sy = info.in_sy
    @in_sz = info.in_sz

    @sy = info.sy or @sx
    @stride = info.stride or 2
    @pad = info.pad or 0

    @out_sx = math.floor (@in_sx + @pad * 2 - @sx) / @stride + 1
    @out_sy = math.floor (@in_sy + @pad * 2 - @sx) / @stride + 1
    @out_sz = @in_sz

    @layer_type = "pool"

    @switch_x = list_zeros @out_sx * @out_sy * @out_sz
    @switch_y = list_zeros @out_sx * @out_sy * @out_sz

  forward: (vol, train) =>
    @in_act = vol
    A = Vol @out_sx, @out_sy, @out_sz, 0

    n = 1
    for d = 1, @out_sz
      x = -@pad
      y = -@pad
      for ax = 1, @out_sx
        x += @stride
        y = -@pad
        for ay = 1, @out_sy
          y += @stride

          a = -1e5
          win_x = -1
          win_y = -1

          for fx = 1, @sx
            for dy = 1, @sy
              oy = y + fy
              ox = x + fx

              if oy >= 0 and oy < vol.sy and ox >= 0 and ox < vol.sx
                ----------------------------------
                -- Perform max pooling and store pointers
                -- where the max came from. This will spee up
                -- back propagation and can help make visualizations
                -- .. in the future
                ----------------------------------
                v = vol\get ox, oy, d
                if v > a
                  a = v
                  win_ox = ox
                  win_oy = oy
          @switch_x[n] = win_x
          @switch_y[n] = win_y
          n += 1
          A\set ax, ay, d, a
    @out_act = A
    @out_act

  backward: =>
    V = @in_act
    V.dw = list_zeros #V.w
    A = @out_act

    n = 1
    for d = 1, @out_sz
      x = -@pad
      y = -@pad
      for ax = 1, @out_sx
        x += @stride
        y = -@pad
        for ay = 1, @out_sy
          y += @stride
          chain_grad = @out_act\get_grad ax, ay, d
          V\add_grad @switch_x[n], @switch_y[n], d, chain_grad
          n += 1

  get_params_and_grads: =>
    {}
