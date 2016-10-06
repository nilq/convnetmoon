export class PoolingLayer
  ----------------------------------
  -- Max pooling layer: finds areas of max activation
  -- "http://deeplearning.net/tutorial/lenet.html#maxpooling"
  ----------------------------------
  new: (opt) =>
    @sx = opt["sx"] -- filter size

    @in_sx = opt["in_sx"]
    @in_sy = opt["in_sy"]
    @in_depth = opt["in_depth"]

    -- optional
    @sy = util.get_opt opt, "sy", @sx
    @stride = util.get_opt opt, "stride", 2
    @pad = util.get_opt opt, "pad", 0 -- padding to borders of input volume

    @out_sx = math.floor (@in_sx - @sx + 2 * @pad) / @stride + 1
    @out_sy = math.floor (@in_sy - @sy + 2 * @pad) / @stride + 1
    @out_depth = @in_depth

    @layer_type = "pool"

    -- store switches for 'x, y' coordinates for where the max comes from, for each output neuron
    switch_size = @out_sx * @out_sy * @out_depth

    @switch_x = util.zeros switch_size
    @switch_y = util.zeros switch_size

  forward: (V, is_training) =>
    @in_act = V
    A = Vol @out_sx, @out_sy, @out_depth, 0
    switch_count = 0

    for d = 1, @out_depth
      x = -@pad
      y = -@pad

      for ax = 1, @out_sx
        y = -@pad

        for ay = 1, @out_sy
          -- convolve centered at this particular location
          max_a = -1e5

          win_x = -1
          win_y = -1

          for fx = 1, @sx
            for fy = 1, @sy
              off_x = x + fy
              off_y = y + fx

              if off_x >= 0 and off_x < V.sx and off_y >= 0 and off_y < V.sy
                v = V\get off_x, off_y, d

                -- max pool
                if v > max_a
                  max_a = v
                  win_x = off_x
                  win_y = off_y

          @switch_x[switch_count] = win_x
          @switch_y[switch_count] = win_y

          switch_count += 1

          A\set ax, ay, d, max_a

          y += @stride
        x += @stride

    @out_act = A
    @out_act

  backward: =>
    ----------------------------------
    -- Pooling layers have no parameters, so simply compute
    -- gradient w.r.t. data here ...
    ----------------------------------
    V = @in_act
    V.dw = util.zeros #V.w
    A = @out_act

    n = 0
    for d = 1, @out_depth
      x = -@pad
      y = -@pad

      for ax = 1, @out_sx
        y = -@pad

        for ay = 1, @out_sy
          chain_grad = @out_act\get_grad ax, ay, d
          V\add_grad @switch_x[n], @switch_y[n], d, chain_grad

          n += 1
          y += @stride
        x += @stride

  get_params_and_grads: =>
    {}

  to_JSON: =>
    {
      ["sx"]: @sx,
      ["sy"]: @sy,
      ["stride"]: @stride,
      ["in_depth"]: @in_depth,
      ["out_sx"]: @out_sx,
      ["out_sy"]: @out_sy,
      ["out_depth"]: @out_depth,
      ["pad"]: @pad,
      ["layer_type"]: @layer_type,
    }

  from_JSON: (json) =>
    @sx = json["sx"]
    @sy = json["sy"]
    @stride = json["stride"]
    @in_depth = json["in_depth"]
    @out_sx = json["out_sx"]
    @out_sy = json["out_sy"]
    @out_depth = json["out_depth"]
    @pad = json["pad"]
    @layer_type = json["layer_type"]

    switch_size = @out_sx * @out_sy * @out_depth

    @switch_x = util.zeros switch_size
    @switch_y = util.zeros switch_size
