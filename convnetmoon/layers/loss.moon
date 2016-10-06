export *
----------------------------------
-- Layers that implement a loss. Currently these
-- are the layers that can initiate a '\backward!' pass.
-- In the future we probably want a more flexible system
-- that can accomodate multiple losses to do multi-task
-- learning, and stuff like that. But for now, one of the
-- layers in this file must be the final layer in a Net.
----------------------------------
class SoftmaxLayer
  ----------------------------------
  -- This is a classifier, with 'N' discrete classes
  -- from 1 to 'N'. It gets it stream of 'N' incoming
  -- numbers and computes the softmax function.
  ----------------------------------
  new: (opt) =>
    @num_inputs = opt["in_sx"] * opt["in_sy"] * opt["in_depth"]
    @out_depth = @num_inputs
    @out_sx = 1
    @out_sy = 1

    @layer_type = "softmax"

  forward: (V, is_training) =>
    @in_act = V
    A = Vol 1, 1, @out_depth, 0

    -- mac activation
    as = V.w
    max_act = V.w[1]
    for i = 1, @out_depth
      if as[i] > max_act
        max_act = as[i]

    -- compute exponentials (carefully not to blow up)
    @exps = util.zeros @out_depth
    esum = 0
    for i = 1, @out_depth
      e = math.exp as[i] - max_act
      esum += e
      @exps[i] = e

    -- normalize and output to sum to one
    for i = 1, @out_depth
      @exps[i] /= esum
      A.w[i] = @exps[i]

    @es = es
    @out_act = A
    @out_act

  backward: (y) =>
    -- compute and accumulate gradient w.r.t. weights and bias of this layer
    x = @in_act
    x.dw = util.zeros #x.w
    for i = 1, @out_depth
      indicator = 0
      if i == y
        indicator = 1
      mul = -(indicator - @exps[i])
      x.dw[i] = mul
    -math.log @exps[y]

  get_params_and_grads: =>
    {}

  to_JSON: =>
    {
      ["out_depth"]: @out_depth,
      ["out_sx"]: @out_sx,
      ["out_sy"]: @out_sy,
      ["layer_type"]: @layer_type,
      ["num_inputs"]: @num_inputs,
    }

  from_JSON: (json) =>
    @out_depth = json["out_depth"]
    @out_sx = json["out_sx"]
    @out_sy = json["out_sy"]
    @layer_type = json["layer_type"]
    @num_inputs = json["num_inputs"]

class RegressionLayer
  ----------------------------------
  -- Implements an 'L2' regression cost layer,
  -- so penalizes 'sum_i(||x_i - y_i||^2)', where 'x' is its input
  -- and 'y' is the user-provided array of supervising values.
  ----------------------------------
  new: (opt) =>
    @num_inputs = opt["in_sx"] * opt["in_sy"] * opt["in_depth"]

    @out_sx = 1
    @out_sy = 1
    @out_depth = @num_inputs

    @layer_type = "regression"

  forward: (V, is_training) =>
    @in_act = V
    @out_act = V
    V

  backward: (y) =>
    ----------------------------------
    -- 'y' is a list of size 'num_inputs'
    -- compute and accumulate gradient w.r.t. weights and bias
    -- - of this layer.
    ----------------------------------
    x = @in_act
    x.dw = util.zeros #x.w
    loss = 0

    if y["dim"] == nil and y["val"] == nil
      for i = 1, @out_depth
        dy = x.w[i] - y[i]
        x.dw[i] = dy
        loss += 2 * dy^2
    else
      i = y["dim"]
      y_i = y["val"]
      dy = x.w[i] - y_i
      x.dw[i] = dy
      loss += 2 * dy^2
    loss

  get_params_and_grads: =>
    {}

  to_JSON: =>
    {
      ["out_depth"]: @out_depth,
      ["out_sx"]: @out_sx,
      ["out_sy"]: @out_sy,
      ["layer_type"]: @layer_type,
      ["num_inputs"]: @num_inputs,
    }

  from_JSON: (json) =>
    @out_depth = json["out_depth"]
    @out_sx = json["out_sx"]
    @out_sy = json["out_sy"]
    @layer_type = json["layer_type"]
    @num_inputs = json["num_inputs"]

class SVMLayer
  new: (opt) =>
    @num_inputs = opt["in_sx"] * opt["in_sy"] * opt["in_depth"]

    @out_depth = opt["out_depth"]
    @out_sx = 1
    @out_sy = 1

    @layer_type = "svm"

  forward: (V, is_training) =>
    @in_act = V
    @out_act = V
    V

  backward: (V) =>
    -- compute and accumulate gradient w.r.t. weights and bias of this layer
    x = @in_act
    x.dw = util.zeros #x.w

    y_score = x.w[y]
    margin = 1
    loss = 0

    for i = 1, @out_depth
      if -y_score + x.w[i] + margin > 0
        -- 'hinge loss'
        x.dw[i] += 1
        x.dw[y] -= 1

        loss += -y_score + x.w[i] + margin
    loss

  get_params_and_grads: =>
    {}

  to_JSON: =>
    {
      ["out_depth"]: @out_depth,
      ["out_sx"]: @out_sx,
      ["out_sy"]: @out_sy,
      ["layer_type"]: @layer_type,
      ["num_inputs"]: @num_inputs,
    }

  from_JSON: (json) =>
    @out_depth = json["out_depth"]
    @out_sx = json["out_sx"]
    @out_sy = json["out_sy"]
    @layer_type = json["layer_type"]
    @num_inputs = json["num_inputs"]
