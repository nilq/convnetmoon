export *
----------------------------------
-- A layer must implement a loss function.
-- Currently these are the layers that can
    -- initiate a \backward! pass
-- In the future we need to be more flexible ..!
----------------------------------

----------------------------------
-- This is a classifier, with 'N' discrete classes from 0 to 'N' - 1.
-- It gets a stream of N incoming numbers and computes the softmax function.
----------------------------------
class SoftmaxLayer
  new: (info) =>
    info = info or {}

    @num_inputs = info.in_sx * info.in_sy * info.in_sz
    @out_sz = @num_inputs
    @out_sx = 1
    @out_sy = 1

    @layer_type = "softmax"

  forward: (vol, train) =>
    @in_act = vol

    A = Vol 1, 1, @out_sz, 0

    -- compute max activation
    as = vol.w
    amax = as[1]
    for i = 2, @out_sz
      if as[i] > amax
        amax = as[i]

    -- carefully compute exponentials
    es = list_zeros @out_sz
    esum = 0
    for i = 1, @out_sz
      e = math.exp as[i] - amax
      esum += e
      es[i] = e

    -- normalize and output to sum to one
    for i = 1, @out_sz
      es[i] /= esum
      A.w[i] = es[i]

    -- save for backpropagation
    @es = es
    @out_act = A
    @out_act

  backward: (y) =>

    x = @in_act
    x.dw = list_zeros #x.w

    for i = 1, @out_sz
      local indicator
      if i == y
        indicator = 1
      else
        indicator = 0

      mul = -(indicator - @es[i])
      x.dw[i] = mul

    -- NLL!
    -math.log @es[y]

  get_params_and_grads: =>
    {}

class RegressionLayer
  new: (info) =>
    info = info or {}

    @num_inputs = info.in_sx * info.in_sy * info.in_sz
    @out_sz = @num_inputs
    @out_sx = 1
    @out_sy = 1

    @layer_type = "regression"

  forward: (vol, train) =>
    @in_act = vol
    @out_act = vol
    vol

  backwrad: (y) =>
    x = @in_act
    x.dw = list_zeros #x.w
    loss = 0
    if (type y) == "table"
      for i = 1, @out_sz
        dy = x.w[i] - y[i]
        x.dw[i] = dy
        loss += 0.5 * dy^2
    elseif (type y) == "number"
      dy = x.w[1] - y
      x.dw[1] = dy
      loss += 0.5 * dy^2
    else
      ----------------------------------
      -- Assume 'y' is a table with keys 'dim' and 'val'.
      -- We pass gradient only along dimension 'dim' to be equal to value 'val'
      ----------------------------------
      dy = x.w[i] - y.val
      x.dw[i] = dy
      loss += 0.5 * dy^2
    loss

  get_params_and_grads: =>
    {}

class SVMLayer
  new: (info) =>
    info = info or {}

    @num_inputs = info.in_sx * info.in_sy * info.in_sz
    @out_sz = @num_inputs
    @out_sx = 1
    @out_sy = 1

    @layer_type = "svm"

  forward: (vol, train) =>
    @in_act = vol
    @out_act = vol
    vol

  backwrad: (y) =>
    x = @in_act
    x.dw = list_zeros #x.w

    ----------------------------------
    -- Using structured loss here, which means that
    -- score of the ground truth should be higher than
    -- the score of any other class by margin.
    ----------------------------------
    y_score = x.w[y]
    margin = 1
    loss = 0
    for i = 1, @out_sz
      if i == y
        continue
      y_diff = -y_score + x.w[i] + margin
      if y_diff > 0
        -- violation of dimensions; apply loss!
        x.dw[i] += 1
        x.dw[y] -= 1
        loss += y_diff
    loss

  get_params_and_grads: =>
    {}
