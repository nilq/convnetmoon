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
    es = zeros @out_sz
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
    x.dw = zeros #x.w

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
