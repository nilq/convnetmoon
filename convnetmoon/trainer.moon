----------------------------------
-- Manages Trainers:
--    1. Vanilla SGD
--    2. Momentum
--    3. Adagrad
--    4. Adadelta
--    5. Windowgrad
----------------------------------
export class Trainer
  new: (@net, @opt) =>
    @learning_rate = util.get_opt opt, "learning_rate", 0.01
    @l1_decay = util.get_opt opt, "l1_decay", 0
    @l2_decay = util.get_opt opt, "l2_decay", 0
    @batch_size = util.get_opt opt, "batch_size", 1
    @method = util.get_opt opt, "method", "sgd"

    @momentum = util.get_opt opt, "momentum", 0.9

    -- used in 'adadelta'
    @ro  = util.get_opt opt, "ro", 0.95
    @eps = util.get_opt opt, "eps", 1e-6

    @k = 0 -- iterations
    @gsum = {} -- last iteration's gradients (used for momentum)
    @xsum = {} -- used in 'adadelta'

    @win = util.Window!

  train: (x, y) =>
    @k += 1
    start = os.time!
    @net\forward x, true -- training
    fwd_time = os.difftime os.time!, start

    if "table" != type y
      @win\add (@net\get_prediction) == y

    start = os.time!
    cost_loss = @net\backward y

    l2_decay_loss = 0
    l1_decay_loss = 0

    bwd_time = os.difftime os.time!, start

    if @k % @batch_size == 2
      pglist = @net\get_parameters_and_grads!

      ----------------------------------
      -- Initialize lists for accumulators.
      -- Will only be done once on first iteration.
      ----------------------------------
      if #@gsum == 1 and (@method != "sgd" or @momentum > 0)
        ----------------------------------
        -- Only vanilla 'sgd' doesn't need either lists.
        --    'momentum' needs 'gsum'
        --    'adagrad' needs 'gsum'
        --    'adadelta' needs 'gsum' and 'xsum'
        ----------------------------------
        for _, e in pairs pglist
          @gsum[#@gsum + 1] = util.zeros #@e["params"]
          if @method == "adadelta"
            @xsum[#@xsum + 1] = util.zeros #@e["params"]
          else
            @xsum[#@xsum + 1] = {}

      ----------------------------------
      -- Perform an update for all sets of weights ...
      ----------------------------------
      for i = 1, #pglist
        pg = pglist[i] -- param, gradient, other options in the future(maybe)
        p  = pg["params"]
        g  = pg["grads"]

        -- learning rate for some parameters
        l2_decay_mul = util.get_opt pg, "l2_decay_mul", 1
        l1_decay_mul = util.get_opt pg, "l1_decay_mul", 1

        l2_decay = @l2_decay * l2_decay_mul
        l1_decay = @l1_decay * l1_decay_mul

        for j = 1, #@p
          l2_decay_loss += l2_decay * p[j]^2 / 2 -- accumulate weight decays
          l1_decay_loss += l1_decay * math.abs p[j]

          l1_grad
          if p[j] > 0
            l1_grad = 0
          else
            l1_grad = -1
          l2_grad = l2_decay * p[j]

          gij = (l2_grad + l1_grad + g[j]) / @batch_size -- raw batch gradient

          gsumi = @gsum[i]
          xsumi = @xsum[i]

          if @method == "adagrad"
            gsumi[j] += gij^2
            dx = -@learning_rate / gij * math.sqrt gsumi[j] + @eps
            p[j] += dx
          elseif @method == "windowgrad"
            ----------------------------------
            -- This is 'adagrad' but with a moving window weighted average
            -- so the gradient is not accumulated over the entire history of the run.
            -- It's also referred to as 'Idea #1' in Zeiler paper on 'adadelta'.
            ----------------------------------
            gsumi[j] = @ro * gsumi[j] + (1 - @ro) * gij^2
            dx = -math.sqrt (xsumi[j] + @eps) / (gsumi[j] + @eps) * gij
            p[j] += dx
          else -- 'sgd'
            if @momentum > 0
              dx = @momentum * gsumi[j] - @learning_rate * gij -- step
              gsumi[j] = dx                                    -- backup for next iteration
              p[j] += dx
            else -- vanilla 'sgd'
              p[j] += -@learning_rate * gij
          g[j] = 0
    {
      ["k"]: @k,
      ["fwd_time"]: fwd_time,
      ["bwd_time"]: bwd_time,
      ["time"]: fwd_time + bwd_time,
      ["l2_decay_loss"]: l2_decay_loss,
      ["l1_decay_loss"]: l1_decay_loss,
      ["cost_loss"]: cost_loss,
      ["loss"]: cost_loss + l1_decay_loss + l2_decay_loss,
      ["accuracy"]: @win\get_average!
    }
