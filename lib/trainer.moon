export *

class Trainer
  new: (@net, info) =>
    info = info or {}
    @learning_rate = info.learning_rate or 0.01
    @l1_decay = info.l1_decay or 0
    @l2_decay = info.l2_decay or 0
    @batch_size = info.batch_size or 1
    @method = info.method or "sgd"

    @momentum = info.momentum or 0.9
    @ro = info.ro or 0.95
    @eps = info.eps or 1e-8
    @beta1 = info.beta1 or 0.9
    @beta2 = info.beta2 or 0.999

    @k = 0

    @gsum = {}
    @xsum = {}

    if @net.layers[#@net.layers - 1].layer_type == "regression"
      @regression = true
    else
      @regression = false

  train: (x, y) =>
    start = os.time!
    @net\forward(x, true) -- flag: training = true
    fwd_time = os.difftime os.time!, start

    start = os.time!
    cost_loss = @net\backward y

    l2_decay_loss = 0
    l1_decay_loss = 0

    bck_time = os.difftime os.time!, start

    @k += 1

    if @k % @batch_size == 0
      pg_list = @net\get_params_and_grads!
      if #@gsum == 0 and (@method != "sgd" or @momentum > 0)
        ----------------------------------
        -- Only vanilla SGD doesn't need either lists ...
        -- 'momentum' requires 'gsum'
        -- 'adagrad' requires 'gsum'
        -- 'adam' and 'adadelta' requires 'gsum' and 'xsum'
        ----------------------------------
        for i = 1, @pg_list
          @gsum[#@gsum + 1] = zeros #pg_list[i].params
          if @method == "adam" or @method == "adadelta"
            @xsum[#@xsum + 1] = zeros #pg_list[i].params
          else
            @xsum[#@xsum + 1] = nil
      -- perform an update for all sets of weights
      for i = 1, #pg_list
        pg = pg_list[i]
        p  = pg.params
        g  = pg.grads

        -- learning rate for some params
        l2_decay_mul = pg.l2_decay_mul or 1
        l1_decay_mul = pg.l1_decay_mul or 1

        l2_decay = @l2_decay * l2_decay_mul
        l1_decay = @l1_decay * l1_decay_mul

        for j = 1, #p
          l2_decay_loss += l2_decay * p[j]^2 / 2
          l1_decay_loss += l1_decay * math.abs p[j]

          l1_grad = l1_decay * math.sign p[j]
          l2_grad = l2_decay * p[j]

          gij = (l2_grad + l1_grad + g[j]) / @batch_size

          if @method == "adam"
            @gsum[i][j] *= @beta1 + (1 - @beta1) * gij
            @xsum[i][j] *= @beta2 + (1 - @beta2) * gij^2

            -- correct bias moment estimates
            bias_corr1 = gsum[i][j] * (1 - @beta1^@k)
            bias_corr2 = xsum[i][j] * (1 - @beta2^@k)

            dx = -@learning_rate * bias_corr1 / ((math.sqrt bias_corr2) + @eps)
          elseif @method == "adagrad"
            @gsum[i][j] += gij^2
            dx = -@learning_rate / (math.sqrt @gsum[i][j] + @eps) * gij
            p[j] += dx
          elseif @method = "windowgrad"
            ----------------------------------
            -- Basically 'adagrad' but with a mocing window weighted average,
            -- so the gradient is not accumulated over the entire history of the run.
            -- It's also referred to as 'Idea #1' in Zeiler paper on Adadelta.
            ----------------------------------
            @gsum[i][j] *= @gsum[i][j] * @ro + (1 - @ro) * gij^2
            dx = -@learning_rate / (math.sqrt @gsum[i][j] + @eps) * gij
            p[j] += dx
          elseif @method == "adadelta"
            @gsum[i][j] *= @ro + (1 - @ro) * gij^2
            dx = -math.sqrt((@xsum[i][j] + @eps) / (@gsum[i][j] + @eps)) * gij
            p[j] += dx
          elseif @method == "nesterov"
            dx = @gsum[i][j]
            @gsum[i][j] *= @momentum + @learning_rate * gij
            dx *= @momentum - (1 + @momentum) * @gsum[i][j]

            p[j] += dx
          else
            if @momentum > 0
              dx = @momentum * @gsum[i][j] - @learning_rate * gij
              @gsum[i][j] = dx
              p[j] += dx
            else
              p[j] -= @learning_rate * gij
          g[j] = 0
    {
      :fwd_time,
      :bck_time,
      :l2_decay_loss,
      :l1_decay_loss,
      :cost_loss,

      softmax_loss: cost_loss,
      loss: cost_loss + l1_decay_loss + l2_decay_loss
    }
