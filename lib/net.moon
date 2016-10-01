export *

----------------------------------
-- Net manages a set of layers.
-- For now constraints:
    -- Simple linear order of layers,
    -- first layer input,
    -- last layer: a cost layer.
----------------------------------

class Net
  new: =>
    @layers = {}

  make_layers: (defs) =>
    assert #defs >= 2, "'Net' needs least one input layer and one cost layer!"
    assert type defs[1].type == "input", "first layer of 'Net' must be input!"
    desugar = ->
      new_defs = {}
      for i, d in ipairs defs
        if d.type == "softmax" or d.type == "svm"
          new_defs[#new_defs + 1] = {
            type: "fc",
            num_neurons: d.num_classes,
          }
        if d.type == "regression"
          new_defs[#new_defs + 1] = {
            type: "fc",
            num_neurons: d.num_neurons,
          }
        if (d.type == "fc" or d.type == "conv") and d.bias_pref == nil
          d.bias_pref = 0
          if d.activation == "relu"
            ----------------------------------
            -- RELUs like a bit of positive bias to get gradients early.
            -- Otherwise, technically possible that RELU unit will never turn on (by chance),
            -- and will never get any gradient and never contribute any computation; dead RELU.
            ----------------------------------
            d.bias_pref = 0.1

        new_defs[#new_defs + 1] = d

        if d.activation != nil
          if d.activation == "relu"
            new_defs[#new_defs + 1] = {type: "relu"}
          elseif d.activation == "sigmoid"
            new_defs[#new_defs + 1] = {type: "sigmoid"}
          elseif d.activation == "tanh"
            new_defs[#new_defs + 1] = {type: "tanh"}
          elseif d.activation == "sigmoid"
            new_defs[#new_defs + 1] = {type: "sigmoid"}
          elseif d.activation == "maxout"
            gs = d.group_size or 2
            new_defs[#new_defs + 1] = {type: "maxout", group_size: gs}
          else
            error "[error] trying to use undefined activation '" .. (d.activation or "nil") .. "'"
        if d.type != "dropout" and d.drop_prob != nil
          new_defs[#new_defs + 1] = {type: "dropout", drop_prob: d.drop_prob}
      new_defs
    -- end of 'desugar'
    defs = desugar defs

    -- create layers
    @layers = {}

    for i, d in ipairs defs
      if i > 1
        prev = @layers[i - 1]

        d.in_sx = prev.out_sx
        d.in_sy = prev.out_sy
        d.in_sz = prev.out_sz

      switch d.type
        when "fc"
          @layers[#@layers + 1] = FullyConnLayer d
        when "conv"
          @layers[#@layers + 1] = ConvLayer d
        when "input"
          @layers[#@layers + 1] = InputLayer d
        when "softmax"
          @layers[#@layers + 1] = SoftmaxLayer d
        when "regression"
          @layers[#@layers + 1] = RegressionLayer d
        when "svm"
          @layers[#@layers + 1] = SVMLayer d
        when "dropout"
          @layers[#@layers + 1] = DropoutLayer d
        when "relu"
          @layers[#@layers + 1] = ReluLayer d
        when "sigmoid"
          @layers[#@layers + 1] = SigmoidLayer d
        when "tanh"
          @layers[#@layers + 1] = TanhLayer d
        when "maxout"
          @layers[#@layers + 1] = MaxoutLayer d
        when "lrn"
          @layers[#@layers + 1] = LocalResponseNormalizationLayer d
        else
          error "[error] trying to use undefined layer '" .. d.type .. "'!"
      -- debug/test
      @layers[#@layers + 1] = {out_sx: 1, out_sy: 1, out_sz: 1}

  forward: (vol, train) =>
    act = @layers[1]\forward vol, train
    for i = 2, #@layers
      act = @layers[i]\forward act, train
    act

  get_cost_loss: (vol, y) =>
    @forward vol, false
    N = #@layers
    loss = @layers[N - 1]\backward y
    loss

  -- backpropagation: compute gradients w.r.t. all parameters
  backward: (y) =>
    N = #@layers
    loss = @layers[N - 1]\backward y
    for i = N - 2, 1, -1
      @layers[i]\backward!
    loss

  get_params_and_grads: =>
    response = {}
    for i = 1, #@layers
      layer_response = @layers[i]\get_params_and_grads!
      for j = 1, #layer_response
        response[#response + 1] = layer_response[j]
    response

  get_prediction: =>
    S = @layers[#@layers - 1]

    assert S.layer_type == "softmax", "'get_prediction' assumes 'softmax' as last layer of network!"

    p = S.out_act.w

    max_v = p[1]
    max_i = 0

    for i = 1, #p
      if p[i] > max_v
        max_v = p[i]
        max_i = i
    return max_i

  ----------------------------------
  -- TODO: Add JSON load and save!
  ----------------------------------
