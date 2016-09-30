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
      for i = 1, #defs
        d = defs[i]
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
          error "trying to use undefined activation '" .. d.activation .. "'"
        if d.type != "dropout" and d.drop_prob != nil
          new_defs[#new_defs + 1] = {type: "dropout", drop_prob: d.drop_prob}
        new_defs
      -- end of 'desugar'
      defs = desugar defs

      -- create layers
      @layers = {}

      for i = 1, #defs
        d = defs[i]
        if i > 1
          prev = @layers[i - 1]

          d.in_sx = prev.out_sx
          d.in_sy = prev.out_sy
          d.in_sz = prev.out_sz

        ----------------------------------
        -- TODO: Construction of layers using switch!
        ----------------------------------
