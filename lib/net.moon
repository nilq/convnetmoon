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
          
