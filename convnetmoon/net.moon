export class Net
  new: (layers) =>
    @layers = {}
    @make_layers layers

  make_layers: (layers) =>
    ----------------------------------
    -- Takes a table of layer definitions and
    -- creates the network layer objects.
    ----------------------------------
    if #layers < 2
      error "[net] layer definitions must contain atleast input and output layer!"
    if layers[1]["type"] != "input"
      error "[net] first layer must be 'input'!"

    add_hidden_layers = =>
      ----------------------------------
      -- Add activations and dropouts.
      ----------------------------------
      new_layers = {}
      for _, l in pairs layers
        layer_type = l["type"]
        if layer_type == "softmax" or layer_type == "svm"
          new_layers[#new_layers + 1] = {
            ["type"]: "fc",
            ["num_neurons"]: l["num_classes"],
          }
        elseif layer_type == "regression"
          new_layers[#new_layers + 1] = {
            ["type"]: "fc",
            ["num_neurons"]: l["num_neurons"],
          }
        elseif ((layer_type == "fc" or layer_type == "conv") and l["bias_pref"] == nil)
          l["bias_pref"] = 0
          if l["activation"] == "relu"
            l["bias_pref"] = 0.1
        elseif layer_type != "capsule"
          new_layers[#new_layers + 1] = l

        if l["activation"]
          l_act = l["activation"]
          if l_act == "relu" or l_act == "sigmoid" or l_act == "tanh" or l_act == "mex"
            new_layers[#new_layers + 1] = {
              ["type"]: l_act,
            }
          elseif l_act == "maxout"
            new_layers[#new_layers + 1] = {
              ["type"]: l_act,
              ["group_size"]: l["group_size"] or 2,
            }
          else
            error "[net] invalid layer activation!"

      if layer_type == "dropout"
        new_layers[#new_layers + 1] = {
          ["type"]: "dropout",
          ["drop_prob"]: l["drop_prob"],
        }

      if layer_type == "capsule"
        fc_recog = {
          ["type"]: "fc",
          ["num_neurons"]: l["num_recog"],
        }
        pose = {
          ["type"]: "add",
          ["delta"]: {l["dx"], l["dy"]},
          ["skip"]: 1,
          ["num_neurons"]: l["num_pose"],
        }
        fc_gen = {
          ["type"]: "fc",
          ["num_neurons"]: l["num_gen"],
        }

        new_layers[#new_layers + 1] = fc_recog
        new_layers[#new_layers + 1] = pose
        new_layers[#new_layers + 1] = fc_gen
      new_layers

    all_layers = add_hidden_layers!

    -- create all the layers!
    for i = 1, #all_layers
      l = all_layers[i]
      if i > 1
        prev = @layers[i - 1]
        l["in_sx"] = prev.out_sx
        l["in_sy"] = prev.out_sy
        l["in_depth"] = prev.out_depth
      layer_type = l["type"]
      layer = {
        out_sx: 10,
        out_sy: 10,
        out_depth: 10,
        to_JSON: =>
          {
            ["out_sx"]: @out_sx,
            ["out_sy"]: @out_sy,
            ["out_depth"]: @out_depth,
          }
      }
      switch layer_type
        when "input"
          print "lol input"
        when "relu"
          print "itsha boi relu"
        when "fc"
          print "yea boi"--layer = FullyConnLayer l
        when "capsule"
          continue
        else
          error "[net] unrecognized layer '" .. layer_type .. "'!"
      if layer
        @layers[#@layers + 1] = layer

  forward: (V, is_training) =>
    ----------------------------------
    -- Forward propagate through the network.
    -- Trainer will pass 'is_training = true'.
    ----------------------------------
    activation = @layers[1]\forward V, is_training
    for i = 2, #@layers
      activation = @layers[i]\forward activation, is_training
    activation

  get_cost_loss: (V, y) =>
    @\forward V, false

  backward: (y) =>
    ----------------------------------
    -- Back propagation: compute gradients w.r.t. all parameters.
    ----------------------------------
    loss = @layers[#@layers]\backward y -- last layer assumed loss
    for i = #@layers - 1, 1, -1
      @layers[i]\backward!
    loss

  get_params_and_grads: =>
    ----------------------------------
    -- Accumulate parameters and gradients for the entire network.
    ----------------------------------
    response = {}
    for i = 1, #@layers
      l_response = @layers[i]\get_params_and_grads!
      for j = 1, #@l_response
        response[#response + 1] = l_response[j]
    response

  get_prediction: =>
    ----------------------------------
    -- This is a 'convenience function' for returning 'argmax
    -- prediction', assuming the last layer of the 'Net' is a 'softmax'.
    ----------------------------------
    softmax = @layers[#@layers]
    p = softmax.out_act.w
    max_i = (util.maxmin p).max_i
    max_i

  to_JSON: =>
    json = {}
    json.layers = {}
    for _, v in pairs @layers
      table.insert json.layers, v\to_JSON!
    json
