do
  local _class_0
  local _base_0 = {
    make_layers = function(self, defs)
      assert(#defs >= 2, "'Net' needs least one input layer and one cost layer!")
      assert(type(defs[1].type == "input", "first layer of 'Net' must be input!"))
      local desugar
      desugar = function()
        local new_defs = { }
        for i, d in ipairs(defs) do
          if d.type == "softmax" or d.type == "svm" then
            new_defs[#new_defs + 1] = {
              type = "fc",
              num_neurons = d.num_classes
            }
          end
          if d.type == "regression" then
            new_defs[#new_defs + 1] = {
              type = "fc",
              num_neurons = d.num_neurons
            }
          end
          if (d.type == "fc" or d.type == "conv") and d.bias_pref == nil then
            d.bias_pref = 0
            if d.activation == "relu" then
              d.bias_pref = 0.1
            end
          end
          new_defs[#new_defs + 1] = d
          if d.activation ~= nil then
            if d.activation == "relu" then
              new_defs[#new_defs + 1] = {
                type = "relu"
              }
            elseif d.activation == "sigmoid" then
              new_defs[#new_defs + 1] = {
                type = "sigmoid"
              }
            elseif d.activation == "tanh" then
              new_defs[#new_defs + 1] = {
                type = "tanh"
              }
            elseif d.activation == "sigmoid" then
              new_defs[#new_defs + 1] = {
                type = "sigmoid"
              }
            elseif d.activation == "maxout" then
              local gs = d.group_size or 2
              new_defs[#new_defs + 1] = {
                type = "maxout",
                group_size = gs
              }
            else
              error("trying to use undefined activation '" .. (d.activation or "nil") .. "'")
            end
          end
          if d.type ~= "dropout" and d.drop_prob ~= nil then
            new_defs[#new_defs + 1] = {
              type = "dropout",
              drop_prob = d.drop_prob
            }
          end
        end
        return new_defs
      end
      defs = desugar(defs)
      self.layers = { }
      for i, d in ipairs(defs) do
        if i > 1 then
          local prev = self.layers[i - 1]
          print(self.layers[1])
          d.in_sx = prev.out_sx
          d.in_sy = prev.out_sy
          d.in_sz = prev.out_sz
        end
        self.layers[#self.layers + 1] = {
          out_sx = 1,
          out_sy = 1,
          out_sz = 1
        }
      end
    end
  }
  _base_0.__index = _base_0
  _class_0 = setmetatable({
    __init = function(self)
      self.layers = { }
    end,
    __base = _base_0,
    __name = "Net"
  }, {
    __index = _base_0,
    __call = function(cls, ...)
      local _self_0 = setmetatable({}, _base_0)
      cls.__init(_self_0, ...)
      return _self_0
    end
  })
  _base_0.__class = _class_0
  Net = _class_0
  return _class_0
end
