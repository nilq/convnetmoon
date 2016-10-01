do
  local _class_0
  local _base_0 = {
    forward = function(self, vol, train)
      self.in_act = vol
      self.out_act = vol
      return self.out_act
    end,
    backward = function(self) end,
    get_params_and_grads = function(self)
      return { }
    end
  }
  _base_0.__index = _base_0
  _class_0 = setmetatable({
    __init = function(self, info)
      info = info or { }
      self.out_sz = get_opt(info, {
        "out_sz",
        "sz",
        "filters"
      }, 0)
      self.out_sx = get_opt(info, {
        "out_sx",
        "sx",
        "width"
      }, 1)
      self.out_sy = get_opt(info, {
        "out_sy",
        "sy",
        "height"
      }, 1)
      self.layer_type = "input"
    end,
    __base = _base_0,
    __name = "InputLayer"
  }, {
    __index = _base_0,
    __call = function(cls, ...)
      local _self_0 = setmetatable({}, _base_0)
      cls.__init(_self_0, ...)
      return _self_0
    end
  })
  _base_0.__class = _class_0
  InputLayer = _class_0
  return _class_0
end
