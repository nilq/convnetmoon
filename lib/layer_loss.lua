do
  local _class_0
  local _base_0 = {
    forward = function(self, vol, train)
      self.in_act = vol
      local A = Vol(1, 1, self.out_sz, 0)
      local as = vol.w
      local amax = as[1]
      for i = 2, self.out_sz do
        if as[i] > amax then
          amax = as[i]
        end
      end
      local es = zeros(self.out_sz)
      local esum = 0
      for i = 1, self.out_sz do
        local e = math.exp(as[i] - amax)
        esum = esum + e
        es[i] = e
      end
      for i = 1, self.out_sz do
        es[i] = es[i] / esum
        A.w[i] = es[i]
      end
      self.es = es
      self.out_act = A
      return self.out_act
    end,
    backward = function(self, y)
      local x = self.in_act
      x.dw = zeros(#x.w)
      for i = 1, self.out_sz do
        local indicator
        if i == y then
          indicator = 1
        else
          indicator = 0
        end
        local mul = -(indicator - self.es[i])
        x.dw[i] = mul
      end
      return -math.log(self.es[y])
    end,
    get_params_and_grads = function(self)
      return { }
    end
  }
  _base_0.__index = _base_0
  _class_0 = setmetatable({
    __init = function(self, info)
      info = info or { }
      self.num_inputs = info.in_sx * info.in_sy * info.in_sz
      self.out_sz = self.num_inputs
      self.out_sx = 1
      self.out_sy = 1
      self.layer_type = "softmax"
    end,
    __base = _base_0,
    __name = "SoftmaxLayer"
  }, {
    __index = _base_0,
    __call = function(cls, ...)
      local _self_0 = setmetatable({}, _base_0)
      cls.__init(_self_0, ...)
      return _self_0
    end
  })
  _base_0.__class = _class_0
  SoftmaxLayer = _class_0
end
do
  local _class_0
  local _base_0 = {
    forward = function(self, vol, train)
      self.in_act = vol
      self.out_act = vol
      return vol
    end,
    backwrad = function(self, y)
      local x = self.in_act
      x.dw = zeros(#x.w)
      local loss = 0
      if (type(y)) == "table" then
        for i = 1, self.out_sz do
          local dy = x.w[i] - y[i]
          x.dw[i] = dy
          loss = loss + (0.5 * dy ^ 2)
        end
      elseif (type(y)) == "number" then
        local dy = x.w[1] - y
        x.dw[1] = dy
        loss = loss + (0.5 * dy ^ 2)
      else
        local dy = x.w[i] - y.val
        x.dw[i] = dy
        loss = loss + (0.5 * dy ^ 2)
      end
      return loss
    end,
    get_params_and_grads = function(self)
      return { }
    end
  }
  _base_0.__index = _base_0
  _class_0 = setmetatable({
    __init = function(self, info)
      info = info or { }
      self.num_inputs = info.in_sx * info.in_sy * info.in_sz
      self.out_sz = self.num_inputs
      self.out_sx = 1
      self.out_sy = 1
      self.layer_type = "regression"
    end,
    __base = _base_0,
    __name = "RegressionLayer"
  }, {
    __index = _base_0,
    __call = function(cls, ...)
      local _self_0 = setmetatable({}, _base_0)
      cls.__init(_self_0, ...)
      return _self_0
    end
  })
  _base_0.__class = _class_0
  RegressionLayer = _class_0
end
do
  local _class_0
  local _base_0 = {
    forward = function(self, vol, train)
      self.in_act = vol
      self.out_act = vol
      return vol
    end,
    backwrad = function(self, y)
      local x = self.in_act
      x.dw = zeros(#x.w)
      local y_score = x.w[y]
      local margin = 1
      local loss = 0
      for i = 1, self.out_sz do
        local _continue_0 = false
        repeat
          if i == y then
            _continue_0 = true
            break
          end
          local y_diff = -y_score + x.w[i] + margin
          if y_diff > 0 then
            x.dw[i] = x.dw[i] + 1
            x.dw[y] = x.dw[y] - 1
            loss = loss + y_diff
          end
          _continue_0 = true
        until true
        if not _continue_0 then
          break
        end
      end
      return loss
    end,
    get_params_and_grads = function(self)
      return { }
    end
  }
  _base_0.__index = _base_0
  _class_0 = setmetatable({
    __init = function(self, info)
      info = info or { }
      self.num_inputs = info.in_sx * info.in_sy * info.in_sz
      self.out_sz = self.num_inputs
      self.out_sx = 1
      self.out_sy = 1
      self.layer_type = "svm"
    end,
    __base = _base_0,
    __name = "SVMLayer"
  }, {
    __index = _base_0,
    __call = function(cls, ...)
      local _self_0 = setmetatable({}, _base_0)
      cls.__init(_self_0, ...)
      return _self_0
    end
  })
  _base_0.__class = _class_0
  SVMLayer = _class_0
  return _class_0
end
