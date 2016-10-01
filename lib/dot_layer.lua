local bit = require("bit")
do
  local _class_0
  local _base_0 = {
    forward = function(self, vol, train)
      self.in_act = vol
      local A = Vol(bit.bor(self.out_sx, 0), bit.bor(self.out_sy, 0), bit.bor(self.out_sy, 0), bit.bor(self.out_sz, 0), 0)
      local vol_sx = bit.bor(vol.sx, 0)
      local vol_sy = bit.bor(vol.sy, 0)
      local xy_stride = bit.bor(self.stride, 0)
      for d = 1, self.out_sz do
        local f = self.filters[d]
        local x = bit.bor(-self.pad, 0)
        local y = bit.bor(-self.pad, 0)
        for ay = 1, self.out_sy do
          x = bit.bor(-self.pad, 0)
          for ax = 1, self.out_sx do
            local a = 0
            for fy = 1, f.sy do
              local oy = y + dy
              for fx = 1, f.sx do
                local ox = x + fx
                if oy >= 1 and oy < vol_sy and ox >= 1 and ox < vol_sx then
                  for fd = 0, f.z do
                    a = a + (f.w[((f.sx * fy) + fx) * f.z + fd] * vol.w[((vol_sx * oy) * vol.z + fd)])
                  end
                end
              end
            end
            a = a + self.biases.w[d]
            A:set(ax, ay, d, a)
          end
        end
      end
      self.out_act = A
      return self.out_act
    end,
    backward = function(self)
      local V = self.in_act
      local V_sx = bit.bor(V.sx, 0)
      local V_sy = bit.bor(V.sy, 0)
      local xy_stride = bit.bor(self.stride, 0)
      for d = 1, self.out_sz do
        local f = self.filter
        local x = bit.bor(-self.pad, 0)
        local y = bit.bor(-self.pad, 0)
        for ay = 1, self.out_sy do
          x = bit.bor(-self.pad)
          for ax = 1, self.out_sx do
            local chain_grad = self.out_act:get_grad(ax, ay, d)
            for fy = 1, f.sy do
              local oy = y + fy
              for fx = 1, f.sx do
                local ox = x + fx
                if oy >= 1 and oy < vol_sy and ox >= 1 and ox < vol_sx then
                  for fd = 1, f.sz do
                    local ix1 = ((V_sx * oy) + ox) * V.sz + fd
                    local ix2 = ((f.sx * fy) + fx) * f.sz + fd
                    f.dw[ix2] = f.dw[ix2] + (V.w[ix1] * chain_grad)
                    V.dw[ix1] = V.dw[ix1] + (f.w[ix2] * chain_grad)
                  end
                end
              end
            end
            self.biases.dw[d] = self.biases.dw[d] + chain_grad
          end
        end
      end
    end,
    get_params_and_grads = function(self)
      local response = { }
      for i = 1, self.out_sz do
        response[#response + 1] = {
          params = self.filters[i].w,
          grads = self.filters[i].dw,
          l2_decay_mul = self.l2_decay_mul,
          l1_decay_mul = self.l1_decay_mul
        }
      end
      response[#response + 1] = {
        params = self.biases.w,
        grads = self.biases.dw,
        l2_decay_mul = 0,
        l1_decay_mul = 0
      }
      return response
    end
  }
  _base_0.__index = _base_0
  _class_0 = setmetatable({
    __init = function(self, info)
      info = info or { }
      self.out_sz = info.filters
      self.sx = info.sx
      self.in_sz = info.in_sz
      self.in_sx = info.in_sx
      self.in_sy = info.in_sy
      self.sy = info.sy or self.sx
      self.stride = info.stride or 1
      self.pad = info.pad or 0
      self.l1_decay_mul = info.l1_decay_mul or 0
      self.l2_decay_mul = info.l1_decay_mul or 1
      self.out_sx = math.floor((self.in_sx + self.pad * 2 - self.sx) / self.stride + 1)
      self.out_sy = math.floor((self.in_sy + self.pad * 2 - self.sx) / self.stride + 1)
      self.layer_type = "conv"
      local bias = info.bias_pref or 0
      self.filters = { }
      for i = 1, self.out_sz do
        self.filters[#self.filters + 1] = Vol(self.sx, self.sy, self.in_sz)
      end
      self.biases = Vol(1, 1, self.out_sz, bias)
    end,
    __base = _base_0,
    __name = "ConvLayer"
  }, {
    __index = _base_0,
    __call = function(cls, ...)
      local _self_0 = setmetatable({}, _base_0)
      cls.__init(_self_0, ...)
      return _self_0
    end
  })
  _base_0.__class = _class_0
  ConvLayer = _class_0
end
do
  local _class_0
  local _base_0 = {
    forward = function(self, vol, train)
      self.in_act = vol
      local A = Vol(1, 1, self.out_sz, 0)
      for i = 1, self.out_sz do
        local a = 0
        local wi = self.filters[i].w
        for d = 1, self.num_inputs do
          a = a + (vol.w[d] * wi[d])
        end
        a = a + self.biases.w[i]
        A.w[i] = a
      end
      self.out_act = A
      return out_act
    end,
    backward = function(self)
      local V = self.in_act
      V.dw = zeros(#V.w)
      for i = 1, self.out_sz do
        local tfi = self.filters[i]
        local chain_grad = self.out_act.dw[i]
        for d = 1, self.num_inputs do
          V.dw[d] = V.dw[d] + (tfi.w[d] * chain_grad)
          tfi.dw[d] = tfi.dw[d] + (V.w[d] * chain_grad)
        end
        self.biases.dw[i] = self.biases.dw[i] + chain_grad
      end
    end,
    get_params_and_grads = function(self)
      local response = { }
      for i = 1, self.out_sz do
        response[#response + 1] = {
          params = self.filters[i].w,
          grads = self.filters[i].dw,
          l2_decay_mul = self.l2_decay_mul,
          l1_decay_mul = self.l1_decay_mul
        }
      end
      response[#response + 1] = {
        params = self.biases.w,
        grads = self.biases.dw,
        l2_decay_mul = 0,
        l1_decay_mul = 0
      }
      return response
    end
  }
  _base_0.__index = _base_0
  _class_0 = setmetatable({
    __init = function(self, info)
      info = info or { }
      self.out_sz = info.num_neurons or info.filters
      self.l1_decay_mul = info.l1_decay_mul or 0
      self.l2_decay_mul = info.l2_decay_mul or 1
      self.num_inputs = info.in_sx * info.in_sy * info.in_sz
      self.out_sx = 1
      self.out_sy = 1
      self.layer_type = "fc"
      local bias = info.bias_pref or 0
      self.filters = { }
      for i = 1, self.out_sz do
        self.filters[#self.filters + 1] = Vol(1, 1, self.num_inputs)
      end
      self.biases = Vol(1, 1, self.out_sz, bias)
    end,
    __base = _base_0,
    __name = "FullyConnLayer"
  }, {
    __index = _base_0,
    __call = function(cls, ...)
      local _self_0 = setmetatable({}, _base_0)
      cls.__init(_self_0, ...)
      return _self_0
    end
  })
  _base_0.__class = _class_0
  FullyConnLayer = _class_0
  return _class_0
end
