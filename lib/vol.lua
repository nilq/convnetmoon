do
  local _class_0
  local _base_0 = { }
  _base_0.__index = _base_0
  _class_0 = setmetatable({
    __init = function(self, sx, sy, sz, c)
      if (type(sx)) == "table" then
        self.sx = 1
        self.sy = 1
        self.sz = #sx
        self.w = list_zeros(self.sz)
        self.dw = list_zeros(self.sz)
        for i, v in ipairs(self.sz) do
          self.w[i] = v
        end
      else
        self.sx = sx
        self.sy = sy
        self.sz = sz
        local n = sx * sy * sz
        self.w = list_zeros(n)
        self.dw = list_zeros(n)
        if not c then
          local scale = math.sqrt(1 / n)
          for i = 1, n do
            self.w[i] = randn(0, scale)
          end
        else
          for i = 1, n do
            self.w[i] = c
          end
        end
      end
      local _ = {
        get = function(self, x, y, z)
          local ix = (self.sx * y + x) * self.sz + z
          return self.w[ix]
        end
      }
      _ = {
        set = function(self, x, y, z, v)
          local ix = (self.sx * y + x) * self.sz + z
          self.w[ix] = v
        end
      }
      _ = {
        add = function(self, x, y, z, v)
          local ix = (self.sx * y + x) * self.sz + z
          self.w[ix] = self.w[ix] + v
        end
      }
      _ = {
        get_grad = function(self, x, y, z)
          local ix = (self.sx * y + x) * self.sz + z
          return self.dw[ix]
        end
      }
      _ = {
        set_grad = function(self, x, y, z, v)
          local ix = (self.sx * y + x) * self.sz + z
          self.dw[ix] = v
        end
      }
      _ = {
        add_grad = function(self, x, y, z, v)
          local ix = (self.sx * y + x) * self.sz + z
          self.dw[ix] = self.dw[ix] + v
        end
      }
      _ = {
        clone_and_zero = function(self)
          return Vol(self.sx, self.sy, self.sz, 0)
        end
      }
      _ = {
        clone = function(self)
          local vol = Vol(self.sx, self.sy, self.sz, 0)
          for i = 1, #self.w do
            vol.w[i] = self.w[i]
          end
          return vol
        end
      }
      _ = {
        add_from = function(self, vol)
          for i, v in ipairs(vol.w) do
            self.w[i] = self.w[i] + v
          end
        end
      }
      _ = {
        add_from_scaled = function(self, vol, s)
          for i, v in ipairs(vol.w) do
            self.w[i] = self.w[i] + (v * s)
          end
        end
      }
      return {
        set_const = function(self, a)
          for i = 1, #self.w do
            self.w[i] = a
          end
        end
      }
    end,
    __base = _base_0,
    __name = "Vol"
  }, {
    __index = _base_0,
    __call = function(cls, ...)
      local _self_0 = setmetatable({}, _base_0)
      cls.__init(_self_0, ...)
      return _self_0
    end
  })
  _base_0.__class = _class_0
  Vol = _class_0
  return _class_0
end
