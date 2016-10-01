do
  local _class_0
  local _base_0 = {
    train = function(self, x, y)
      local start = os.time()
      self.net:forward(x, true)
      local fwd_time = os.difftime(os.time(), start)
      start = os.time()
      local cost_loss = self.net:backward(y)
      local l2_decay_loss = 0
      local l1_decay_loss = 0
      local bck_time = os.difftime(os.time(), start)
      self.k = self.k + 1
      if self.k % self.batch_size == 0 then
        local pg_list = self.net:get_params_and_grads()
        if #self.gsum == 0 and (self.method ~= "sgd" or self.momentum > 0) then
          for i = 1, self.pg_list do
            self.gsum[#self.gsum + 1] = zeros(#pg_list[i].params)
            if self.method == "adam" or self.method == "adadelta" then
              self.xsum[#self.xsum + 1] = zeros(#pg_list[i].params)
            else
              self.xsum[#self.xsum + 1] = nil
            end
          end
        end
        for i = 1, #pg_list do
          local pg = pg_list[i]
          local p = pg.params
          local g = pg.grads
          local l2_decay_mul = pg.l2_decay_mul or 1
          local l1_decay_mul = pg.l1_decay_mul or 1
          local l2_decay = self.l2_decay * l2_decay_mul
          local l1_decay = self.l1_decay * l1_decay_mul
          for j = 1, #p do
            l2_decay_loss = l2_decay_loss + (l2_decay * p[j] ^ 2 / 2)
            l1_decay_loss = l1_decay_loss + (l1_decay * math.abs(p[j]))
            local l1_grad = l1_decay * math.sign(p[j])
            local l2_grad = l2_decay * p[j]
            local gij = (l2_grad + l1_grad + g[j]) / self.batch_size
            if self.method == "adam" then
              self.gsum[i][j] = self.gsum[i][j] * (self.beta1 + (1 - self.beta1) * gij)
              self.xsum[i][j] = self.xsum[i][j] * (self.beta2 + (1 - self.beta2) * gij ^ 2)
              local bias_corr1 = gsum[i][j] * (1 - self.beta1 ^ self.k)
              local bias_corr2 = xsum[i][j] * (1 - self.beta2 ^ self.k)
              local dx = -self.learning_rate * bias_corr1 / ((math.sqrt(bias_corr2)) + self.eps)
            elseif self.method == "adagrad" then
              self.gsum[i][j] = self.gsum[i][j] + (gij ^ 2)
              local dx = -self.learning_rate / (math.sqrt(self.gsum[i][j] + self.eps)) * gij
              p[j] = p[j] + dx
            else
              do
                self.method = "windowgrad"
                if self.method then
                  self.gsum[i][j] = self.gsum[i][j] * (self.gsum[i][j] * self.ro + (1 - self.ro) * gij ^ 2)
                  local dx = -self.learning_rate / (math.sqrt(self.gsum[i][j] + self.eps)) * gij
                  p[j] = p[j] + dx
                elseif self.method == "adadelta" then
                  self.gsum[i][j] = self.gsum[i][j] * (self.ro + (1 - self.ro) * gij ^ 2)
                  local dx = -math.sqrt((self.xsum[i][j] + self.eps) / (self.gsum[i][j] + self.eps)) * gij
                  p[j] = p[j] + dx
                elseif self.method == "nesterov" then
                  local dx = self.gsum[i][j]
                  self.gsum[i][j] = self.gsum[i][j] * (self.momentum + self.learning_rate * gij)
                  dx = dx * (self.momentum - (1 + self.momentum) * self.gsum[i][j])
                  p[j] = p[j] + dx
                else
                  if self.momentum > 0 then
                    local dx = self.momentum * self.gsum[i][j] - self.learning_rate * gij
                    self.gsum[i][j] = dx
                    p[j] = p[j] + dx
                  else
                    p[j] = p[j] - (self.learning_rate * gij)
                  end
                end
              end
            end
            g[j] = 0
          end
        end
      end
      return {
        fwd_time = fwd_time,
        bck_time = bck_time,
        l2_decay_loss = l2_decay_loss,
        l1_decay_loss = l1_decay_loss,
        cost_loss = cost_loss,
        softmax_loss = cost_loss,
        loss = cost_loss + l1_decay_loss + l2_decay_loss
      }
    end
  }
  _base_0.__index = _base_0
  _class_0 = setmetatable({
    __init = function(self, net, info)
      self.net = net
      info = info or { }
      self.learning_rate = info.learning_rate or 0.01
      self.l1_decay = info.l1_decay or 0
      self.l2_decay = info.l2_decay or 0
      self.batch_size = info.batch_size or 1
      self.method = info.method or "sgd"
      self.momentum = info.momentum or 0.9
      self.ro = info.ro or 0.95
      self.eps = info.eps or 1e-8
      self.beta1 = info.beta1 or 0.9
      self.beta2 = info.beta2 or 0.999
      self.k = 0
      self.gsum = { }
      self.xsum = { }
      if self.net.layers[#self.net.layers - 1].layer_type == "regression" then
        self.regression = true
      else
        self.regression = false
      end
    end,
    __base = _base_0,
    __name = "Trainer"
  }, {
    __index = _base_0,
    __call = function(cls, ...)
      local _self_0 = setmetatable({}, _base_0)
      cls.__init(_self_0, ...)
      return _self_0
    end
  })
  _base_0.__class = _class_0
  Trainer = _class_0
  return _class_0
end
