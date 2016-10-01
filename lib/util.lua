return_v = false
val_v = 0
gaussRandom = function()
  if return_v then
    return_v = false
    return val_v
  end
  local u = 2 * math.random() - 1
  local v = 2 * math.random() - 1
  local r = u ^ 2 + v ^ 2
  if r == 0 or r > 1 then
    return gaussRandom()
  end
  local c = math.sqrt(-2 * math.log(r) / r)
  val_v = v * c
  return_v = true
  return u * c
end
randf = function(a, b)
  return math.random() * (b - a) + a
end
randi = function(a, b)
  return math.floor(math.random() * (b - a) + a)
end
randn = function(mu, std)
  return mu + gaussRandom() * std
end
list_zeros = function(n)
  local _accum_0 = { }
  local _len_0 = 1
  for i = 1, n do
    _accum_0[_len_0] = 0
    _len_0 = _len_0 + 1
  end
  return _accum_0
end
list_contains = function(l, e)
  for k, v in pairs(l) do
    if e == v then
      return true
    end
  end
  return false
end
list_unique = function(l)
  local b = { }
  for i, v in ipairs(l) do
    if not list_contains(b, v) then
      b[i] = v
    end
  end
  return b
end
list_maxmin = function(l)
  local max_v = l[1]
  local min_v = l[1]
  local max_i = 0
  local min_i = 0
  for i, v in ipairs(l) do
    if v > max_v then
      max_v = v
      max_i = i
    elseif v < min_v then
      min_v = v
      min_i = i
    end
  end
  return {
    max_i = max_i,
    max_v = max_v,
    min_i = min_i,
    min_v = min_v,
    dv = max_v - min_v
  }
end
list_randperm = function(n)
  local l = { }
  for i = 1, n do
    l[i] = i
  end
  for i = n, 0, -1 do
    local j = math.floor(math.random() * (i + 1))
    local tmp = l[i]
    l[i] = l[j]
    l[j] = tmp
  end
  return l
end
weighted_sample = function(l1, props)
  local p = randf(0, 1)
  local cumprob = 0
  for i, _ in ipairs(l1) do
    cumprob = cumprob + probs[i]
    if p < cumprob then
      return lst[i]
    end
  end
end
get_opt = function(opt, k, d)
  if type(k == "string") then
    return opt[k] or d
  end
  local ret = d
  for i, v in ipairs(k) do
    if opt[v] then
      ret = opt[v]
    end
  end
  return d
end
