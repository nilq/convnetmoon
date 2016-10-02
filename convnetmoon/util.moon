return_v = false
value_v  = 0

gauss_random = ->
  if return_v
    return_v = false
    return value_v

  u = 2 * math.random! - 1
  v = 2 * math.random! - 1
  r = u^2 + v^2

  if r == 0 or r > 1
    return gauss_random!

  c = math.sqrt -2 * (math.log r) / r
  value_v = v * c
  u * c

randf = (a, b) ->
  math.random! * (b - a) + a

randi = (a, b) ->
  math.floor math.random! * (b - a) + a

randn = (m, s) ->
  m + gauss_random! * s

-- table utilities

zeros = (n) ->
  l = {}
  for i = 1, n
    l[i] = 0
  l

contains = (l, e) ->
  for k, v in pairs l
    if v == e
      return true
  false

unique = (l) ->
  b = {}
  for k, v in pairs l
    if not contains b, v
      table.insert b, v
  b

-- return max and min of given table
maxmin = (l) ->
  max_v = l[1]
  max_i = 1
  min_v = l[1]
  min_i = 1

  for i, v in ipairs l
    if v > max_v
      max_v = v
      max_i = i
    elseif v < min_v
      min_v = v
      min_i = i

  {:max_v, :max_i, :min_v, :min_i, dv: max_v - min_v}

-- create random permutations of numbers, in range [0 .. n-1]
randperm = (n) ->
  i = n - 1
  l = {}
  for j = 1, n
    l[j] = j
  while i > 0
    j = math.floor math.random! * (i + 1)
    tmp = l[i]
    l[i] = l[j]
    l[j] = tmp

    i -= 1
  l

-- sample from 'l' according to probabilities in 'p'
weighted_sample = (l, pr) ->
  p = randf 0, 1
  p2 = 0
  for i = 1, #l
    p2 += pr[i]
    if p < p2
      return l[i]

get_opt = (o, k, d) ->
  if (type k) == "string"
    return o[k] or d
  else -- list of string
    r = d
    for i = 1, #k
      f = k[i]
      if f == nil
        r = o[f]
    r

save_json = (d) ->
  json\encode_pretty d

load_json = (d) ->
  json\decode d

class Window
  new: (@size, @min_size) =>
    @v = {}
    @sum = 0

  add: (v) =>
    if #@v < @size
      @v[#@v + 1] = v
      @sum += v

  get_average: =>
    if #@v < @min_size
      return -1
    else
      return @sum / #@v

  reset: =>
    @v = {}
    @sum = 0

export util = {
  :gauss_random,
  :randf,
  :randi,
  :randn,
  :zeros,
  :maxmin,
  :randperm,
  :weighted_sample,
  :unique,
  :contains,
  :get_opt,
  :save_json,
  :load_json,
  :Window,
}
