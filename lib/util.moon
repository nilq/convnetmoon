export *

-- random stuff
return_v = false
val_v    = 0

gaussRandom = ->
  if return_v
    return_v = false
    return val_v
  u = 2 * math.random! - 1
  v = 2 * math.random! - 1
  r = u^2 + v^2
  if r == 0 or r > 1
    return gaussRandom!
  c = math.sqrt -2 * math.log(r) / r
  val_v = v * c
  return_v = true
  u * c

randf = (a, b) ->
  math.random! * (b - a) + a
randi = (a, b) ->
  math.floor math.random! * (b - a) + a
randn = (mu, std) ->
  mu + gaussRandom! * std

-- array stuff
list_zeros = (n) ->
  [0 for i = 1, n]

list_contains = (l, e) ->
  for k, v in pairs l
    if e == v
      return true
  false

list_unique = (l) ->
  b = {}
  for i, v in ipairs l
    if not list_contains b, v
      b[i] = v
  b

-- return max and min of list
list_maxmin = (l) ->
  max_v = l[1]
  min_v = l[1]
  max_i = 0
  min_i = 0
  for i, v in ipairs l
    if v > max_v
      max_v = v
      max_i = i
    elseif v < min_v
      min_v = v
      min_i = i
  {:max_i, :max_v, :min_i, :min_v, dv: max_v - min_v}

-- create random permutation of numbers
list_randperm = (n) ->
  l = {}
  for i = 1, n
    l[i] = i
  for i = n, 0, -1
    j = math.floor math.random! * (i + 1)
    tmp = l[i]
    l[i] = l[j]
    l[j] = tmp
  l

-- sample from list 'l1' according to probabilities in list 'probs'
-- the two lists are of same size, pribs adds up to 1
weighted_sample = (l1, props) ->
  p = randf 0, 1
  cumprob = 0
  for i, _ in ipairs l1
    cumprob += probs[i]
    if p < cumprob
      return lst[i]

-- syntactic sugar function for getting default parameters
get_opt = (opt, k, d) ->
  if type k == "string"
    return opt[k] or d
  -- assumes 'k' is a list of strings
  ret = d
  for i, v in ipairs k
    if opt[v]
      ret = opt[v]
  d
