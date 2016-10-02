augment = (V, crop, grayscale) ->
  ----------------------------------
  -- NOTE: Assumes square outputs of size [crop*crop].
  -- Randomly sample a crop in the input volume.
  ----------------------------------
  if crop == V.sx
    return V

  dx = util.randi 0, V.sx - crop
  dy = util.randi 0, V.sy - crop

  W = Vol crop, crop, V.depth

  for x = 1, crop
    for y = 1, crop
      if x + dx < 1 or x + dx >= V.sx or y + dy < 1 or y + dy >= V.sy
        continue
      for d = 1, V.depth
        W\set x, y, d, (V\get x + dx, y + dy, d)

  if grayscale
    G = Vol crop, crop, 1, 0
    for i = 1, crop
      for j = 1, crop
        G\set i, j, 0, (W\get i, j, 0)
    W =G
  W

export vol_util = {
  :augment,
}
