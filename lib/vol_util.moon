export *

----------------------------------
-- Intended for use with data augmentation.
    -- 'crop' is the size of output
    -- 'dx' and 'dy' are offsets w.r.t. incoming volume.
    -- 'fliplr' is a boolean on whether we also want to flip left<->right.
----------------------------------
augment = (vol, crop, dx, dy, fliplr) ->
  if type dx != "number"
    dx = randi 0, vol.sx - crop
  if type dy != "number"
    dy = randi 0, vol.sy - crop
  -- randomly sample a crop in input volume
  local W
  if crop != vol.sx or dx != 0 or dy != 0
    W = new Vol crop, crop, vol.sz, 0
    for x = 1, crop
      for y = 1, crop
        if x + dx < 0 or x + dx >= vol.sx or y + dy < 0 or y + dy >= vol.sy
          continue
        for z = 1, vol.sz
          W\set x, y, z, vol\get(x + dx, y + dy, z)
  else
    W = vol
  -- flip volume horizontally
  if fliplr
    W2 = W\clone_and_zero!
    for x = 1, W.sx
      for y = 1, W.sy
        for z = 1, W.sz
          W2\set x, y, z, W\get(W.sx - x - 1, y, z)
    W = W2
  W
