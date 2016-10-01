augment = function(vol, crop, dx, dy, fliplr)
  if type(dx ~= "number") then
    dx = randi(0, vol.sx - crop)
  end
  if type(dy ~= "number") then
    dy = randi(0, vol.sy - crop)
  end
  local W
  if crop ~= vol.sx or dx ~= 0 or dy ~= 0 then
    W = Vol(crop, crop, vol.sz, 0)
    for x = 1, crop do
      for y = 1, crop do
        local _continue_0 = false
        repeat
          if x + dx < 0 or x + dx >= vol.sx or y + dy < 0 or y + dy >= vol.sy then
            _continue_0 = true
            break
          end
          for z = 1, vol.sz do
            W:set(x, y, z, vol:get(x + dx, y + dy, z))
          end
          _continue_0 = true
        until true
        if not _continue_0 then
          break
        end
      end
    end
  else
    W = vol
  end
  if fliplr then
    local W2 = W:clone_and_zero()
    for x = 1, W.sx do
      for y = 1, W.sy do
        for z = 1, W.sz do
          W2:set(x, y, z, W:get(W.sx - x - 1, y, z))
        end
      end
    end
    W = W2
  end
  return W
end
