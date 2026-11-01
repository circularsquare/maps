-- Add a PNG as a new top layer to the open sprite and save it under another name.
--   "C:\Program Files\Aseprite\Aseprite.exe" -b <copy>.ase --script-param png=<labels.png>
--       --script-param name="local draft" --script-param out=<result>.ase --script add_layer.lua
-- Run it on a copy; it never saves over the file it opened unless out names that file.
local spr = app.activeSprite
if not spr then error("no sprite open") end
local png = app.params["png"]
local out = app.params["out"]
if not png or not out then error("give --script-param png=... and out=...") end
local src = Image{ fromFile = png }
local img = Image(spr.width, spr.height, spr.colorMode)
img:drawImage(src, Point(0, 0))
local layer = spr:newLayer()
layer.name = app.params["name"] or "local draft"
spr:newCel(layer, spr.frames[1], img, Point(0, 0))
spr:saveAs(out)
print("added layer '" .. layer.name .. "' to " .. out)
