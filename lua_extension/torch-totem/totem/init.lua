require 'torch'

totem = {}

local ondemand = {nn = true}
local mt = {}

--[[ Extends the totem package on-demand.

A sub-package that has not been loaded when totem was initially required can be
added on demand by defining the __index function of totem's metatable. Then
the associated file is being included and the functions defined in it are added
to the totem package.

Arguments:

* `table`, the first argument to the __index function should be self.
* `key`, the name of the sub-package to be included

Returns:

1. a reference to the newly included sub-package.
]]
function mt.__index(table, key)
    if ondemand[key] then
        torch.include('totem', key .. '.lua')
        return totem[key]
    end
end

setmetatable(totem, mt)

torch.include('totem', 'asserts.lua')
torch.include('totem', 'Tester.lua')
torch.include('totem', 'TestSuite.lua')

return totem
