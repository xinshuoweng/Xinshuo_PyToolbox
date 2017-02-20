-- Functions for checking Tensor, Storage and Table equality.

--[[ Tests for tensor equality between two tensors of matching sizes and types

Tests whether the maximum element-wise difference between `ta` and `tb` is less
than or equal to `tolerance`.

Arguments:

* `ta` (tensor)
* `tb` (tensor)
* `tolerance` (optional number, default 0) maximum elementwise difference
    between `ta` and `tb`.
* `negate` (optional boolean, default false) if `negate` is true, we invert
    success and failure.
* `storage` (optional boolean, default false) if `storage` is true, we print an
    error message referring to Storages rather than Tensors.

Returns:

1. success, boolean that indicates success
2. failure_message, string or nil
]]
local function areSameFormatTensorsEq(ta, tb, tolerance, negate, storage)

  local function ensureHasAbs(t)
    -- Byte, Char and Short Tensors don't have abs
    return t.abs and t or t:double()
  end

  ta = ensureHasAbs(ta)
  tb = ensureHasAbs(tb)

  local diff = ta:clone():add(-1, tb):abs()
  local err = diff:max()
  local success = err <= tolerance
  if negate then
    success = not success
  end

  local errMessage
  if not success then
    local prefix = storage and 'Storage' or 'Tensor'
    local violation = negate and 'NE(==)' or 'EQ(==)'
    errMessage = string.format('%s%s violation: max diff=%s, tolerance=%s',
                               prefix,
                               violation,
                               tostring(err),
                               tostring(tolerance))
  end

  return success, errMessage
end


--[[ Tests for tensor equality

Tests whether the maximum element-wise difference between `ta` and `tb` is less
than or equal to `tolerance`. Throws an error if one of its two first arguments
is not a tensor.

Arguments:

* `ta` (tensor)
* `tb` (tensor)
* `tolerance` (optional number, default 0) maximum elementwise difference
    between `ta` and `tb`.
* `negate` (optional boolean, default false) if negate is true, we invert
    success and failure.

Returns:

1. success, boolean that indicates success
2. failure_message, string or nil
]]
function totem.areTensorsEq(ta, tb, tolerance, negate)
  negate = negate or false
  tolerance = tolerance or 0
  assert(torch.isTensor(ta), "First argument should be a Tensor")
  assert(torch.isTensor(tb), "Second argument should be a Tensor")
  assert(type(tolerance) == 'number',
         "Third argument should be a number describing a tolerance on"
         .. " equality for a single element")

  if ta:dim() ~= tb:dim() then
    return negate, 'The tensors have different dimensions'
  end

  if ta:type() ~= tb:type() then
    return negate, 'The tensors have different types'
  end

  -- If we are comparing two empty tensors, return true.
  -- This is needed because some functions below cannot be applied to tensors
  -- of dimension 0.
  if ta:dim() == 0 then
    return not negate, 'Both tensors are empty'
  end

  if not ta:isSameSizeAs(tb) then
    return negate, 'Both tensors are empty'
  end

  return areSameFormatTensorsEq(ta, tb, tolerance, negate, false)

end


--[[ Asserts tensor equality.

Asserts that the maximum elementwise difference between `ta` and `tb` is less
than or equal to `tolerance`. Fails if one of its two first arguments is not a
tensor.

Arguments:

* `ta` (tensor)
* `tb` (tensor)
* `tolerance` (optional number, default 0) maximum elementwise difference
    between `a` and `b`.
]]
function totem.assertTensorEq(ta, tb, tolerance)
  return assert(totem.areTensorsEq(ta, tb, tolerance))
end


--[[ Tests for tensor inequality

The tensors are considered unequal if the maximum elementwise difference >
`tolerance`. Throws an error if one of its two first arguments is not a tensor.

Arguments:

* `ta` (tensor)
* `tb` (tensor)
* `tolerance` (optional number, default 0).

Returns:
1. success, a boolean indicating success
2. failure_message, string or nil
]]
function totem.areTensorsNe(ta, tb, tolerance)
  return totem.areTensorsEq(ta, tb, tolerance, true)
end


--[[ Asserts tensor inequality.

The tensors are considered unequal if the maximum elementwise difference >
`tolerance`. Fails if one of its two first arguments is not a tensor.

Arguments:

* `ta` (tensor)
* `tb` (tensor)
* `tolerance` (optional number, default 0).
]]
function totem.assertTensorNe(ta, tb, tolerance)
  return assert(totem.areTensorsNe(ta, tb, tolerance))
end


local typesMatching = {
    ['torch.ByteStorage'] = torch.ByteTensor,
    ['torch.CharStorage'] = torch.CharTensor,
    ['torch.ShortStorage'] = torch.ShortTensor,
    ['torch.IntStorage'] = torch.IntTensor,
    ['torch.LongStorage'] = torch.LongTensor,
    ['torch.FloatStorage'] = torch.FloatTensor,
    ['torch.DoubleStorage'] = torch.DoubleTensor,
}


--[[ Tests for storage equality

Tests whether the maximum element-wise difference between `sa` and `sb` is less
than or equal to `tolerance`. Throws an error if one of its two first arguments
is not a storage.

Arguments:

* `sa` (storage)
* `sb` (storage)
* `tolerance` (optional number, default 0) maximum elementwise difference
    between `a` and `b`.
* `negate` (optional boolean, default false) if negate is true, we invert
    success and failure.

Returns:

1. success, boolean that indicates success
2. failure_message, string or nil
]]
function totem.areStoragesEq(sa, sb, tolerance, negate)
  -- If negate is true, we invert success and failure
  negate = negate or false
  tolerance = tolerance or 0
  assert(torch.isStorage(sa), "First argument should be a Storage")
  assert(torch.isStorage(sb), "Second argument should be a Storage")
  assert(type(tolerance) == 'number',
         "Third argument should be a number describing a tolerance on"
         .. " equality for a single element")

  if sa:size() ~= sb:size() then
    return negate, 'The storages have different sizes'
  end

  local typeOfsa = torch.type(sa)
  local typeOfsb = torch.type(sb)

  if typeOfsa ~= typeOfsb then
    return negate, 'The storages have different types'
  end

  local ta = typesMatching[typeOfsa](sa)
  local tb = typesMatching[typeOfsb](sb)

  return areSameFormatTensorsEq(ta, tb, tolerance, negate, true)
end


--[[ Asserts storage equality.

Asserts that the maximum elementwise difference between `sa` and `sb` is less
than or equal to `tolerance`. Fails if one of its two first arguments is not a
storage.

Arguments:

* `sa` (storage)
* `sb` (storage)
* `tolerance` (optional number, default 0) maximum elementwise difference
    between `a` and `b`.
]]
function totem.assertStorageEq(sa, sb, tolerance)
  return assert(totem.areStoragesEq(sa, sb, tolerance))
end


--[[ Tests for storage inequality

The storages are considered unequal if the maximum elementwise difference >
`tolerance`. Throws an error if one of its two first arguments is not a storage.

Arguments:

* `sa` (storage)
* `sb` (storage)
* `tolerance` (optional number, default 0).

Returns:
1. success, a boolean indicating success
2. failure_message, string or nil
]]
function totem.areStoragesNe(sa, sb, tolerance)
  return totem.areStoragesEq(sa, sb, tolerance, true)
end


--[[ Asserts storage inequality.

The storages are considered unequal if the maximum elementwise difference >
`tolerance`. Fails if one of its two first arguments is not a storage.

Arguments:

* `sa` (storage)
* `sb` (storage)
* `tolerance` (optional number, default 0).
]]
function totem.assertStorageNe(sa, sb, tolerance)
  return assert(totem.areStoragesNe(sa, sb, tolerance))
end


--[[ Tests for general (deep) equality

The types of `got` and `expected` must match.
Tables are compared recursively. Keys and types of the associated values must
match, recursively. Numbers are compared with the given tolerance.
Torch tensors and storages are compared with the given tolerance on their
elementwise difference. Other types are compared for strict equality with the
regular Lua == operator.

Arguments:

* `got`
* `expected`
* `tolerance` (optional number, default 0) maximum elementwise difference
    between `a` and `b`.
* `negate` (optional boolean, default false) if negate is true, we invert
    success and failure.

Returns:

1. success, boolean that indicates success
2. failure_message, string or nil
]]
function totem.areEq(got, expected, tolerance, negate)
    negate = negate or false
    tolerance = tolerance or 0
    local errMessage
    local violation = negate and 'NE(==)' or 'EQ(==)'
    if type(got) ~= type(expected) then
      if not negate then
        errMessage = 'Arguments are not of the same type (got: ' .. type(got) ..
            ', expected: ' .. type(expected) .. ')'
      end
      return negate, errMessage
    elseif type(got) == 'number' then
        local diff = math.abs(got - expected)
        local ok = (diff <= tolerance)
        if negate then
          ok = not ok
        end
        if not ok then
          errMessage = string.format('%s%s violation: max diff=%s,' ..
                                      ' tolerance=%s',
                                      type(got),
                                      violation,
                                      tostring(diff),
                                      tostring(tolerance))
        end
        return ok, errMessage
    elseif type(expected) == "table" then
      return totem.areTablesEq(got, expected, tolerance, negate)
    elseif torch.isTensor(got) then
      return totem.areTensorsEq(got, expected, tolerance, negate)
    elseif torch.isStorage(got) then
      return totem.areStoragesEq(got, expected, tolerance, negate)
    else
    -- Below: we have the same type which is either userdata or a lua type
    -- which is not a number.
      local ok = (got == expected)
      if negate then
        ok = not ok
      end
      if not ok then
        errMessage = string.format('%s%s violation: val1=%s, val2=%s',
                          type(got),
                          violation,
                          tostring(got),
                          tostring(expected))
      end
      return ok, errMessage
    end
end


--[[ Tests for (deep) table equality

Tables are compared recursively. Keys and types of the associated values must
match, recursively. Numbers are compared with the given tolerance.
Torch tensors and storages are compared with the given tolerance on their
elementwise difference. Other types are compared for strict equality with the
regular Lua == operator. Throws an error if one of its two first arguments is
not a table.

Arguments:

* `t1` (table)
* `t2` (table)
* `tolerance` (optional number, default 0) maximum elementwise difference
    between `a` and `b`.
* `negate` (optional boolean, default false) if negate is true, we invert
    success and failure.

Returns:

1. success, boolean that indicates success
2. failure_message, string or nil
]]
function totem.areTablesEq(t1, t2, tolerance, negate)
  negate = negate or false
  tolerance = tolerance or 0
  assert(type(t1) == 'table', "First argument should be a Table")
  assert(type(t2) == 'table', "Second argument should be a Table")
  assert(type(tolerance) == 'number',
         "Third argument should be a number describing a tolerance on"
         .. " equality for a single element")

  for k, v in pairs(t2) do
    local ok, message = totem.areEq(t1[k], v, tolerance, false)
      if not ok then
        return negate, message
      end
  end

  for k, v in pairs(t1) do
    local ok, message = totem.areEq(v, t2[k], tolerance, false)
      if not ok then
        return negate, message
      end
  end

  return not negate, 'The tables are equal.'
end


--[[ Asserts (deep) table equality.

Tables are compared recursively. Keys and types of the associated values must
match, recursively. Numbers are compared with the given tolerance.
Torch tensors and storages are compared with the given tolerance on their
elementwise difference. Other types are compared for strict equality with the
regular Lua == operator. Fails if one of its two first arguments is not a table.

Arguments:

* `ta` (table)
* `tb` (table)
* `tolerance` (optional number, default 0).
]]
function totem.assertTableEq(ta, tb, tolerance)
  return assert(totem.areTablesEq(ta, tb, tolerance))
end


--[[ Asserts (deep) table inequality.

Tables are compared recursively. If any of their key or associated value type
doesn't match, they are considered unequal. Otherwise, numbers are compared with
the given tolerance. Torch tensors and storages are compared with the given
tolerance on their elementwise difference. Other types are compared for strict
equality with the regular Lua == operator. Fails if one of its two first
arguments is not a table.

Arguments:

* `ta` (table)
* `tb` (table)
* `tolerance` (optional number, default 0).
]]
function totem.assertTableNe(ta, tb, tolerance)
  return assert(totem.areTablesEq(ta, tb, tolerance, true))
end
