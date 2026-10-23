local rows = {}
for i, key in ipairs(KEYS) do
    local value = redis.call('HMGET', key, 'v', 'ver')
    if value[1] then table.insert(rows, {i, value[1], value[2]}) end
end
return rows
