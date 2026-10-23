local live = {}
for i = 2, #KEYS do
    if redis.call('HEXISTS', KEYS[i], ARGV[1]) == 1 then
        table.insert(live, ARGV[i + 1])
    else
        redis.call('SREM', KEYS[1], ARGV[i + 1])
    end
end
if #live > 0 then redis.call('PEXPIRE', KEYS[1], ARGV[2]) end
return live
