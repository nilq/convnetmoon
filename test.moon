math.randomseed os.time!

require "convnetmoon/init"

print "\r"

----------------------------------
-- UTILS
----------------------------------
print "[util] randf [0; 15]", util.randf 0, 15
print "[util] randi [0; 15]", util.randi 0, 15
print "[util] randn [0; 15]", util.randn 0, 15
-- lists
zeros = util.zeros 100

assert 100 == #zeros, "[util] zeros not working!"
assert (util.contains zeros, 0), "[util] zeros do not contain zero!"
assert 1 == #util.unique zeros, "[util] unique list fucking up"

print "\n[util] utils passing!\r"

foo = {100, 1337, 420}

print "[util][maxmin] {100, 1337, 420}"

minmax = util.maxmin foo

assert minmax.max_v == 1337
assert minmax.max_i == 2

assert minmax.min_v == 100
assert minmax.min_i == 1

print "[util][maxmin] success!"
print "\r"

print "[util][randperm] (#8) coming up!"

permutation = util.randperm 8

for i, v in pairs permutation
  print i, v

print "[util][randperm] seems random to me\n"

print "[util] weighted sample ({1, 2, 3, 4}, {0.2, 0.5, 0.9, 0.3})"

sample = util.weighted_sample {1, 2, 3, 4}, {0.1, 5, 9, 3}

print "[util][weighted_sample]", sample

print "\n[util][Window]"
win = util.Window 4, 1
win\add 10
win\add 10
win\add 12
win\add 10

print "[util][Window] average:", win\get_average!

print "\n[util] all units passing; moving on.\n"
