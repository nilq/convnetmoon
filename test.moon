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

----------------------------------
-- VOL
----------------------------------

vol = Vol 5, 2, 6

print "[Vol] [5, 2, 6] as JSON", vol\to_JSON!

for k, v in pairs vol\to_JSON!
  print "[JSON]", k, v

print "\n[actual json]\n\n" .. util.save_json vol\to_JSON!

----------------------------------
-- NET
----------------------------------

print "\n [net]"

defs = {
  {
    ["type"]: "input",
    ["out_sx"]: 2,
    ["out_sy"]: 1,
    ["out_depth"]: 1
  },
  {
    ["type"]: "fc",
    ["activation"]: "relu",
  }
  {
    ["type"]: "softmax",
  },
}

boi = Net defs

print "\n [net] Network as JSON ...\n"

print util.save_json boi\to_JSON!

----------------------------------
-- TRAINER
----------------------------------
print "[trainer] your boi trainer here"
t = Trainer boi, {["method"]: "adadelta", ["batch_size"]: 2, ["l2_decay"]: 0.0001}

x = Vol {0.3, 0.5}

t\train x, 0
