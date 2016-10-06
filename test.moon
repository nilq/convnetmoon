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
    ["num_classes"]: 2,
    ["num_neurons"]: 30,
    ["activation"]: "relu",
  }
  {
    ["type"]: "softmax",
    ["num_classes"]: 2,
    ["num_neurons"]: 30,
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
t\train x, 1

print "[TRAINING] SUCCESSSSS\n"

print "network:"
print "loss:", t["cost_loss"]
print #(boi\forward x).w
for i = 1, #(boi\forward x).w
  print (boi\forward x).w[i]
print "\ntraining ..."
stuff_and_things = t\train x, 1
print "training done ...\n"
print "network again:"
print #(boi\forward x).w
for i = 1, #(boi\forward x).w
  print (boi\forward x).w[i]

print "loss!!:", stuff_and_things["cost_loss"]

----------------------------------
-- stuff; the real test
----------------------------------
print "[real test]\n"

-- design clever network architecture ...
layer_defs = {
  {
    ["type"]: "input",
    ["out_sx"]: 1,
    ["out_sy"]: 1,
    ["out_depth"]: 2,
  },
  {
    ["type"]: "fc",
    ["num_neurons"]: 20,
    ["activation"]: "sigmoid",
  },
  {
    ["type"]: "softmax",
    ["num_classes"]: 10,
  },
}

-- create network from definitions ...
net = Net layer_defs
-- define important training data ...
x = Vol {0.3, -0.5,}
-- get (untrained) predictions from network ...
prob = net\forward x
-- confirm badness of network ...
print "probability that 'x' is class 1:", prob.w[1]
-- create trainer from trainer definitions ...
trainer = Trainer net, {["learning_rate"]: 0.01, ["l2_decay"]: 0.001,}
-- train the network from data!
trainer\train x, 1
-- confirm less badness of network ...
prob2 = net\forward x
print "probability that 'x' is class 1:", prob2.w[1]
