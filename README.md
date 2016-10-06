# ConvNetMoon

ConvNetMoon is a MoonScript implementation of neural networks, based on ConvNetJS. It currently supports:

- Common **Neural Network modules** (fully connected layers, non-linearities)
- Classification (SVM/softmax) and Regressoin (L2) **cost functions**
- Ability to specify and train **Convolutional Networks**

---

**[STATUS]** Finishing up different types of layers.

---

## Example

```moon
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
    ["activation"]: "relu",
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
```

---

The primary reason I'm making this thing is to express my eternal love to the MoonScript language. It is for educational purposes and for everyone to use. Being made in MoonScript will make ConvNetMoon(when compiled) able to work in any Lua environment(basically everywhere) ... which is nice.

---
