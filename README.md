# ConvNetMoon

ConvNetMoon is a MoonScript implementation of neural networks, based on ConvNetJS. It currently supports:

- Common **Neural Network modules** (fully connected layers, non-linearities)
- Classification (SVM/softmax) and Regressoin (L2) **cost functions**
- Ability to specify and train **Convolutional Networks**

---

**[STATUS]** Some things still need proper testing.

---
##Example

```moonscript
require "lib"

layer_defs = {
  {type: "input", out_sx: 1, out_sz: 2},
  {type: "fc", num_neurons: 20, activation: "relu"},
  {type: "softmax", num_classes: 10},
}

net = Net!
net\make_layers layer_defs

x = Vol {0.3, -0.5}

prob = net\forward x

print "probability that 'x' is class '1': " .. prob.w[1]

trainer = Trainer net, {learning_rate: 0.01, l2_decay: 0.001}

trainer\train x, 1

print "training ..."

prob = net\forward x

print "probability that 'x' is class '1': " .. prob.w[1]
```

output:

```
probability that 'x' is class '1': 0.10565505576001
training ...
probability that 'x' is class '1': 1
```

