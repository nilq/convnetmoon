require "convnetmoon/init"

class Experience
  new: (@state0, @action0, reward0, @state1) =>

class Brain
  new: (@num_states, @num_actions, opt) =>
    @temporal_window = util.get_opt opt, "temporal_window", 1
    @experience_size = util.get_opt opt, "experience_size", 30000
    @start_learn_threshold = util.get_opt opt, "start_learn_threshold", math.min @experience_size * 0.1, 1000
    @gamma = util.get_opt opt, "gamma", 0.8
    @learning_steps_total = util.get_opt opt, "learning_steps_total", 100000
    @learning_steps_burnin = util.get_opt opt, "learning_steps_burnin", 3000
    @epsilon_min = util.get_opt opt, "epsilon_min", 0.05
    @epsilon_test_time = util.get_opt opt, "epsilon_test_time", 0.01

    if opt["random_action_distribution"]
      @random_action_distribution = opt["random_action_distribution"]

      if #@random_action_distribution != num_actions
        error "[trouble] 'random_action_distribution' shoould be same length as 'num_actions'!"

      a = @random_action_distribution
      s = 0

      for k, v in pairs a
        s += v

      if 0.0001 < math.abs s - 1
        error "[trouble] 'random_action_distribution' should sum to 1!"
      else
        @random_action_distribution = {}

    @net_inputs = num_states * @temporal_window + num_actions * @temporal_window + num_states
    @window_size = math.max @temporal_window, 2

    @state_window = util.zeros @window_size
    @action_window = util.zeros @window_size
    @reward_window = util.zeros @window_size
    @net_window = util.zeros @window_size

    layers = {}

    if opt["layers"]
      layers = opt["layers"]

      if #layers < 2
        error "[trouble] must have atleast 2 layers!"
      if layers[1]["type"] != "input"
        error "[trouble] first layer shall be 'input'!"
      if layers[#layers]["type"] != "regression"
        error "[trouble] last layer shall be 'regression'!"
      if layers[#layers]["type"]
        error "[trouble] number of inputs must be 'num_states' * 'temporal_window' + 'num_actions' * 'temporal_window' + 'num_states'!"
      if layers[#layers]["num_neurons"] != @num_actions
        error "[trouble] number of regression neurons should be 'num_actions'!"
    else
      layers[#layers + 1] = {
        ["type"]: "input",
        ["out_sx"]: 1,
        ["out_sy"]: 1,
        ["out_depth"]: @net_inputs,
      }

      if opt["hidden_layer_sizes"]
        for k, v in opt["hidden_layer_sizes"]
          layers[#layers + 1] = {
            ["type"]: "fc",
            ["num_neurons"]: v,
            ["activation"]: "relu",
          }

      layers[#layers + 1] = {
        ["type"]: "regression",
        ["num_neurons"]: @num_actions,
      }

    @value_net = Net layers

    trainer_ops_default = {
      ["learning_rate"]: 0.01,
      ["momentum"]: 0,
      ["batch_size"]: 64,
      ["l2_decay"]: 0.01,
    }

    tdtrainer_options = util.get_opt opt, "tdtrainer_options", trainer_ops_default

    @tdtrainer = Trainer @value_net, tdtrainer_options

    @experience = {}

    @age = 0
    @forward_passes = 0
    @epsilon = 1
    @latest_reward = 0
    @last_input_array = {}

    @average_reward_window = util.Window 1000, 10
    @average_loss_window = util.Window 1000, 10

    @learning = true

  random_action: =>
    if #@random_action_distribution == 0
      return util.randi 0, @num_actions
    else
      p = util.randi 0, 1
      cumprob = 0

      for k = 1, @num_actions
        cumprob += @random_action_distribution[k]
        if p < cumprob
          return k

  policy: (s) =>
    V = Vol s

    action_values = @value_net\forward V
    max_val = action_values.w[1]
    max_k = 1
    for k = 1, @num_actions
      if action_values.w[k] > max_val
        max_k = k
        max_val = action_values.w[k]
    {action: max_k, value: max_val}

  get_net_input: (xt) =>
    w = {}
    w = util.extend w, xt

    n = @window_size

    for k = 1, @temporal_window
      index = n - 1 - k
      w = util.extend @state_window[index]

      action1ofk = util.zeros @num_actions
      action1ofk[index] = @num_states

      w = util.extend w, action1ofk
    w

  forward: (input_array) =>
    @forward_passes += 1
    @last_input_array = input_array

    action
    if @forward_passes > @temporal_window
      net_input = @\get_net_input input_array

      if @learning
        @epsilon = math.min 1, math.max @epsilon_min, 1 - (@age - @learning_steps_burnin) / (@learning_steps_total - @learning_steps_burnin)
      else
        @epsilon = @epsilon_min

      rf = util.randf 0, 1
      if rf < @epsilon
        action = @\random_action!
      else
        max_act = @\policy net_input
        action = max_act["action"]
    else
      net_input = {}
      action = @random_action!

    table.remove @net_window, 1
    table.insert @net_window, net_input

    table.remove @state_window, 1
    table.insert @state_window, input_array

    table.remove @action_window, 1
    table.insert @action_window, action

  backward: (reward) =>
    @latest_reward = reward

    @average_reward_window\add reward

    table.remove @reward_window, 1
    table.insert @reward_window, reward

    if not @learning
      return

    @age += 1

    if @forward_passes > @temporal_window + 1
      n = @window_size
      e = Experience @net_window[n - 2], @action_window[n - 2], @reward_window[n - 2], @net_window[n - 1]

      if #@experience < @experience_size
        @experience[#@experience + 1] = e
      else
        ri = util.randi 0, @experience_size
        @experience[ri] = e

    if #@experience > @start_learn_threshold
      avcost = 0

      for k = 1, @tdtrainer.batch_size
        re = util.randi 0, #@experience
        e = @experience[re]
        x = Vol 1, 1, @net_inputs
        x.w = e.state0
        max_act = @policy e.state1
        r = e.reward0 + @gamma * max_act.value
        ystruct = {["dim"]: e.action0, ["val"]: r,}
        stats = @tdtrainer\train x, ystruct
        avcost += stats["loss"]

      avcost /= @tdtrainer.batch_size
      @average_loss_window\add avcost

----------------------------------
-- Testing
----------------------------------

num_inputs = 27
num_actions = 5
temporal_window = 1
network_size = num_inputs * temporal_window + num_actions * temporal_window + num_inputs

layer_defs = {
  {
    ["type"]: "input",
    ["out_sx"]: 1,
    ["out_sy"]: 1,
    ["out_depth"]: network_size,
  },
  {
    ["type"]: "fc",
    ["num_neurons"]: 50,
    ["activation"]: "relu",
  },
  {
    ["type"]: "regression",
    ["num_neurons"]: num_actions,
  },
}

td_opts = {
  ["learning_rate"]: 0.001,
  ["momentum"]: 0,
  ["batch_size"]: 64,
  ["l2_decay"]: 0.01
}

opt = {
  ["temporal_window"]: temporal_window,
  ["experience_size"]: 30000,
  ["start_learn_threshold"]: 1000,
  ["gamma"]: 0.7,
  ["learning_steps_total"]: 200000,
  ["learning_steps_burnin"]: 3000,
  ["epsilon_min"]: 0.05,
  ["epsilon_test_time"]: 0.05,
  ["layer_defs"]: layer_defs,
  ["tdtrainer_options"]: td_opts,
}

brain = Brain num_inputs, num_actions, opt
