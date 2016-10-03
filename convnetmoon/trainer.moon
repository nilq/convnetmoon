----------------------------------
-- Manages Trainers:
--    1. Vanilla SGD
--    2. Momentum
--    3. Adagrad
--    4. Adadelta
--    5. Windowgrad
----------------------------------
export class Trainer
  new: (@net, @opt) =>
    @learning_rate = util.get_opt opt, "learning_rate", 0.01
    @l1_decay = util.get_opt opt, "l1_decay", 0
    @l2_decay = util.get_opt opt, "l2_decay", 0
    @batch_size = util.get_opt opt, "batch_size", 1
    @method = util.get_opt opt, "method", "sgd"

    @momentum = util.get_opt opt, "momentum", 0.9

    -- used in 'adadelta'
    @ro  = util.get_opt opt, "ro", 0.95
    @eps = util.get_opt opt, "eps", 1e-6

    @k = 0 -- iterations
    @gsum = {} -- last iteration's gradients (used for momentum)
    @xsum = {} -- used in 'adadelta'

    @win = util.Window!

  train: (x, y) =>
    @k += 1
    start = os.time!
    -- @net\forward x, true -- training
    fwd_time = os.difftime os.time!, start

    if "table" != type y
      @win\add @net\get_prediction == y

    start = os.time!
    cost_loss = @net\backward y
