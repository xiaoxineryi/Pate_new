from datetime import datetime
import math
import time

import numpy as np
import tensorflow.compat.v1 as tf

FLAGS = tf.flags.FLAGS

tf.app.flags.DEFINE_integer('dropout_seed', 123, """seed for dropout.""")
tf.app.flags.DEFINE_integer('batch_size', 128, """Nb of images in a batch.""")
tf.app.flags.DEFINE_integer('epochs_per_decay', 350, """Nb epochs per decay""")
tf.app.flags.DEFINE_integer('learning_rate', 5, """100 * learning rate""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """see TF doc""")


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

tf.compat.v1.disable_eager_execution()
def input_placeholder():
    """
    这个函数声明一个占位符，占位符可以在图中运行，但其数据是动态输入的

    """
    image_size = 28
    num_channels = 1

    train_node_shape = (FLAGS.batch_size,image_size,image_size,num_channels)
    return tf.placeholder(tf.float32,shape=train_node_shape)

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  # 生成一个随机矩阵 作为核函数
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def inference(images, dropout=False):
    """
        创建对于Mnist的CNN神经网络
    """

    first_conv_shape = [5, 5, 1, 64]

    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=first_conv_shape,
                                             stddev=1e-4,
                                             wd=0.0)
        # nn.conv2d: 输入的input图像为[batch,in_height,out_height,in_channels],输入的核函数为[filter_height,filter_weight,in_channels,
        # out_channels] strides是步长，一般是[1,strides,strides,1]
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        if dropout:
            conv1 = tf.nn.dropout(conv1, 0.3, seed=FLAGS.dropout_seed)

    # pool1
    pool1 = tf.nn.max_pool(conv1,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool1')

    # norm1
    norm1 = tf.nn.lrn(pool1,
                      4,
                      bias=1.0,
                      alpha=0.001 / 9.0,
                      beta=0.75,
                      name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 64, 128],
                                             stddev=1e-4,
                                             wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        if dropout:
            conv2 = tf.nn.dropout(conv2, 0.3, seed=FLAGS.dropout_seed)

    # norm2
    norm2 = tf.nn.lrn(conv2,
                      4,
                      bias=1.0,
                      alpha=0.001 / 9.0,
                      beta=0.75,
                      name='norm2')

    # pool2
    pool2 = tf.nn.max_pool(norm2,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool2')

    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1]
        weights = _variable_with_weight_decay('weights',
                                              shape=[dim, 384],
                                              stddev=0.04,
                                              wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        if dropout:
            local3 = tf.nn.dropout(local3, 0.5, seed=FLAGS.dropout_seed)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights',
                                              shape=[384, 192],
                                              stddev=0.04,
                                              wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        if dropout:
            local4 = tf.nn.dropout(local4, 0.5, seed=FLAGS.dropout_seed)

    # compute logits
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights',
                                              [192, FLAGS.nb_labels],
                                              stddev=1 / 192.0,
                                              wd=0.0)
        biases = _variable_on_cpu('biases',
                                  [FLAGS.nb_labels],
                                  tf.constant_initializer(0.0))
        logits = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

    return logits


def loss_fun(logits, labels):
    """
    添加L2损失函数
    """
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')

    # Calculate the average cross entropy loss across the batch.
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

    # Add to TF collection for losses

    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def moving_av(total_loss):
  """
  Generates moving average for all losses

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  return loss_averages_op

def train_op_fun(total_loss, global_step):
  """Train model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  nb_ex_per_train_epoch = int(60000 / FLAGS.nb_teachers)

  num_batches_per_epoch = nb_ex_per_train_epoch / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * FLAGS.epochs_per_decay)

  initial_learning_rate = float(FLAGS.learning_rate) / 100.0

  # Decay the learning rate exponentially based on the number of steps.
  # 产生变化的学习率
  lr = tf.train.exponential_decay(initial_learning_rate,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  # 使用滑动平均来获得平均的损失值
  loss_averages_op = moving_av(total_loss)

  # Compute gradients.
  # 依赖器来表示必须先计算loss_averages_op 才可以继续下面的创建optimizer和计算梯度 防止并发运行造成错误？
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


def batch_indices(batch_nb, data_length, batch_size):
    """
    计算开始和结束的标签位置
    """
    start = int(batch_nb * batch_size)
    end = int((batch_nb + 1) * batch_size)

    # When there are not enough inputs left, we reuse some to complete the batch
    if end > data_length:
        shift = end - data_length
        start -= shift
        end -= shift

    return start, end


def train(images,labels,ckpt_path,drop_out = False):
    # 先判定对应的数据类型是否匹配
    assert len(images) == len(labels)
    assert images.dtype == np.float32
    assert labels.dtype == np.int32

    # 设置默认的图
    with tf.Graph().as_default():
        global_step = tf.Variable(0,trainable=False)
        # 生成占位符
        train_data_node = input_placeholder()
        train_labels_shape = (FLAGS.batch_size,)
        train_labels_node =tf.placeholder(tf.int32,train_labels_shape)

        print("Done Initializing Training Placeholders")

        # 构造图
        logits = inference(train_data_node, dropout=drop_out)

        # 基于图计算损失函数
        loss = loss_fun(logits, train_labels_node)
        # 添加优化器
        train_op = train_op_fun(loss, global_step)

        # 创建一个保存器
        saver =tf.train.Saver(tf.global_variables())

        print("Graph constructed and saver created")

        # 初始化
        init = tf.global_variables_initializer()

        #创建session并且初始化
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        print("Session ready, beginning training loop")

        data_length = len(images)
        nb_batches = math.ceil(data_length/FLAGS.batch_size)

        for step in range(FLAGS.max_steps):
            start_time = time.time();

            batch_nb = step % nb_batches

            start,end = batch_indices(batch_nb,data_length,FLAGS.batch_size)
            # 填入数据
            feed_dict = {train_data_node:images[start:end],
                         train_labels_node:labels[start:end]}
            _,loss_value =sess.run([train_op,loss],feed_dict=feed_dict)
            # 计算总用时
            duration = time.time() - start_time
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            if step% 100 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))
            if (step + 1) == FLAGS.max_steps:
                saver.save(sess, ckpt_path, global_step=step)
    return True

def softmax_preds(images, ckpt_path,return_logits = False):
    """
    根据保存的教师模型信息来预测，检查准确率
    """
    data_length = len(images)
    nb_batches = math.ceil(len(images)/FLAGS.batch_size)

    train_data_node = input_placeholder()
    logits = inference(train_data_node)

    if return_logits:
        output = logits
    else:
        output = tf.nn.softmax(logits)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    preds = np.zeros((data_length, FLAGS.nb_labels), dtype=np.float32)

    # 建立会话
    with tf.Session() as sess:
        # 恢复模型
        saver.restore(sess,ckpt_path)
        for batch_nb in range(0,int(nb_batches+1)):
            start,end =batch_indices(batch_nb,data_length,FLAGS.batch_size)
            feed_dict = {train_data_node:images[start:end]}
            preds[start:end,:] =sess.run([output],feed_dict=feed_dict)[0]

    #恢复原始以便多次使用
    tf.reset_default_graph()

    return preds