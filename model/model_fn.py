"""Define the model."""

import tensorflow as tf

from model.triplet_loss import batch_all_triplet_loss
from model.triplet_loss import batch_hard_triplet_loss
from model import nets_factory

slim = tf.contrib.slim


def _configure_learning_rate(num_samples_per_epoch, global_step, cf):
    """Configures the learning rate.
  
    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.
  
    Returns:
      A `Tensor` representing the learning rate.
  
    Raises:
      ValueError: if
    """
    # Note: when num_clones is > 1, this will actually have each clone to go
    # over each epoch cf.num_epochs_per_decay times. This is different
    # behavior from sync replicas and is expected to produce different results.
    decay_steps = int(num_samples_per_epoch * cf.num_epochs_per_decay /
                      cf.batch_size)

    if cf.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(cf.learning_rate,
                                          global_step,
                                          decay_steps,
                                          cf.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif cf.learning_rate_decay_type == 'fixed':
        return tf.constant(cf.learning_rate, name='fixed_learning_rate')
    elif cf.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(cf.learning_rate,
                                         global_step,
                                         decay_steps,
                                         cf.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized' %
                         cf.learning_rate_decay_type)


def _configure_optimizer(learning_rate, cf):
    """Configures the optimizer used for training.
  
    Args:
      learning_rate: A scalar or `Tensor` learning rate.
  
    Returns:
      An instance of an optimizer.
  
    Raises:
      ValueError: if cf.optimizer is not recognized.
    """
    if cf.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=cf.adadelta_rho,
            epsilon=cf.opt_epsilon)
    elif cf.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=cf.adagrad_initial_accumulator_value)
    elif cf.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=cf.adam_beta1,
            beta2=cf.adam_beta2,
            epsilon=cf.opt_epsilon)
    elif cf.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=cf.ftrl_learning_rate_power,
            initial_accumulator_value=cf.ftrl_initial_accumulator_value,
            l1_regularization_strength=cf.ftrl_l1,
            l2_regularization_strength=cf.ftrl_l2)
    elif cf.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=cf.momentum,
            name='Momentum')
    elif cf.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=cf.rmsprop_decay,
            momentum=cf.rmsprop_momentum,
            epsilon=cf.opt_epsilon)
    elif cf.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized' % cf.optimizer)
    return optimizer


def build_model_default(is_training, images, params):
    """Compute outputs of the model (embeddings for triplet loss).

    Args:
        is_training: (bool) whether we are training or not
        images: (dict) contains the inputs of the graph (features)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    out = images
    # Define the number of channels of each convolution
    # For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
    num_channels = params.num_channels
    bn_momentum = params.bn_momentum
    channels = [num_channels, num_channels * 2]
    for i, c in enumerate(channels):
        with tf.variable_scope('block_{}'.format(i + 1)):
            out = tf.layers.conv2d(out, c, 3, padding='same')
            if params.use_batch_norm:
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            out = tf.nn.relu(out)
            out = tf.layers.max_pooling2d(out, 2, 2)

    assert out.shape[1:] == [7, 7, num_channels * 2]

    out = tf.reshape(out, [-1, 7 * 7 * num_channels * 2])
    with tf.variable_scope('fc_1'):
        out = tf.layers.dense(out, params.embedding_size)

    return out


def build_slim_model(is_training, images, params):
    """Compute outputs of the model (embeddings for triplet loss).

    Args:
        is_training: (bool) whether we are training or not
        images: (dict) contains the inputs of the graph (features)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    wd = 0.
    if hasattr(params, "weight_decay"):
        wd = params.weight_decay
    model_f = nets_factory.get_network_fn(params.model_name, int(params.embedding_size), wd,
                                          is_training=is_training)
    out, end_points = model_f(images)

    return out, end_points


def model_fn(features, labels, mode, params):
    """Model function for tf.estimator

    Args:
        features: input batch of images
        labels: labels of the images
        mode: can be one of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}
        params: contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        model_spec: tf.estimator.EstimatorSpec object
    """
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    images = features
    if len(images.get_shape()) == 2:
        images = tf.reshape(images, [-1, params.image_size, params.image_size, 1])
        assert images.shape[1:] == [params.image_size, params.image_size, 1], "{}".format(images.shape)

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model'):
        # Compute the embeddings with the model
        if params.model_name == "base_model":
            embeddings = build_model_default(is_training, images, params)
        else:
            embeddings, _ = build_slim_model(is_training, images, params)
    embedding_mean_norm = tf.reduce_mean(tf.norm(embeddings, axis=1))

    tf.summary.scalar("embedding_mean_norm", embedding_mean_norm)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'embeddings': embeddings}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    labels = tf.cast(labels, tf.int64)

    # Define triplet loss
    if params.triplet_strategy == "batch_all":
        loss, fraction = batch_all_triplet_loss(labels, embeddings, margin=params.margin,
                                                squared=params.squared)
    elif params.triplet_strategy == "batch_hard":
        loss = batch_hard_triplet_loss(labels, embeddings, margin=params.margin,
                                       squared=params.squared)
    else:
        raise ValueError("Triplet strategy not recognized: {}".format(params.triplet_strategy))

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    # TODO: some other metrics like rank-1 accuracy?
    with tf.variable_scope("metrics"):
        eval_metric_ops = {"embedding_mean_norm": tf.metrics.mean(embedding_mean_norm)}

        if params.triplet_strategy == "batch_all":
            eval_metric_ops['fraction_positive_triplets'] = tf.metrics.mean(fraction)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    if params.triplet_strategy == "batch_all":
        tf.summary.scalar('fraction_positive_triplets', fraction)

    tf.summary.image('train_image', images, max_outputs=10)

    # Define training step that minimizes the loss with the Adam optimizer
    optimizer = tf.train.AdamOptimizer(params.learning_rate)
    global_step = tf.train.get_global_step()
    if params.use_batch_norm:
        # Add a dependency to update the moving mean and variance for batch normalization
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(loss, global_step=global_step)
    else:
        train_op = optimizer.minimize(loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def _get_variables_to_train(cf):
    """Returns a list of variables to train.

    Returns:
      A list of variables to train by the optimizer.
    """
    if cf.trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in cf.trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def train_op_fun(total_loss, global_step, num_examples, cf):
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
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    if cf.moving_average_decay:
        moving_average_variables = slim.get_model_variables()
        variable_averages = tf.train.ExponentialMovingAverage(
            cf.moving_average_decay, global_step)
        update_ops.append(variable_averages.apply(moving_average_variables))

    lr = _configure_learning_rate(num_examples, global_step, cf)
    tf.summary.scalar('learning_rate', lr)
    opt = _configure_optimizer(lr, cf)
    variables_to_train = _get_variables_to_train(cf)
    grads = opt.compute_gradients(total_loss, variables_to_train)
    grad_updates = opt.apply_gradients(grads, global_step=global_step)
    update_ops.append(grad_updates)
    update_op = tf.group(*update_ops)
    with tf.control_dependencies([update_op]):
        train_op = tf.identity(total_loss, name='train_op')

    return train_op


def build_model(features, labels, cf, attrs=None, is_training=True, use_attr_net=False, num_hidden_attr_net=1,
                num_examples=None, global_step=None):
    images = features

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model'):
        # Compute the embeddings with the model
        embeddings, end_points = build_slim_model(is_training, images, cf)
        if cf.l2norm:
            embeddings = tf.nn.l2_normalize(embeddings, axis=1)
        if attrs is not None and use_attr_net:
            hidden_step = int((cf.attr_dim - cf.embedding_size) / (num_hidden_attr_net + 1))
            for i in range(num_hidden_attr_net):
                print(cf.attr_dim - (hidden_step * (i + 1)))
                attr_net = tf.layers.dense(attrs, cf.attr_dim - (hidden_step * (i + 1)), tf.nn.relu,
                                           trainable=is_training)
                attr_net = tf.layers.dropout(attr_net, training=is_training)
            attrs = tf.layers.dense(attr_net, cf.embedding_size, tf.nn.relu, trainable=is_training)
        if cf.l2norm:
            attrs = tf.nn.l2_normalize(attrs, axis=1)
    embedding_mean_norm = tf.reduce_mean(tf.norm(embeddings, axis=1))

    tf.summary.scalar("embedding_mean_norm", embedding_mean_norm)

    if not is_training:
        return embeddings

    labels = tf.cast(labels, tf.int64)

    # Define triplet loss
    if cf.triplet_strategy == "batch_all":
        loss, fraction = batch_all_triplet_loss(labels, embeddings, margin=cf.margin, attrs=attrs,
                                                attr_weight=cf.attr_loss_weight, squared=cf.squared)
    elif cf.triplet_strategy == "batch_hard":
        loss = batch_hard_triplet_loss(labels, embeddings, margin=cf.margin, attrs=attrs,
                                       attr_weight=cf.attr_loss_weight,
                                       squared=cf.squared)

    else:
        raise ValueError("Triplet strategy not recognized: {}".format(cf.triplet_strategy))

    vars = tf.trainable_variables()
    loss += tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * cf.weight_decay

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    # TODO: some other metrics like rank-1 accuracy?
    with tf.variable_scope("metrics"):
        eval_metric_ops = {"embedding_mean_norm": tf.metrics.mean(embedding_mean_norm)}

        if cf.triplet_strategy == "batch_all":
            eval_metric_ops['fraction_positive_triplets'] = tf.metrics.mean(fraction)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    if cf.triplet_strategy == "batch_all":
        tf.summary.scalar('fraction_positive_triplets', fraction)

    tf.summary.image('train_image', images, max_outputs=1)

    train_op = train_op_fun(loss, global_step, num_examples, cf)

    return loss, end_points, train_op
