import numpy as np
import tensorflow as tf
from aer import read_naacl_alignments, AERSufficientStatistics
from utils import iterate_minibatches, prepare_data

# for TF 1.1
import tensorflow
try:
  from tensorflow.contrib.keras.initializers import glorot_uniform
except:  # for TF 1.0
  from tensorflow.contrib.layers import xavier_initializer as glorot_uniform


class NeuralIBM1Model:
  """Our Neural IBM1 model."""
  
  def __init__(self, batch_size=8, 
               x_vocabulary=None, y_vocabulary=None, 
               emb_dim=32, mlp_dim=64,
               session=None, context=None):

    self.batch_size = batch_size
    self.emb_dim = emb_dim
    self.mlp_dim = mlp_dim
    self.context = context
    
    self.x_vocabulary = x_vocabulary
    self.y_vocabulary = y_vocabulary
    self.x_vocabulary_size = len(x_vocabulary)
    self.y_vocabulary_size = len(y_vocabulary)

    self._create_placeholders()
    self._create_weights()
    self._build_model()
    
    self.saver = tf.train.Saver()
    self.session = session
    
  def _create_placeholders(self):
    """We define placeholders to feed the data to TensorFlow."""
    # "None" means the batches may have a variable maximum length.
    self.x = tf.placeholder(tf.int64, shape=[None, None])
    self.y = tf.placeholder(tf.int64, shape=[None, None])
    self.y_prev = tf.placeholder(tf.int64, shape=[None, None])
    
  def _create_weights(self):
    """Create weights for the model."""
    with tf.variable_scope("MLP") as scope:
      mult = 1
      if self.context=="concat":
        mult = 2

      self.mlp_W_ = tf.get_variable(
        name="W_", initializer=glorot_uniform(),
        shape=[self.emb_dim*mult, self.mlp_dim])

      self.mlp_b_ = tf.get_variable(
        name="b_", initializer=tf.zeros_initializer(),
        shape=[self.mlp_dim])
      
      self.mlp_W = tf.get_variable(
        name="W", initializer=glorot_uniform(),
        shape=[self.mlp_dim, self.y_vocabulary_size])

      self.mlp_b = tf.get_variable(
        name="b", initializer=tf.zeros_initializer(),
        shape=[self.y_vocabulary_size])

      ### Weights for T3 ###
      # first layer for t
      self.mlp_Wt_ = tf.get_variable(
        name="Wt_", initializer=glorot_uniform(),
        shape=[self.emb_dim, self.mlp_dim])

      self.mlp_bt_ = tf.get_variable(
        name="bt_", initializer=tf.zeros_initializer(),
        shape=[self.mlp_dim])

      # first layer for i and s
      self.mlp_Wis_ = tf.get_variable(
        name="Wis_", initializer=glorot_uniform(),
        shape=[self.emb_dim, self.mlp_dim])

      self.mlp_bis_ = tf.get_variable(
        name="bis_", initializer=tf.zeros_initializer(),
        shape=[self.mlp_dim])

      # layer for translation P(F|E)
      self.mlp_W_t = tf.get_variable(
        name="W_t", initializer=glorot_uniform(),
        shape=[self.mlp_dim, self.y_vocabulary_size])

      self.mlp_b_t = tf.get_variable(
        name="b_t", initializer=tf.zeros_initializer(),
        shape=[self.y_vocabulary_size])

      # layer for insertion P(F|Fprev)
      self.mlp_W_i = tf.get_variable(
        name="W_i", initializer=glorot_uniform(),
        shape=[self.mlp_dim, self.y_vocabulary_size])

      self.mlp_b_i = tf.get_variable(
        name="b_i", initializer=tf.zeros_initializer(),
        shape=[self.y_vocabulary_size])

      # layer for collocation P(C|Fprev)
      self.mlp_W_s = tf.get_variable(
        name="W_s", initializer=glorot_uniform(),
        shape=[self.mlp_dim, 1])

      self.mlp_b_s = tf.get_variable(
        name="b_s", initializer=tf.zeros_initializer(),
        shape=[1])

  def save(self, session, path="model.ckpt"):
    """Saves the model."""
    return self.saver.save(session, path)
    
  def _build_model(self):
    """Builds the computational graph for our model."""
    
    # 1. Let's create a (source) word embeddings matrix. 
    # These are trainable parameters, so we use tf.get_variable.
    # Shape: [Vx, emb_dim] where Vx is the source vocabulary size
    x_embeddings = tf.get_variable(
      name="x_embeddings", initializer=tf.random_uniform_initializer(),
      shape=[self.x_vocabulary_size, self.emb_dim])

    # Now we start defining our graph.
    
    # This looks up the embedding vector for each word given the word IDs in self.x.
    # Shape: [B, M, emb_dim] where B is batch size, M is (longest) source sentence length.
    x_embedded = tf.nn.embedding_lookup(x_embeddings, self.x)

    # In case for additional french context
    y_embeddings = tf.get_variable(
      name="y_embeddings", initializer=tf.random_uniform_initializer(),
      shape=[self.y_vocabulary_size, self.emb_dim])

    y_prev_embedded = tf.nn.embedding_lookup(y_embeddings, self.y_prev)

    # 2. Now we define the generative model P(Y | X=x)
    
    # first we need to know some sizes from the current input data
    batch_size = tf.shape(self.x)[0]
    longest_x = tf.shape(self.x)[1]  # longest M
    longest_y = tf.shape(self.y)[1]  # longest N
    
    # It's also useful to have masks that indicate what
    # values of our batch we should ignore.
    # Masks have the same shape as our inputs, and contain
    # 1.0 where there is a value, and 0.0 where there is padding.
    x_mask = tf.cast(tf.sign(self.x), tf.float32)  # Shape: [B, M]
    y_mask = tf.cast(tf.sign(self.y), tf.float32)  # Shape: [B, N]
    x_len = tf.reduce_sum(tf.sign(self.x), axis=1)  # Shape: [B]
    y_len = tf.reduce_sum(tf.sign(self.y), axis=1)  # Shape: [B]
    
    # 2.a Build an alignment model P(A | X, M, N)
    
    # This just gives you 1/length_x (already including NULL) per sample.
    # i.e. the lengths are the same for each word y_1 .. y_N.
    lengths = tf.expand_dims(x_len, -1)  # Shape: [B, 1]
    pa_x = tf.div(x_mask, tf.cast(lengths, tf.float32))  # Shape: [B, M]
    
    # We now have a matrix with 1/M values.
    # For a batch of 2 sentences, with lengths 2 and 3:
    #
    #  pa_x = [[1/2 1/2   0]
    #          [1/3 1/3 1/3]]
    #
    # But later we will need it N times. So we repeat (=tile) this
    # matrix N times, and for that we create a new dimension
    # in between the current ones (dimension 1).
    pa_x = tf.expand_dims(pa_x, 1)  # Shape: [B, 1, M]
    
    #  pa_x = [[[1/2 1/2   0]]
    #          [[1/3 1/3 1/3]]]
    # Note the extra brackets.

    # Now we perform the tiling:
    pa_x = tf.tile(pa_x, [1, longest_y, 1])  # [B, N, M]    

    # Result:
    #  pa_x = [[[1/2 1/2   0]
    #           [1/2 1/2   0]]
    #           [[1/3 1/3 1/3]
    #           [1/3 1/3 1/3]]]

    # 2.b P(Y | X, A) = P(Y | X_A)
        
    # First we make the input to the MLP 2-D.
    # Every output row will be of size Vy, and after a softmax
    # will sum to 1.0.
    mlp_input = tf.reshape(x_embedded, [batch_size * longest_x, self.emb_dim])

    # case if french additional context
    if self.context=="concat":
      # shape [B, M*N, emb_dim]
      x_embedded = tf.tile(x_embedded, [1, longest_y, 1])
      # shape [B, N*M, emb_dim]
      y_prev_embedded = tf.tile(y_prev_embedded, [1, longest_x, 1])
      # shape [B, N, M*N]
      pa_x = tf.tile(pa_x, [1, 1, longest_y])
      x_mlp = tf.reshape(x_embedded, [batch_size * longest_x * longest_y, self.emb_dim])
      y_prev_mlp = tf.reshape(y_prev_embedded, [batch_size * longest_y * longest_x, self.emb_dim])
      # shape [B, M*N, 2*emb_dim]
      mlp_input = tf.concat([x_mlp, y_prev_mlp], axis=1)

    if self.context == "gate":
      # shape [B, M*N, emb_dim]
      x_embedded = tf.tile(x_embedded, [1, longest_y, 1])
      # shape [B, N*M, emb_dim]
      y_prev_embedded = tf.tile(y_prev_embedded, [1, longest_x, 1])
      # shape [B, M, M*N]
      pa_x = tf.tile(pa_x, [1, 1, longest_y])
      x_mlp = tf.reshape(x_embedded, [batch_size * longest_x * longest_y, self.emb_dim])
      y_prev_mlp = tf.reshape(y_prev_embedded, [batch_size * longest_y * longest_x, self.emb_dim])
      s = tf.sigmoid(y_prev_mlp)
      # shape [B, M*N, emb_dim]
      x_mlp = tf.tanh(x_mlp)
      y_prev_mlp = tf.tanh(y_prev_mlp)
      # element wise multiplication
      mlp_input = tf.multiply(x_mlp, 1 - s) + tf.multiply(y_prev_mlp, s)

    if self.context == "col_discrete":
      h_t = tf.matmul(mlp_input, self.mlp_Wt_, name='x1') + self.mlp_bt_  # Shape [B*M, emb_dim]
      h_t = tf.tanh(h_t)
      h_t = tf.matmul(h_t, self.mlp_W_t, name='x2') + self.mlp_b_t  # Shape [B*M, emb_dim]

      # P(F|Fprev) and P(C|Fprev), insertion i and collocation c
      # Use shared first layer!
      mlp_input = tf.reshape(y_prev_embedded, [batch_size * longest_y, self.emb_dim])
      h_is = tf.matmul(mlp_input, self.mlp_Wis_, name='y1') + self.mlp_bis_  # Shape [B*N, emb_dim]
      h_is = tf.tanh(h_is)
      # This is P(F|Fprev) insertion i
      h_i = tf.matmul(h_is, self.mlp_W_i, name='y2') + self.mlp_b_i  # Shape [B*N, emb_dim]
      py_y = tf.nn.softmax(h_i)  # Shape: [B*N, Vy]
      py_y = tf.reshape(py_y, [batch_size, longest_y, self.y_vocabulary_size])  # Shape [B, N, Vy]
      # This is s(Fprev) for P(C|Fprev) = Bern(s(Fprev))
      h_s = tf.matmul(h_is, self.mlp_W_s, name='3')
      h_s = h_s + self.mlp_b_s
      s = tf.sigmoid(h_s)
      s = tf.squeeze(s)  # get rid of trainling 1-dimension
      s = tf.reshape(s, [batch_size, longest_y])

      # Here we apply the MLP to our input.
    if self.context == "col_discrete":
      h = h_t
    else:
      h = tf.matmul(mlp_input, self.mlp_W_) + self.mlp_b_  # affine transformation
      h = tf.tanh(h)                                       # non-linearity
      h = tf.matmul(h, self.mlp_W) + self.mlp_b            # affine transformation [B * M, Vy]

    # You could also use TF fully connected to create the MLP.
    # Then you don't have to specify all the weights and biases separately.
    #h = tf.contrib.layers.fully_connected(mlp_input, self.mlp_dim, activation_fn=tf.tanh, trainable=True)
    #h = tf.contrib.layers.fully_connected(h, self.y_vocabulary_size, activation_fn=None, trainable=True)
    
    # Now we perform a softmax which operates on a per-row basis.
    py_xa = tf.nn.softmax(h)
    # add case for context
    if self.context=="concat" or self.context=="gate":
      py_xa = tf.reshape(py_xa, [batch_size, longest_x * longest_y, self.y_vocabulary_size])
    else:
      # This is P(F|E)
      py_xa = tf.reshape(py_xa, [batch_size, longest_x, self.y_vocabulary_size])  # Shape [B, M, Vy]
    # 2.c Marginalise alignments: \sum_a P(a|x) P(Y|x,a)

    # Here comes a rather fancy matrix multiplication.
    # Note that tf.matmul is defined to do a matrix multiplication
    # [N, M] @ [M, Vy] for each item in the first dimension B.
    # So in the final result we have B matrices [N, Vy], i.e. [B, N, Vy].
    #
    # We matrix-multiply:
    #   pa_x       Shape: [B, N, *M*]
    # and
    #   py_xa      Shape: [B, *M*, Vy]
    # to get 
    #   py_x  Shape: [B, N, Vy]
    #
    # Note: P(y|x) = prod_j p(y_j|x) = prod_j sum_aj p(aj|m)p(y_j|x_aj) 
    #
    py_x = tf.matmul(pa_x, py_xa)  # Shape: [B, N, Vy]
    if self.context=="col_discrete":
      s_tiled = tf.expand_dims(s, 2)  # Shape: [B, N, 1]
      s_tiled = tf.tile(s_tiled, [1, 1, self.y_vocabulary_size])  # Shape: [B, N, Vy]
      py_xa = tf.multiply(s_tiled, py_x, name='s1') + tf.multiply(1 - s_tiled, py_y, name='s2')

    # This calculates the accuracy, i.e. how many predictions we got right.
    predictions = tf.argmax(py_x, axis=2)
    acc = tf.equal(predictions, self.y)
    acc = tf.cast(acc, tf.float32) * y_mask
    acc_correct = tf.reduce_sum(acc)
    acc_total = tf.reduce_sum(y_mask)
    acc = acc_correct / acc_total
    
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=tf.reshape(self.y, [-1]),
      logits=tf.log(tf.reshape(py_x,[batch_size * longest_y, self.y_vocabulary_size])),
      name="logits"
    )
    cross_entropy = tf.reshape(cross_entropy, [batch_size, longest_y])
    cross_entropy = tf.reduce_sum(cross_entropy * y_mask, axis=1)
    cross_entropy = tf.reduce_mean(cross_entropy, axis=0)
    
    # Now we define our cross entropy loss 
    # Play with this if you want to try and replace TensorFlow's CE function.
    # Disclaimer: untested code
#     y_one_hot = tf.one_hot(self.y, depth=self.y_vocabulary_size)     # [B, N, Vy]
#     cross_entropy = tf.reduce_sum(y_one_hot * tf.log(py_x), axis=2)  # [B, N]
#     cross_entropy = tf.reduce_sum(cross_entropy * y_mask, axis=1)    # [B]
#     cross_entropy = -tf.reduce_mean(cross_entropy)  # scalar

    self.pa_x = pa_x
    self.py_x = py_x  
    self.py_xa = py_xa
    self.loss = cross_entropy
    self.predictions = predictions
    self.accuracy = acc
    self.accuracy_correct = tf.cast(acc_correct, tf.int64)
    self.accuracy_total = tf.cast(acc_total, tf.int64)
    if(self.context == "col_discrete"):
      self.s = s
      self.py_y = py_y
    
    
  def evaluate(self, data, ref_alignments, batch_size=4):
    """Evaluate the model on a data set."""
    
    ref_align = read_naacl_alignments(ref_alignments)
    
    ref_iterator = iter(ref_align)
    metric = AERSufficientStatistics()
    accuracy_correct = 0
    accuracy_total = 0
    
    for batch_id, batch in enumerate(iterate_minibatches(data, batch_size=batch_size)):
      x, y = prepare_data(batch, self.x_vocabulary, self.y_vocabulary)
      y_len = np.sum(np.sign(y), axis=1, dtype="int64")

      if self.context == "col_discrete":
        align, prob, acc_correct, acc_total = self.get_viterbi_col(x, y)
      else:
        align, prob, acc_correct, acc_total = self.get_viterbi(x, y)
      accuracy_correct += acc_correct
      accuracy_total += acc_total
      
#       if batch_id == 0:
#         print(batch[0])      
#      s = 0

      for alignment, N, (sure, probable) in zip(align, y_len, ref_iterator):
        # the evaluation ignores NULL links, so we discard them
        # j is 1-based in the naacl format
        pred = set((aj, j) for j, aj in enumerate(alignment[:N], 1) if aj > 0)
        metric.update(sure=sure, probable=probable, predicted=pred)
 #       print(batch[s])
 #       print(alignment[:N])
 #       print(pred)
 #       s +=1

    accuracy = accuracy_correct / float(accuracy_total)
    return metric.aer(), accuracy

    
  def get_viterbi(self, x, y):
    """Returns the Viterbi alignment for (x, y)"""

    y_prev = np.roll(y, 1, axis=1)
    y_prev[:, 0] = 0

    feed_dict = {
      self.x : x,  # English
      self.y : y,   # French
      self.y_prev : y_prev
    }
    
    # run model on this input
    py_xa, acc_correct, acc_total = self.session.run(
      [self.py_xa, self.accuracy_correct, self.accuracy_total], 
      feed_dict=feed_dict)
    
    # things to return
    batch_size, longest_y = y.shape
    _, longest_x = x.shape
    alignments = np.zeros((batch_size, longest_y), dtype="int64")
    probabilities = np.zeros((batch_size, longest_y), dtype="float32")

    for b, sentence in enumerate(y):
      fprev = 0
      for j, french_word in enumerate(sentence):
        if french_word == 0:  # Padding
          break

        # index by previous f word. Keep track.
        # py_xa      Shape: [B, *N*M*, Vy]
        # probs = py_xa[b,:,y[b,j]] # y[b,j] means only the word f_j in the sentence b
        fprevs = [fprev + (k * longest_y) for k in range(longest_x)]
        probs = py_xa[b, fprevs, y[b, j]]  # y[b,j] means only the word f_j in the sentence b
        a_j = probs.argmax()
        p_j = probs[a_j]

        alignments[b, j] = a_j
        probabilities[b, j] = p_j

        fprev = j
    
    return alignments, probabilities, acc_correct, acc_total

  def get_viterbi_col(self, x, y):
    """Returns the Viterbi alignment for (x, y)"""

    y_prev = np.roll(y, 1, axis=1)
    y_prev[:, 0] = 0

    feed_dict = {
      self.x: x,  # English
      self.y: y,  # French
      self.y_prev : y_prev
    }

    # run model on this input
    py_xa, py_y, s, acc_correct, acc_total = self.session.run(
      [self.py_xa,
       self.py_y,
       self.s,
       self.accuracy_correct,
       self.accuracy_total],
       feed_dict=feed_dict)

    # things to return
    batch_size, longest_y = y.shape
    _, longest_x = x.shape
    alignments = np.zeros((batch_size, longest_y), dtype="int64")
    probabilities = np.zeros((batch_size, longest_y), dtype="float32")

    for b, sentence in enumerate(y):
      for j, french_word in enumerate(sentence):
        if french_word == 0:  # Padding
          break
        fprev = j
        sj = s[b,j]
        # if b in range(20): print(sj)
        c = int(np.random.uniform() < sj) # sample c ~ Bernouilli(sj)
        if c == 0: # then we align
            probs = py_xa[b, : , y[b,j]] # y[b,j] means only the word f_j in the sentence b
            a_j = probs.argmax()
            p_j = probs[a_j]
        if c == 1: # then we `insert` (i.e. NULL align - see NLP2 blog post)
            # if b in range(20): print('Null aligned')
            a_j = 0 # NULL align
            p_j = 1

        alignments[b, j] = a_j
        probabilities[b, j] = p_j


    return alignments, probabilities, acc_correct, acc_total


  
