# SDNE
import tensorflow as tf
#import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import networkx as nx


def fc_op(input_op, name, n_out, layer_collector, act_func=tf.nn.leaky_relu):
    n_in = input_op.get_shape()[-1]  #去掉value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w", shape=[n_in, n_out],
                                 initializer=tf.keras.initializers.glorot_normal(), dtype=tf.float32)
        biases = tf.Variable(tf.constant(0, shape=[1, n_out], dtype=tf.float32), name=scope + 'b')

        fc = tf.add(tf.matmul(input_op, kernel), biases)
        activation = act_func(fc, name=scope + 'act')
        layer_collector.append([kernel, biases])
        return activation

class SDNE(object):
    #这是SDNE类的构造函数（初始化方法）。当创建一个新的SDNE对象时，该方法会被自动调用来进行初始化设置。
    def __init__(self, graph, encoder_layer_list, alpha=0.01, beta=10., nu=1e-5,
                 batch_size=100, max_iter=500, learning_rate=0.01, adj_mat=None):
        # graph：这是一个图数据，表示图结构的对象。通常用邻接矩阵或邻接表表示。
        # encoder_layer_list：这是一个列表，包含SDNE算法编码器（encoder）的每一层的节点数。列表的最后一个元素表示最终的节点表示向量的维度。
        # alpha: 这是一个浮点数，用于控制SDNE算法中的重构误差项（reconstruction term）的权重
        # beta: 这是一个浮点数，用于控制SDNE算法中的正则化项（regularization term）的权重。
        # nu: 这是一个浮点数，用于控制SDNE算法的学习速率（learning rate）或步长（step size）。
        # batch_size: 这是一个整数，表示SDNE算法中每次迭代所使用的批次大小（batch size）。批次大小决定了每次更新模型时使用的样本数量。
        # max_iter: 这是一个整数，表示SDNE算法的最大迭代次数。算法会在达到最大迭代次数或其他终止条件时停止优化。
        # learning_rate: 这是一个浮点数，表示SDNE算法的学习率。学习率决定了每次参数更新的幅度。
        # adj_mat: 这是一个可选的参数，表示图数据的邻接矩阵。在一些实现中，可能需要直接传递邻接矩阵作为参数，而不是通过graph参数来表示图数据。


        self.g = graph
        self.node_size = self.g.G.number_of_nodes()#计算图中节点的数量，并将其赋值给类的成员变量self.node_size。self.g.G表示图数据中的图对象，number_of_nodes()是图对象的方法，用于计算节点的数量。
        self.rep_size = encoder_layer_list[-1]#将传递给初始化函数的encoder_layer_list列表的最后一个元素赋值给类的成员变量self.rep_size。
        # encoder_layer_list是SDNE算法中编码器每一层的节点数列表，[-1]表示列表的最后一个元素，即为最终的节点表示向量维度
        self.encoder_layer_list = [self.node_size]#初始化self.encoder_layer_list列表，并将其设为包含图中节点数self.node_size的列表。这是为了确保编码器的输入层节点数与图的节点数一致。
        self.encoder_layer_list.extend(encoder_layer_list)#将传递的encoder_layer_list列表中的元素添加到self.encoder_layer_list列表中，以构建完整的编码器层节点数列表
        self.encoder_layer_num = len(encoder_layer_list)+1#计算编码器的总层数，并将其赋值给类的成员变量self.encoder_layer_num。编码器的层数为encoder_layer_list中元素的数量加1，因为还包括输入层的节点数。
        #self.g存储图数据，
        # self.node_size存储图中节点数量，
        # self.rep_size存储节点表示向量的维度，
        # self.encoder_layer_list存储编码器每一层的节点数列表，
        # self.encoder_layer_num存储编码器的总层数。


        self.alpha = alpha
        self.beta = beta
        self.nu = nu
        self.bs = batch_size
        self.max_iter = max_iter
        self.lr = learning_rate

        # self.sess = tf.Session()
        self.sess = tf.compat.v1.Session()
        self.vectors = {}

        self.adj_mat = nx.to_numpy_array(self.g.G)
        self.embeddings = self.get_train()

        look_back = self.g.look_back_list

        for i, embedding in enumerate(self.embeddings):
            self.vectors[look_back[i]] = embedding

    def get_train(self):
        adj_mat = self.adj_ma

        AdjBatch = tf.placeholder(tf.float32, [None, self.node_size], name='adj_batch')
        Adj = tf.placeholder(tf.float32, [None, None], name='adj_mat')
        B = tf.placeholder(tf.float32, [None, self.node_size], name='b_mat')

        fc = AdjBatch
        scope_name = 'encoder'
        layer_collector = []

        with tf.name_scope(scope_name):
            for i in range(1, self.encoder_layer_num):
                print("encoder" + str(i))
                fc = fc_op(fc,
                           name=scope_name+str(i),
                           n_out=self.encoder_layer_list[i],
                           layer_collector=layer_collector)

        _embeddings = fc

        scope_name = 'decoder'
        with tf.name_scope(scope_name):
            for i in range(self.encoder_layer_num-2, 0, -1):
                print("decoder" + str(i))
                fc = fc_op(fc,
                           name=scope_name+str(i),
                           n_out=self.encoder_layer_list[i],
                           layer_collector=layer_collector)
            fc = fc_op(fc,
                       name=scope_name+str(0),
                       n_out=self.encoder_layer_list[0],
                       layer_collector=layer_collector,)

        _embeddings_norm = tf.reduce_sum(tf.square(_embeddings), 1, keepdims=True)

        L_1st = tf.reduce_sum(
            Adj * (
                    _embeddings_norm - 2 * tf.matmul(
                        _embeddings, tf.transpose(_embeddings)
                    ) + tf.transpose(_embeddings_norm)
            )
        )

        L_2nd = tf.reduce_sum(tf.square((AdjBatch - fc) * B))

        L = L_2nd + self.alpha * L_1st

        for param in layer_collector:
            L += self.nu * (tf.reduce_sum(tf.square(param[0]) + tf.abs(param[0])))

        optimizer = tf.train.AdamOptimizer()

        train_op = optimizer.minimize(L)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        for step in range(self.max_iter):
            index = np.random.randint(self.node_size, size=self.bs)
            adj_batch_train = adj_mat[index, :]
            adj_mat_train = adj_batch_train[:, index]
            b_mat_train = 1.*(adj_batch_train <= 1e-10) + self.beta * (adj_batch_train > 1e-10)

            self.sess.run(train_op, feed_dict={AdjBatch: adj_batch_train,
                                               Adj: adj_mat_train,
                                               B: b_mat_train})
            if step % 20 == 0:
                print("step %i: %s" % (step, self.sess.run([L, L_1st, L_2nd],
                                                           feed_dict={AdjBatch: adj_batch_train,
                                                                      Adj: adj_mat_train,
                                                                      B: b_mat_train})))

        return self.sess.run(_embeddings, feed_dict={AdjBatch: adj_mat})

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors)
        fout.write("{} {}\n".format(node_num, self.rep_size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
        fout.close()
