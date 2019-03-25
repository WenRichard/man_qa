# -*- coding: utf-8 -*-
# @Time    : 2019/3/19 16:21
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : model.py
# @Software: PyCharm

import tensorflow as tf
from model_utils import *
from tensorflow.contrib import rnn


class MAN(object):
    def __init__(self, config):
        self.ques_len = config.ques_length
        self.ans_len = config.ans_length
        self.hidden_size = config.hidden_size
        self.output_size = config.output_size
        self.rnn_size = config.rnn_size
        self.learning_rate = config.learning_rate
        self.optimizer = config.optimizer
        self.l2_lambda = config.l2_lambda
        self.clip_value = config.clip_value
        self.embeddings = config.embeddings
        self.window_sizes = config.window_sizes
        self.n_filters = config.n_filters
        self.margin = config.margin
        self.num_steps = config.num_steps
        self.layer_size = config.layer_size

        self._placeholder_init_pointwise()
        self._initialize_weights()
        self.q_a_cosine, self.q_aneg_cosine = self._build(self.embeddings)
        # 损失和精确度
        self.total_loss, self.accu = self._add_loss_op(self.q_a_cosine, self.q_aneg_cosine, self.l2_lambda)
        # 训练节点
        self.train_op = self._add_train_op(self.total_loss)

    def _placeholder_init_pointwise(self):
        self._ques = tf.placeholder(tf.int32, [None, self.ques_len], name='ques_point')
        self._ans = tf.placeholder(tf.int32, [None, self.ans_len], name='ans_point')
        self._ans_neg = tf.placeholder(tf.int32, [None, self.ans_len], name='ans_point')
        self._ques_mask = tf.placeholder(tf.int32, [None], 'ques_mask')
        self._ans_mask = tf.placeholder(tf.int32, [None], 'ans_mask')
        self._ans_neg_mask = tf.placeholder(tf.int32, [None], 'ans_neg_mask')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size, self.list_size = tf.shape(self._ans)[0], tf.shape(self._ans)[1]

    def _initialize_weights(self):
        self.W_q1 = weight_variable('W_q1', [2 * self.rnn_size, 2*self.rnn_size])
        self.W_m1 = weight_variable('W_m1', [2 * self.rnn_size, 2 * self.rnn_size])
        self.W_s1 = weight_variable('W_s1', [2 * self.rnn_size, 1])
        self.W_q2 = weight_variable('W_q2', [2 * self.rnn_size, 2*self.rnn_size])
        self.W_m2 = weight_variable('W_m2', [2 * self.rnn_size, 2 * self.rnn_size])
        self.W_s2 = weight_variable('W_s2', [2 * self.rnn_size, 1])
        self.W_q3 = weight_variable('W_q3', [2 * self.rnn_size, 2*self.rnn_size])
        self.W_m3 = weight_variable('W_m3', [2 * self.rnn_size, 2 * self.rnn_size])
        self.W_s3 = weight_variable('W_s3', [2 * self.rnn_size, 1])

    def _bilstm_layer(self, inputs, rnn_size, seq_len, layer_size, batch_size, keep_prob, scope, reuse=False):
        """
        双向LSTM
        """
        with tf.variable_scope(scope, reuse=reuse):
            def cell():
                lstm_cell = rnn.LSTMCell(num_units=rnn_size)
                lstm_cell = rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)  ##
                return lstm_cell
            cell_bw = cell_fw = tf.nn.rnn_cell.MultiRNNCell([cell() for _ in range(layer_size)])
            cell_fw_initial = cell_fw.zero_state(batch_size, tf.float32)
            cell_bw_initial = cell_bw.zero_state(batch_size, tf.float32)
            output = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                     cell_bw=cell_bw,
                                                     inputs=inputs,
                                                     initial_state_fw=cell_fw_initial,
                                                     initial_state_bw=cell_bw_initial,
                                                     sequence_length=seq_len,
                                                     dtype=tf.float32)
            return output

    def _multihop_layer(self, q):
        with tf.variable_scope("MultihopAttention_layer", reuse=False):
            '''
            主要是为了更新o_q(k)
            1. o_q[0] = sum(h_q(t)) / l
            2. m_q[0] = sum(h_q(t)) / l
            3. M = tanh(W_q(k).h_q(t))
            4. N = tanh(W_m(k).m_q(k-1))
            5. s_t = tanh(W_q(k).h_q(t)) * tanh(W_m(k).m_q(k-1))
            6. w = w_s(k).s_t(k)
            7. alpha_t = softmax(w_s(k).s_t(k))
            8. o_q[1] = sum( alpha_t * h_q(t))
             '''
            # num_steps = 1的情况
            m_q = [None] * self.num_steps
            m_q[0] = tf.reduce_mean(q, axis=1)  # (bz, 2hz)
            o_q = [None] * (self.num_steps + 1)
            o_q[0] = tf.reduce_mean(q, axis=1)  # (bz, 2hz)
            M = tf.tanh(multiply_3_2(q, self.W_q1))  # (bz, len, 2hz)
            N = tf.expand_dims(tf.tanh(tf.matmul(m_q[0], self.W_m1)), axis=1)  # (bz, 1, 2hz)
            s_t = tf.multiply(M, N)  # (bz, len, 2hz)
            w = multiply_3_2(s_t, self.W_s1)  # (bz, len, 1)
            alpha_t = tf.nn.softmax(w, axis=1)  # (bz, len, 1)
            o_q[1] = tf.reduce_sum(tf.multiply(q, alpha_t), axis=1)  # (bz, 2hz)

            if self.num_steps > 1:  # num_steps = 2的情况
                m_q[1] = m_q[0] + o_q[1]  # (bz, 2hz)
                M = tf.tanh(multiply_3_2(q, self.W_q2))  # (bz, len, 2hz)
                N = tf.expand_dims(tf.tanh(tf.matmul(m_q[1], self.W_m2)), axis=1)  # (bz, 1, 2hz)
                s_t = tf.multiply(M, N)  # (bz, len, 2hz)
                w = multiply_3_2(s_t, self.W_s2)  # (bz, len, 1)
                alpha_t = tf.nn.softmax(w, axis=1)  # (bz, len, 1)
                o_q[2] = tf.reduce_sum(tf.multiply(q, alpha_t), axis=1)  # (bz, 2hz)
            if self.num_steps > 2:  # num_steps = 3的情况
                m_q[2] = m_q[1] + o_q[2]  # (bz, 2hz)
                M = tf.tanh(multiply_3_2(q, self.W_q3))  # (bz, len, 2hz)
                N = tf.expand_dims(tf.tanh(tf.matmul(m_q[2], self.W_m3)), axis=1)  # (bz, 1, 2hz)
                s_t = tf.multiply(M, N)  # (bz, len, 2hz)
                w = multiply_3_2(s_t, self.W_s3)  # (bz, len, 1)
                alpha_t = tf.nn.softmax(w, axis=1)  # (bz, len, 1)
                o_q[3] = tf.reduce_sum(tf.multiply(q, alpha_t), axis=1)  # (bz, 2hz)
            return o_q

    def _sequential_layer(self, inputs1, inputs2, ans_mask, rnn_size, batch_size, seq_len, dropout_keep_prob, index,
                          isreuse = False):
        """
        论文地址：
        .代表矩阵乘法
        1. Gama = Gama_1 = j *W.h_i
        2. outputs_n = n_i = GRU(n_i, gama_i)
        3. n_ = n_i = concat(n_i, n_i)
        4. alpha = alpha_i = softmax(1(T).n_i)
        5. partial_bilinear_output = O = sum(alpha * h_)
        6. pred = a = argmax(M_a.O)    在本模型中，这一步取消
        """
        j = tf.expand_dims(inputs1, axis=1)  # (bz, 1, 2hz)
        h = inputs2  # (bz, len, 2hz)
        # W_h = tf.matmul(h, self.W)
        Gama = tf.multiply(j, h)  # (bz, len, 2hz)
        print_shape('Gama', Gama)
        outputs_n, finalState_n = self._bilstm_layer(Gama, rnn_size, seq_len, 1, batch_size, 1.0,
                                                     'sequentional_attention_{}'.format(index), reuse=isreuse)
        n_ = tf.concat(outputs_n, axis=2)  # (bz, len, 2hz)
        alpha = tf.nn.softmax(tf.reduce_sum(n_, axis=2), axis=-1)  # (bz, len)
        # 将句子padding部分权重消除
        # alpha = tf.expand_dims(tf.multiply(alpha, ans_mask), axis=-1)
        alpha = tf.expand_dims(alpha, axis=-1)
        partial_bilinear_output = tf.reduce_sum(tf.multiply(alpha, h), axis=1)  # (bz, 2hz)
        print_shape('SequentialAttention_output', partial_bilinear_output)
        return partial_bilinear_output

    def _match_layer(self, q, a_pos, a_neg, p_mask, n_mask, rnn_size, batch_size, p_seq_len, n_seq_len, dropout_keep_prob):
        for i in range(self.num_steps+1):
            pos = self._sequential_layer(q[i], a_pos, p_mask, rnn_size, batch_size, p_seq_len, dropout_keep_prob, i+1,
                                         isreuse=False)
            neg = self._sequential_layer(q[i], a_neg, n_mask, rnn_size, batch_size, n_seq_len, dropout_keep_prob, i+1,
                                         isreuse=True)
            q_ = tf.nn.l2_normalize(q[i], dim=1)
            as_pos = tf.nn.l2_normalize(pos, dim=1)
            as_neg = tf.nn.l2_normalize(neg, dim=1)
            q_pos_cosine = tf.reduce_sum(tf.multiply(q_, as_pos), 1)
            q_neg_cosine = tf.reduce_sum(tf.multiply(q_, as_neg), 1)
            sim_pos = q_pos_cosine if i == 0 else sim_pos + q_pos_cosine
            sim_neg = q_neg_cosine if i == 0 else sim_neg + q_neg_cosine
        return sim_pos, sim_neg

    def _build(self, embeddings, rnn_type = 'lstm'):
        self.Embedding = tf.Variable(tf.to_float(embeddings), trainable=False, name='Embedding')
        self.q_embed = tf.nn.dropout(tf.nn.embedding_lookup(self.Embedding, self._ques), keep_prob=self.dropout_keep_prob)
        self.a_embed = tf.nn.dropout(tf.nn.embedding_lookup(self.Embedding, self._ans), keep_prob=self.dropout_keep_prob)
        self.a_neg_embed = tf.nn.dropout(tf.nn.embedding_lookup(self.Embedding, self._ans_neg), keep_prob=self.dropout_keep_prob)
        q_mask = tf.sequence_mask(self._ques_mask, self.ques_len, dtype=tf.float32)
        a_mask = tf.sequence_mask(self._ans_mask, self.ans_len, dtype=tf.float32)
        a_neg_mask = tf.sequence_mask(self._ans_neg_mask, self.ans_len, dtype=tf.float32)

        # 上下文编码
        if rnn_type == 'lstm':
            q_outputs, q_final_state = self._bilstm_layer(self.q_embed, self.rnn_size, self._ques_mask, self.layer_size,
                                                          self.batch_size, self.dropout_keep_prob, 'lstm')
            a_outputs, a_final_state = self._bilstm_layer(self.a_embed, self.rnn_size, self._ans_mask, self.layer_size,
                                                          self.batch_size, self.dropout_keep_prob, 'lstm',
                                                          reuse=True)
            a_neg_outputs, a_neg_final_state = self._bilstm_layer(self.a_embed, self.rnn_size, self._ans_neg_mask,
                                                                  self.layer_size, self.batch_size,
                                                                  self.dropout_keep_prob, 'lstm', reuse=True)
        elif rnn_type == 'gru':
            q_outputs, q_final_state = self._bilstm_layer(self.q_embed, self.rnn_size, self._ques_mask, self.layer_size,
                                                          self.batch_size, self.dropout_keep_prob, 'gru')
            a_outputs, a_final_state = self._bilstm_layer(self.a_embed, self.rnn_size, self._ans_mask, self.layer_size,
                                                          self.batch_size, self.dropout_keep_prob, 'gru',
                                                          reuse=True)
            a_neg_outputs, a_neg_final_state = self._bilstm_layer(self.a_embed, self.rnn_size, self._ans_neg_mask,
                                                                  self.layer_size, self.batch_size,
                                                                  self.dropout_keep_prob, 'gru', reuse=True)
        rnn_q = tf.concat(q_outputs, axis=-1)
        rnn_a = tf.concat(a_outputs, axis=-1)
        rnn_a_neg = tf.concat(a_neg_outputs, axis=-1)

        o_q = self._multihop_layer(rnn_q)
        sim_pos, sim_neg = self._match_layer(o_q, rnn_a, rnn_a_neg, a_mask, a_neg_mask, self.rnn_size, self.batch_size,
                                             self._ans_mask, self._ans_neg_mask, self.dropout_keep_prob)
        return sim_pos, sim_neg

    def _margin_loss(self, pos_sim, neg_sim):
        original_loss = self.margin - pos_sim + neg_sim
        l = tf.maximum(tf.zeros_like(original_loss), original_loss)
        loss = tf.reduce_sum(l)
        return loss, l

    def _add_loss_op(self, p_sim, n_sim, l2_lambda=0.0001):
        """
        损失节点
        """
        loss, l = self._margin_loss(p_sim, n_sim)
        accu = tf.reduce_mean(tf.cast(tf.equal(0., l), tf.float32))
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        l2_loss = sum(reg_losses) * l2_lambda
        pairwise_loss = loss + l2_loss
        tf.summary.scalar('pairwise_loss', pairwise_loss)
        self.summary_op = tf.summary.merge_all()
        return pairwise_loss, accu

    def _add_train_op(self, loss):
        """
        训练节点
        """
        with tf.name_scope('train_op'):
            # 记录训练步骤
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            opt = tf.train.AdamOptimizer(self.learning_rate)
            # train_op = opt.minimize(loss, self.global_step)

            gradients, v = zip(*opt.compute_gradients(loss))
            clip_gradients = gradients
            if self.clip_value is not None:
                clip_gradients, _ = tf.clip_by_global_norm(gradients, self.clip_value)
            train_op = opt.apply_gradients(zip(clip_gradients, v), global_step= self.global_step)
        return train_op
