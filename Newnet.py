import tensorflow as tf
from keras.layers import BatchNormalization, Activation, Add
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Embedding, Convolution1D, MaxPooling1D, Concatenate, Dropout, AveragePooling1D, GlobalAveragePooling1D, Multiply, Reshape, GlobalMaxPooling1D
from tensorflow.keras.layers import Flatten, Dense, LSTM, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam,Adagrad
from tensorflow.keras.layers import Bidirectional
from tensorflow.python.keras import backend as K
from tensorflow.keras import initializers
# from tensorflow.keras.layers import MultiHeadAttention

def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
         不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
         本文。
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = K.zeros_like(y_pred[..., :1])
    y_pred_neg = K.concatenate([y_pred_neg, zeros], axis=-1)
    y_pred_pos = K.concatenate([y_pred_pos, zeros], axis=-1)
    # neg_loss = K.logsumexp(y_pred_neg, axis=-1)
    neg_loss = K.logsumexp(y_pred_neg, axis=-1)
    pos_loss = K.logsumexp(y_pred_pos, axis=-1)
    return neg_loss + pos_loss


# 公式：L(pt) = -αt(1-pt)^γ log(pt)，
# pt=p and αt=α  when y=1 ,pt=1-p and αt=1-α when y=-1或者0 视情况而定
def focal_loss(alpha=0.5, gamma=1.5, epsilon=1e-6):
    print('*' * 20, 'alpha={}, gamma={}'.format(alpha, gamma))

    def focal_loss_calc(y_true, y_probs):
        positive_pt = tf.where(tf.equal(y_true, 1), y_probs, tf.ones_like(y_probs))
        negative_pt = tf.where(tf.equal(y_true, 0), 1 - y_probs, tf.ones_like(y_probs))

        loss = -alpha * tf.pow(1 - positive_pt, gamma) * tf.math.log(tf.clip_by_value(positive_pt, epsilon, 1.)) - \
               (1 - alpha) * tf.pow(1 - negative_pt, gamma) * tf.math.log(tf.clip_by_value(negative_pt, epsilon, 1.))

        return tf.reduce_sum(loss)

    return focal_loss_calc

def focal_loss_calc(y_true, y_probs):
    alpha = 0.25
    gamma = 2
    epsilon = 1e-6
    positive_pt = tf.where(tf.equal(y_true, 1), y_probs, tf.ones_like(y_probs))
    negative_pt = tf.where(tf.equal(y_true, 0), 1 - y_probs, tf.ones_like(y_probs))

    loss = -alpha * tf.pow(1 - positive_pt, gamma) * tf.math.log(tf.clip_by_value(positive_pt, epsilon, 1.)) - \
           (1 - alpha) * tf.pow(1 - negative_pt, gamma) * tf.math.log(tf.clip_by_value(negative_pt, epsilon, 1.))

    return tf.reduce_sum(loss)


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim=32, num_heads=8, **kwargs):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)


        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def get_config(self):
        # 自定义层里面的属性， 需要根据自己层里面的属性进行替换
         config = {
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'projection_dim': self.projection_dim,
            'query_dense': self.query_dense,
            'key_dense': self.key_dense,
            'value_dense': self.value_dense,
            'combine_heads': self.combine_heads
          }
         base_config = super(MultiHeadSelfAttention, self).get_config()
         return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim=32, num_heads=2, ff_dim=32, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
            # [
            #     layers.Dense(ff_dim),
            #     layers.Activation(tf.keras.activations.gelu),  # 使用 GELU 激活函数
            #     layers.Dense(embed_dim),
            # ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        # 自定义层里面的属性， 需要根据自己层里面的属性进行替换
         config = {
            'embed_dim':self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
            'att': self.att,
            'ffn': self.ffn,
            'layernorm1': self.layernorm1,
            'layernorm2': self.layernorm2,
            'dropout1': self.dropout1,
            'dropout2': self.dropout2
          }
         base_config = super(TransformerBlock, self).get_config()
         return dict(list(base_config.items()) + list(config.items()))

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen=50, vocab_size=7899, embed_dim=32, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        # 自定义层里面的属性， 需要根据自己层里面的属性进行替换
         config = {
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'token_emb': self.token_emb,
            'pos_emb': self.pos_emb
          }
         base_config = super(TokenAndPositionEmbedding, self).get_config()
         return dict(list(base_config.items()) + list(config.items()))




embed_dim = 192  # Embedding size for each token
num_heads = 8  # Number of attention heads
ff_dim = 64  # Hidden layer size in feed forward network inside transformer
vocab_size = 7899

def Newnet(length, length_f, out_length, para):
    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    # 编码模块
    main_input = Input(shape=(length,), dtype='int64', name='main_input')
    embedding_layer = TokenAndPositionEmbedding(length, vocab_size, embed_dim)
    x = embedding_layer(main_input)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)

    # 特征提取模块
    a = Convolution1D(64, 2, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    apool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(a)
    apool2 = AveragePooling1D(pool_size=ps, strides=1, padding='same')(a)
    apool = apool2 + apool
    apool = tf.keras.layers.Bidirectional(LSTM(units=50, return_sequences=True))(apool)

    b = Convolution1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    bpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(b)
    bpool2 = AveragePooling1D(pool_size=ps, strides=1, padding='same')(b)
    bpool = bpool2 + bpool
    bpool = tf.keras.layers.Bidirectional(LSTM(units=50, return_sequences=True))(bpool)

    d = Convolution1D(64, 5, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    dpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(d)
    dpool2 = AveragePooling1D(pool_size=ps, strides=1, padding='same')(d)
    dpool = dpool2 + dpool
    dpool = tf.keras.layers.Bidirectional(LSTM(units=50, return_sequences=True))(dpool)

    c = Convolution1D(64, 1, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    cpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(c)
    cpool2 = AveragePooling1D(pool_size=ps, strides=1, padding='same')(c)
    cpool = cpool2 + cpool
    cpool = tf.keras.layers.Bidirectional(LSTM(units=50, return_sequences=True))(cpool)

    merge = Concatenate(axis=-1)([apool, bpool, dpool, cpool])
    x = Dropout(0.5)(merge)
    x = Flatten()(x)

    # 前馈网络
    # x = tf.keras.layers.Dense(1000, activation='relu', name="dense4", kernel_regularizer=l2(l2value))(x)
    # s = x
    # x = tf.keras.layers.Dense(2000, activation='relu', name="dense5", kernel_regularizer=l2(l2value))(x)
    # x = tf.keras.layers.Dense(1000, activation='relu', name="dense6", kernel_regularizer=l2(l2value))(x)
    # x = s + x
    # x = tf.keras.layers.Dropout(0.5)(x)

    # 密集Dropout分类模块
    x = tf.keras.layers.Dense(1000, activation='relu', name="dense1", kernel_regularizer=l2(l2value))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu', name="dense2", kernel_regularizer=l2(l2value))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64, activation='relu', name="dense3", kernel_regularizer=l2(l2value))(x)

    output = Dense(out_length, name='output', kernel_regularizer=l2(l2value))(x)

    model = Model(inputs=main_input, outputs=output)
    adam = Adam(learning_rate=0.0015)
    model.compile(optimizer=adam, loss=multilabel_categorical_crossentropy, metrics=['accuracy'])

    model.summary()

    return model

def Newnet_loss(length, length_f, out_length, para):
    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    # 编码模块
    main_input = Input(shape=(length,), dtype='int64', name='main_input')
    embedding_layer = TokenAndPositionEmbedding(length, vocab_size, embed_dim)
    x = embedding_layer(main_input)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)

    # 特征提取模块
    a = Convolution1D(64, 2, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    apool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(a)
    apool2 = AveragePooling1D(pool_size=ps, strides=1, padding='same')(a)
    apool = apool2 + apool
    apool = tf.keras.layers.Bidirectional(LSTM(units=50, return_sequences=True))(apool)

    b = Convolution1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    bpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(b)
    bpool2 = AveragePooling1D(pool_size=ps, strides=1, padding='same')(b)
    bpool = bpool2 + bpool
    bpool = tf.keras.layers.Bidirectional(LSTM(units=50, return_sequences=True))(bpool)

    d = Convolution1D(64, 5, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    dpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(d)
    dpool2 = AveragePooling1D(pool_size=ps, strides=1, padding='same')(d)
    dpool = dpool2 + dpool
    dpool = tf.keras.layers.Bidirectional(LSTM(units=50, return_sequences=True))(dpool)

    c = Convolution1D(64, 1, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    cpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(c)
    cpool2 = AveragePooling1D(pool_size=ps, strides=1, padding='same')(c)
    cpool = cpool2 + cpool
    cpool = tf.keras.layers.Bidirectional(LSTM(units=50, return_sequences=True))(cpool)

    merge = Concatenate(axis=-1)([apool, bpool, dpool, cpool])
    x = Dropout(0.5)(merge)
    x = Flatten()(x)

    # 前馈网络
    # x = tf.keras.layers.Dense(1000, activation='relu', name="dense4", kernel_regularizer=l2(l2value))(x)
    # s = x
    # x = tf.keras.layers.Dense(2000, activation='relu', name="dense5", kernel_regularizer=l2(l2value))(x)
    # x = tf.keras.layers.Dense(1000, activation='relu', name="dense6", kernel_regularizer=l2(l2value))(x)
    # x = s + x
    # x = tf.keras.layers.Dropout(0.5)(x)

    # 密集Dropout分类模块
    x = tf.keras.layers.Dense(1000, activation='relu', name="dense1", kernel_regularizer=l2(l2value))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu', name="dense2", kernel_regularizer=l2(l2value))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64, activation='relu', name="dense3", kernel_regularizer=l2(l2value))(x)

    output = Dense(out_length, activation='sigmoid', name='output', kernel_regularizer=l2(l2value))(x)

    model = Model(inputs=main_input, outputs=output)
    adam = Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=adam, loss=dice_loss, metrics=['accuracy'])

    model.summary()

    return model

class MultiHeadAttention(layers.Layer):
    def __init__(self, output_dim, num_head, kernel_initializer='glorot_uniform', **kwargs):
        self.output_dim = output_dim
        self.num_head = num_head
        self.kernel_initializer = initializers.get(kernel_initializer)
        super(MultiHeadAttention, self).__init__(**kwargs)

    def get_config(self):
        return {"output_dim": self.output_dim, "num_head": self.num_head}

    def build(self, input_shape):
        self.W = self.add_weight(name='W',
                                 shape=(self.num_head, 3, input_shape[2], self.output_dim),
                                 initializer=self.kernel_initializer,
                                 trainable=True)
        self.Wo = self.add_weight(name='Wo',
                                  shape=(self.num_head * self.output_dim, self.output_dim),
                                  initializer=self.kernel_initializer,
                                  trainable=True)
        self.built = True

    def call(self, x):
        q = K.dot(x, self.W[0, 0])
        k = K.dot(x, self.W[0, 1])
        v = K.dot(x, self.W[0, 2])
        e = K.batch_dot(q, K.permute_dimensions(k, [0, 2, 1]))  # 把k转置，并与q点乘
        e = e / (self.output_dim ** 0.5)
        e = K.softmax(e)
        outputs = K.batch_dot(e, v)
        for i in range(1, self.W.shape[0]):
            q = K.dot(x, self.W[i, 0])
            k = K.dot(x, self.W[i, 1])
            v = K.dot(x, self.W[i, 2])
            # print('q_shape:'+str(q.shape))
            e = K.batch_dot(q, K.permute_dimensions(k, [0, 2, 1]))  # 把k转置，并与q点乘
            e = e / (self.output_dim ** 0.5)
            e = K.softmax(e)
            # print('e_shape:'+str(e.shape))
            o = K.batch_dot(e, v)
            outputs = K.concatenate([outputs, o])
        z = K.dot(outputs, self.Wo)
        return z

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

def PrMABP(length, length_f, out_length, para):
    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    # 编码模块
    main_input = Input(shape=(length,), dtype='int64', name='main_input')
    embedding_layer = TokenAndPositionEmbedding(length, vocab_size, embed_dim)
    x = embedding_layer(main_input)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)

    # 特征提取模块
    a = Convolution1D(64, 2, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    apool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(a)
    apool2 = AveragePooling1D(pool_size=ps, strides=1, padding='same')(a)
    apool = apool2 + apool
    apool = tf.keras.layers.Bidirectional(LSTM(units=50, return_sequences=True))(apool)

    b = Convolution1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    bpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(b)
    bpool2 = AveragePooling1D(pool_size=ps, strides=1, padding='same')(b)
    bpool = bpool2 + bpool
    bpool = tf.keras.layers.Bidirectional(LSTM(units=50, return_sequences=True))(bpool)

    d = Convolution1D(64, 5, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    dpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(d)
    dpool2 = AveragePooling1D(pool_size=ps, strides=1, padding='same')(d)
    dpool = dpool2 + dpool
    dpool = tf.keras.layers.Bidirectional(LSTM(units=50, return_sequences=True))(dpool)

    c = Convolution1D(64, 1, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    cpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(c)
    cpool2 = AveragePooling1D(pool_size=ps, strides=1, padding='same')(c)
    cpool = cpool2 + cpool
    cpool = tf.keras.layers.Bidirectional(LSTM(units=50, return_sequences=True))(cpool)

    merge = Concatenate(axis=-1)([apool, bpool, dpool, cpool])
    x = Dropout(0.5)(merge)
    x = Flatten()(x)

    # 前馈网络
    # x = tf.keras.layers.Dense(1000, activation='relu', name="dense4", kernel_regularizer=l2(l2value))(x)
    # s = x
    # x = tf.keras.layers.Dense(2000, activation='relu', name="dense5", kernel_regularizer=l2(l2value))(x)
    # x = tf.keras.layers.Dense(1000, activation='relu', name="dense6", kernel_regularizer=l2(l2value))(x)
    # x = s + x
    # x = tf.keras.layers.Dropout(0.5)(x)

    # 密集Dropout分类模块
    x = tf.keras.layers.Dense(1000, activation='relu', name="dense1", kernel_regularizer=l2(l2value))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu', name="dense2", kernel_regularizer=l2(l2value))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64, activation='relu', name="dense3", kernel_regularizer=l2(l2value))(x)

    output = Dense(out_length, name='output', kernel_regularizer=l2(l2value))(x)

    model = Model(inputs=main_input, outputs=output)
    adam = Adam(learning_rate=0.0015)
    model.compile(optimizer=adam, loss=multilabel_categorical_crossentropy, metrics=['accuracy'])

    model.summary()

    return model

def model_base(length, out_length, para):
    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='int64', name='main_input')
    x = Embedding(output_dim=ed, input_dim=21, input_length=length, name='Embadding')(main_input)

    a = Convolution1D(64, 2, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    apool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(a)

    b = Convolution1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    bpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(b)

    c = Convolution1D(64, 8, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    cpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(c)

    merge = Concatenate(axis=-1)([apool, bpool, cpool])
    merge = Dropout(dp)(merge)

    x = Bidirectional(LSTM(100, return_sequences=True))(merge)

    x = MultiHeadAttention(80, 5)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(l2value))(x)

    output = Dense(out_length, activation='sigmoid',name='output', kernel_regularizer=l2(l2value))(x)

    model = Model(inputs=main_input, outputs=output)
    adam = Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model
