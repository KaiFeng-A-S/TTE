import json
import copy
import tensorflow as tf
import math

class TransEnConfig(object):
    
    def __init__(self, 
                vocab_size,
                hidden_size = 768,
                num_hidden_layers = 12,
                num_attention_heads = 12,
                intermediate_size = 3072,
                hidden_act = 'gelu',
                hidden_dropout_prob = 0.1,
                attention_probs_dropout_prob = 0.1,
                max_position_embeddings = 512,
                type_vocab_size = 16,
                initializer_rang = 0.02):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_rang = initializer_rang
    
    @classmethod
    def from_dict(cls, json_object):
        config = TransConfig(vocab_size = None)
        for (key, value) in json_object.items():
            config.__dict__[key] = value
        
        return config
    
    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, 'r') as reader:
            text = reader.read()

        return cls.from_dict(json.loads(text))
    
    def to_dict(self):
        output = copy.deepcopy(self.__dict__)

        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent = 2, sort_keys = True) + '\n'
    
class TransEnModel(object):

    def __init__(self,
                config,
                is_training,
                input_ids,
                input_mask = None,
                token_type_ids = None,
                use_one_hot_embeddings = False,
                scope = None,
                reuse = None):
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0
        
        input_shape = get_shape_list(input_ids)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        
        print('input shape is: ', input_shape)

        if input_mask is None:
            input_mask = tf.ones(shape = [batch_size, seq_length], dtype = tf.int32)
        
        if token_type_ids is None:
            token_type_ids = tf.zeros(shape = [batch_size, seq_length], dtype = tf.int32)
        
        with tf.variable_scope(scope, reuse = reuse):
            with tf.variable_scope('embeddings'):
                (self.embedding_output, self.embedding_table) = embedding_lookup(
                    input_ids = input_ids,
                    vocab_size = config.vocab_size,
                    embedding_size = config.hidden_size,
                    initializer_rang = config.initializer_rang,
                    name = 'word_embeddings',
                    use_one_hot_embeddings = use_one_hot_embeddings)

                self.embedding_output = embedding_postprocessor(
                    input_tensor = self.embedding_output,
                    use_token_type = True,
                    token_type_ids = token_type_ids,
                    token_type_vocab_size = config.type_vocab_size,
                    token_type_embedding_name = 'token_type_embeddings',
                    use_position_embeddings = True,
                    position_embedding_name = 'position_embeddings',
                    initializer_rang = config.initializer_rang,
                    max_position_embeddings = config.max_position_embeddings,
                    dropout_prob = config.hidden_dropout_prob)
        
            with tf.variable_scope('encoder', reuse = reuse):
                attention_mask = create_attention_mask(input_ids, input_mask)

                self.all_encoder_layers = transformer_model(
                    input_tensor = self.embedding_output,
                    attention_mask = attention_mask,
                    hidden_size = config.hidden_size,
                    num_hidden_layers = config.num_hidden_layers,
                    num_attention_heads = config.num_attention_heads,
                    intermediate_size = config.intermediate_size,
                    intermediate_act_fn = get_activation(config.hidden_act),
                    hidden_dropout_prob = config.hidden_dropout_prob,
                    attention_probs_dropout_prob = config.attention_probs_dropout_prob,
                    initializer_rang = config.initializer_range,
                    do_return_all_layers = True)

def gelu(input_tensor):
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / math.sqrt(2.0)))

    return input_tensor * cdf

def get_activation(name):
    act = activation.lower()
    if act == 'linear':
        return None
    elif act == 'relu':
        return tf.nn.relu
    elif act == 'gelu':
        return gelu
    elif act == "tanh":
        return tf.tanh

def get_shape_list(tensor):
    shape = tensor.shape.as_list()
    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)
    
    if not non_static_indexes:
        return shape
    
    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]

    return shape

def embedding_output(input_ids,
                    vocab_size,
                    embedding_size = 128,
                    initializer_rang = 0.02,
                    name = 'word_embeddings',
                    use_one_hot_embeddings = False):
    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis = [-1])
    
    embedding_table = tf.get_variable(
        name = name,
        shape = [vocab_size, embedding_size],
        initializer = create_initializer(initializer_rang))
    
    if use_one_hot_embeddings:
        flat_input_ids = tf.reshape(input_ids, [-1])
        one_hot_input_ids = tf.one_hot(flat_input_ids, depth = vocab_size)
        output = tf.matmul(one_hot_input_ids, embedding_table)
    else:
        output = tf.nn.embedding_loopup(embedding_table, input_ids)
    
    input_shape = get_shape_list(input_ids)

    output = tf.reshape(output, input_shape[0: - 1] + [input_shape[-1] * embedding_size])

    return (output, embedding_table)

def create_initializer(initializer_rang = 0.02):
    return tf.truncated_normal_initializer(stddev = initializer_rang)

def embedding_postprocessor(input_tensor,
                            use_token_type = False,
                            token_type_ids = None,
                            token_type_vocab_size = 16,
                            token_type_embedding_name = 'token_type_embeddings',
                            use_position_embeddings = True,
                            position_embedding_name = 'position_embeddings',
                            initializer_rang = 0.02,
                            max_position_embeddings = 512,
                            dropout_prob = 0.1):
    input_shape = get_shape_list(input_tensor)
    batct_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    output = input_tensor

    if use_token_type:
        token_type_table = tf.get_variable(
            name = token_type_embedding_name,
            shape = [token_type_vocab_size, width],
            initializer_rang = create_initializer(initializer_rang))
        
        flat_token_type_ids = tf.reshape(token_type_ids, [-1])
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth = token_type_vocab_size)
        token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
        token_type_embeddings = tf.reshape(token_type_embeddings, input_shape)
        output += token_type_embeddings
    
    if use_position_embeddings:
        assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
        with tf.control_dependencies([assert_op]):
            full_position_embeddings = tf.get_variable(
                name = position_embedding_name,
                shape = [max_position_embeddings, width],
                initializer_rang = create_initializer(initializer_rang))
            position_embeddings = tf.slice(full_position_embeddings, [0, 0], [seq_length, -1])
            num_dims = len(output.shape.as_list())
            position_broadcast_shape = []
            for _ in range(num_dims - 2):
                position_broadcast_shape.append(1)
            position_broadcast_shape.extend([seq_length, width])
            position_embeddings = tf.reshape(position_embeddings, position_broadcast_shape)
            output += position_embeddings
    
    output = layer_norm_and_dropout(output, dropout_prob)

    return output

    def layer_norm_and_dropout(input_tensor, dropout_prob):
        output_tensor = layer_norm(input_shape)
        output_tensor = dropout(output_tensor, dropout_prob)

        return output_tensor

    def layer_norm(input_tensor):
        return tf.contrib.layers.layer_norm(
            inputs = input_tensor, begin_norm_axis = -1, begin_params_axis = -1)
    
    def dropout(input_tensor, dropout_prob):
        if dropout_prob is None or dropout_prob == 0.0:
            return input_tensor

        output_tensor = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)

def create_attention_mask(ids, mask):
    ids_shape = get_shape_list(ids)
    batch_size = ids_shape[0]
    ids_seq_length = ids_shape[1]

    mask_shape = get_shape_list(mask)
    mask_seq_length = mask_shape[1]

    mask = tf.cast(tf.reshape(mask, [batch_size, 1, mask_seq_length]), tf.float32)
    broadcast_ones = tf.ones(shape = [batch_size, ids_seq_length, 1], dtype = tf.float32)
    mask = broadcast_ones * mask

    return mask

def transformer_model(input_tensor,
                    attention_mask = None,
                    hidden_size = 768,
                    num_hidden_layers = 12,
                    num_attention_heads = 12,
                    intermediate_size = 3072,
                    intermediate_act_fn = gelu,
                    hidden_dropout_prob = 0.1,
                    attention_probs_dropout_prob = 0.1,
                    initializer_range = 0.02,
                    do_return_all_layers = False):
    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape = get_shape_list(input_tensor)

    prev_ouput = reshape_to_matrix(input_tensor)

    all_layer_outputs = []
    for layer_idx in range(num_hidden_layers):
        with tf.variable_scope('layer_%d' % layer_idx):
            layer_input = prev_ouput

            with tf.variable_scope('attention'):
                with tf.variable_scope('self'):
                    attention_head = attention_layer(
                        from_tensor = layer_input,
                        to_tensor = layer_input,
                        attention_mask = attention_mask,
                        num_attention_heads = num_attention_heads,
                        size_per_head = size_per_head,
                        attention_probs_dropout_prob = attention_probs_dropout_prob,
                        initializer_range = initializer_range,
                        do_return_2d_tensor = True)
                
                with tf.variable_scope('output'):
                    attention_output = tf.layer.dense(
                        attention_head,
                        hidden_size,
                        kernel_initializer = create_initializer(initializer_range))
                    attention_output = dropout(attention_output, hidden_dropout_prob)
                    attention_output = layer_norm(attention_output + layer_input)
            
            with tf.variable_scope('intermediate'):
                intermediate_output = tf.layers.dense(
                    attention_output,
                    intermediate_size,
                    activation = intermediate_act_fn,
                    kernel_initializer = create_initializer(initializer_range))
            
            with tf.variable_scope('output'):
                layer_output = tf.layers.dense(
                    intermediate_output,
                    hidden_size,
                    kernel_initializer = create_initializer(initializer_range))
                layer_output = dropout(layer_output, hidden_dropout_prob)
                layer_output = layer_norm(layer_output + attention_output)
                prev_ouput = layer_output
                all_layer_outputs.append(layer_output)
    
    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = tf.reshape(layer_output, input_shape)
            final_outputs.append(final_output)
        
        return final_outputs
    else:
        final_output = tf.reshape(prev_ouput, input_shape)
        return final_output

def reshape_to_matrix(input_tensor):
    ndims = input_tensor.shape.ndims
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])

    return output_tensor

def attention_layer(from_tensor,
                        to_tensor,
                        attention_mask = None,
                        num_attention_heads = 1,
                        size_per_head = 512,
                        query_act = None,
                        key_act = None,
                        value_act = None,
                        attention_probs_dropout_prob = 0.0,
                        initializer_range = 0.02,
                        do_return_2d_tensor = False):
        
        def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
            output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])

            output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    
            return output_tensor
        
        from_shape = get_shape_list(from_tensor)
        to_shape = get_shape_list(to_tensor)

        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]

        from_tensor_2d = reshape_to_matrix(from_tensor)
        to_tensor_2d = reshape_to_matrix(from_tensor)

        query_layer = tf.layers.dense(
            from_tensor_2d, 
            num_attention_heads * size_per_head,
            activation = query_act,
            name = 'query',
            kernel_initializer = create_initializer(initializer_range))
        
        key_layer = tf.layers.dense(
            to_tensor_2d,
            num_attention_heads * size_per_head,
            activation = key_act,
            name = 'key',
            kernel_initializer = create_initializer(initializer_range))
        
        value_layer = tf.layers.dense(
            to_tensor_2d,
            num_attention_heads * size_per_head,
            activation = value_act,
            name = 'value',
            kernel_initializer = create_initializer(initializer_range))

        query_layer = transpose_for_scores(query_layer, batch_size, 
                                    num_attention_heads, from_seq_length, size_per_head)

        key_layer = transpose_for_scores(key_layer, batch_size, 
                                    num_attention_heads, to_seq_length, size_per_head)

        attention_scores = tf.matmul(query_layer, key_layer, transpose_b = True)
        attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(size_per_head)))

        if attention_mask is not None:
            attention_mask = tf.expand_dims(attention_mask, axis = [1])
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -1e9
            attention_scores += adder
        
        attention_probs = tf.nn.softmax(attention_scores)
        attention_probs = dropout(attention_mask, attention_probs_dropout_prob)

        value_layer = transpose_for_scores(value_layer, batch_size,
                                    num_attention_heads, to_seq_length, size_per_head)
        
        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
        
        if do_return_2d_tensor:
            context_layer = tf.reshape(context_layer, 
                                        [batch_size * from_seq_length, num_attention_heads * size_per_head])
        else:
            context_layer = tf.reshape(context_layer,
                                        [batch_size, from_seq_length, num_attention_heads * size_per_head])

        return context_layer

    
