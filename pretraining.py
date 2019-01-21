import tensorflow as tf
import transformer_basic as tb
import os

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('transen_config_file', None, '')
flags.DEFINE_string('input_file', None, '')
flags.DEFINE_string('output_dir', None, '')
flags.DEFINE_string('init_checkpoint', None, '')
flags.DEFINE_integer('max_seq_length', 320, '')
flags.DEFINE_integer('train_batch_size', 1024, '')
flags.DEFINE_integer('eval_batch_size', 8, '')
flags.DEFINE_integer('num_train_steps', 100000, '')
flags.DEFINE_integer('num_warmup_steps', 10000, '')
flags.DEFINE_integer('save_checkpoints_steps', 1000, '')
flags.DEFINE_integer('worker_gpu', 4, '')
flags.DEFINE_bool('do_train', False, '')
flags.DEFINE_bool('do_eval', False, '')
flags.DEFINE_float('learning_rate', 5e-5, '')

def main(_):
    if not FLAGS.transen_config_file:
        transen_config = tb.TransEnConfig(
            vocab_size = 10000,
            hidden_size = 384,
            num_hidden_layers = 12,
            num_attention_heads = 12,
            intermediate_size = 1024,
            hidden_act = 'gelu',
            hidden_dropout_prob = 0.1,
            attention_probs_dropout_prob = 0.1,
            max_position_embeddings = 384,
            type_vocab_size = 2,
            initializer_rang = 0.02)
        with open(os.path.join(FLAGS.output_dir, 'transen_config.json'), 'w') as f:
            f.write(transen_config.to_json_string())

if __name__ == '__main__':
    flags.mark_flag_as_required('input_file')
    flags.mark_flag_as_required('output_dir')
    tf.app.run()
