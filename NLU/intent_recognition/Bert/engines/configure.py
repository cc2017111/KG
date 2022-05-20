class Configure:
    def __init__(self, config_file='system.config'):
        config = self.config_file_to_dict(config_file)

        # Status
        the_item = 'mode'
        if the_item in config:
            self.mode = config[the_item]

        the_item = 'datasets_fold'
        if the_item in config:
            self.datasets_fold = config[the_item]

        the_item = "checkpoint_dir"
        if the_item in config:
            self.checkpoint_dir = config[the_item]

        the_item = 'checkpoint_name'
        if the_item in config:
            self.checkpoint_name = config[the_item]

        the_item = 'vocabs_dir'
        if the_item in config:
            self.vocabs_dir = config[the_item]

        the_item = 'log_dir'
        if the_item in config:
            self.log_dir = config[the_item]

        the_item = 'train_file'
        if the_item in config:
            self.train_file = config[the_item]

        the_item = 'token_level'
        if the_item in config:
            self.token_level = config[the_item]

        # word2vec config
        the_item = 'stop_word_file'
        if the_item in config:
            self.stop_word_file = self.str2none(config[the_item])

        the_item = 'w2v_train_data'
        if the_item in config:
            self.w2v_train_data = config[the_item]

        the_item = 'w2v_model_dir'
        if the_item in config:
            self.w2v_model_dir = config[the_item]

        the_item = 'w2v_model_name'
        if the_item in config:
            self.w2v_model_name = config[the_item]

        the_item = 'w2v_model_dim'
        if the_item in config:
            self.w2v_model_dim = int(config[the_item])

        the_item = 'w2v_min_count'
        if the_item in config:
            self.w2v_min_count = int(config[the_item])

        the_item = 'sg'
        if the_item in config:
            self.sg = config[the_item]

        the_item = 'batch_size'
        if the_item in config:
            self.batch_size = int(config[the_item])

        the_item = 'epoch'
        if the_item in config:
            self.epoch = int(config[the_item])

        the_item = 'max_sequence_length'
        if the_item in config:
            self.max_sequence_length = int(config[the_item])

        the_item = 'embedding_dim'
        if the_item in config:
            self.embedding_dim = int(config[the_item])

        the_item = 'train_file'
        if the_item in config:
            self.train_file = config[the_item]

        the_item = 'dev_file'
        if the_item in config:
            self.dev_file = self.str2none(config[the_item])
        else:
            self.dev_file = None

        the_item = 'max_to_keep'
        if the_item in config:
            self.max_to_keep = int(config[the_item])

        the_item = 'learning_rate'
        if the_item in config:
            self.learning_rate = float(config[the_item])

        the_item = 'optimizer'
        if the_item in config:
            self.optimizer = config[the_item]

        the_item = 'model'
        if the_item in config:
            self.model = config[the_item]

        the_item = 'bert_pretrain_path'
        if the_item in config:
            self.bert_pretrain_path = config[the_item]

        the_item = 'print_per_batch'
        if the_item in config:
            self.print_per_batch = int(config[the_item])

        the_item = 'measuring_metrics'
        if the_item in config:
            self.measuring_metrics = config[the_item]

        the_item = 'pb_model_sava_dir'
        if the_item in config:
            self.pb_model_sava_dir = config[the_item]

        the_item = 'is_early_stop'
        if the_item in config:
            self.is_early_stop = self.str2bool(config[the_item])

        the_item = 'patient'
        if the_item in config:
            self.patient = int(config[the_item])

        the_item = 'dropout_rate'
        if the_item in config:
            self.dropout_rate = float(config[the_item])

        # TextCNN model Configure
        the_item = 'num_filters'
        if the_item in config:
            self.num_filters = int(config[the_item])

        the_item = 'use_attention'
        if the_item in config:
            self.use_attention = self.str2bool(config[the_item])

        the_item = 'attention_size'
        if the_item in config:
            self.attention_size = int(config[the_item])

        the_item = 'embedding_method'
        if the_item in config:
            self.embedding_method = config[the_item]

        # TextRNN model configure
        the_item = 'hidden_dim'
        if the_item in config:
            self.hidden_dim = int(config[the_item])

        # Transformer model configure
        the_item = 'head_num'
        if the_item in config:
            self.head_num = int(config[the_item])

        the_item = 'encoder_num'
        if the_item in config:
            self.encoder_num = int(config[the_item])

    @staticmethod
    def config_file_to_dict(input_file):
        config = {}
        fins = open(input_file, mode='r', encoding='utf-8').readlines()
        for line in fins:
            if len(line) > 0 and line[0] == '#':
                continue
            if '=' in line:
                pair = line.strip().split('#', 1)[0].split('=', 1)
                item = pair[0]
                value = pair[1]
                try:
                    if item in config:
                        print('Warning: duplicated config item found: {}, update.'.format((pair[0])))
                    if value[0] == '[' and value[-1] == ']':
                        value_items = list(value[1:-1].split(','))
                        config[item] = value_items
                    else:
                        config[item] =  value

                except Exception:
                    print('configuration parsing error, please check correctness of the config file.')
                    exit(1)
        return config

    @staticmethod
    def str2none(string):
        if string == 'None' or string == 'none' or string == 'NONE':
            return None
        else:
            return string

    @staticmethod
    def str2bool(string):
        if string == 'True' or string == 'true' or string == 'TRUE':
            return True
        else:
            return False