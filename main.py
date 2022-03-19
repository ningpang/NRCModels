import os

from argparse import ArgumentParser
from config import Config
from REModels.BertMk_model import Bert_model
from REModels.BertCLS_model import Bert_CLS_model
from REModels.BertCNN_model import Bert_CNN_model
from REModels.BertLSTM_model import Bert_LSTM_model
from REModels.CNN_model import CNN_model
from REModels.LSTM_model import LSTM_model
from REDataloaders.Bert_dataloader import get_bert_loaders
from REDataloaders.Old_dataloader import get_old_loaders
import framework

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    parser = ArgumentParser(description='Reinforcement Learning')
    parser.add_argument('--config', default='config.ini')
    args = parser.parse_args()
    config = Config(args.config)
    config.batch_size_per_step = int(config.batch_size / config.gradient_accumulation_steps)

    config.model_type = ['Bert', 'BertCnn', 'BertCls', 'CNN', 'LSTM']

    model_name = 'BertLstm'

    if model_name == 'BertMk':
        trainloader, validloader, testloader = get_bert_loaders(config)
        model = Bert_model(config).to(config.device)
        best_acc = 0.0
        for i in range(config.total_round):
            framework.train_bert_model(config, model, trainloader, config.step1_epochs)
            cur_acc = framework.evaluate_bert_model(config, model, testloader)
            print(f'current test acc:{cur_acc}')
            if cur_acc > best_acc:
                best_acc = cur_acc
        print(f'Best test acc:{best_acc}')
    elif model_name == 'BertCls':
        trainloader, validloader, testloader = get_bert_loaders(config)
        model = Bert_CLS_model(config).to(config.device)
        best_acc = 0.0
        for i in range(config.total_round):
            framework.train_bert_model(config, model, trainloader, config.step1_epochs)
            cur_acc = framework.evaluate_bert_model(config, model, testloader)
            print(f'current test acc:{cur_acc}')
            if cur_acc > best_acc:
                best_acc = cur_acc
        print(f'Best test acc:{best_acc}')
    elif model_name == 'BertCnn':
        trainloader, validloader, testloader = get_bert_loaders(config)
        model = Bert_CNN_model(config).to(config.device)
        best_acc = 0.0
        for i in range(config.total_round):
            framework.train_bertcnn_model(config, model, trainloader, config.step1_epochs)
            cur_acc = framework.evaluate_bert_model(config, model, testloader)
            print(f'current test acc:{cur_acc}')
            if cur_acc > best_acc:
                best_acc = cur_acc
        print(f'Best test acc:{best_acc}')
    elif model_name == 'BertLstm':
        trainloader, validloader, testloader = get_bert_loaders(config)
        model = Bert_LSTM_model(config).to(config.device)
        best_acc = 0.0
        for i in range(config.total_round):
            framework.train_bertcnn_model(config, model, trainloader, config.step1_epochs)
            cur_acc = framework.evaluate_bert_model(config, model, testloader)
            print(f'current test acc:{cur_acc}')
            if cur_acc > best_acc:
                best_acc = cur_acc
        print(f'Best test acc:{best_acc}')
    elif model_name == 'CNN':
        trainloader, validloader, testloader, word2id, word_vec = get_old_loaders(config)
        model = CNN_model(config, word_vec, word2id).to(config.device)
        best_acc = 0.0
        for i in range(config.total_round):
            framework.train_cnn_model(config, model, trainloader, config.step1_epochs)
            cur_acc = framework.evaluate_cnn_model(config, model, testloader)
            print(f'current test acc:{cur_acc}')
            if cur_acc > best_acc:
                best_acc = cur_acc
        print(f'Best test acc:{best_acc}')
    elif model_name == 'LSTM':
        trainloader, validloader, testloader, word2id, word_vec = get_old_loaders(config)
        model = LSTM_model(config, word_vec, word2id).to(config.device)
        best_acc = 0.0
        for i in range(config.total_round):
            framework.train_lstm_model(config, model, trainloader, config.step1_epochs)
            cur_acc = framework.evaluate_lstm_model(config, model, testloader)
            print(f'current test acc:{cur_acc}')
            if cur_acc > best_acc:
                best_acc = cur_acc
        print(f'Best test acc:{best_acc}')
    else:
        print("[Error] Model type does not exists !")
        assert 0




