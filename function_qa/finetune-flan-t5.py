import torch
import argparse
import json
import torch.optim as optim
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torchmetrics.text.rouge import ROUGEScore
from data import data_helper
from pprint import pprint


def save_log(batch_idx, predictions, targets, rouge_scores):

    for key in rouge_scores.keys():
        rouge_scores[key] = rouge_scores[key].item()

    with open("log-T5/preds/prediction_{}".format(batch_idx), "w") as f:
        json.dump(predictions, f)

    with open("log-T5/targets/target_{}".format(batch_idx), "w") as f:
        json.dump(targets, f)

    with open("log-T5/metrics/rouge_score_{}".format(batch_idx), "w") as f:
        json.dump(rouge_scores, f)


def train_epoch(config, epoch, train_loader, val_loader, model, optimizer, metric, tokenizer):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        text_input_ids = batch['text_input_ids'].cuda()
        text_attention_mask = batch['text_attention_mask'].cuda()
        label_input_ids = batch['label_input_ids'].cuda()
        # label_attention_mask = batch['label_attention_mask'].cuda()

        output = model(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            labels=label_input_ids
        )

        loss = output.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % config.train_checkpoint == 0:
            print("Epoch: {:3d} | Batch: {:6d} | Loss: {:8.6f} ".format(epoch, batch_idx + 1, loss.item()))

        if (batch_idx + 1) % config.eval_checkpoint == 0:
            predictions, targets, rouge_score = evaluate(val_loader, model, metric, tokenizer)
            pprint(rouge_score)
            save_log(batch_idx, predictions, targets, rouge_score)
            model.train()


def evaluate(val_loader, model, metric, tokenizer):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            text_input_ids = batch['text_input_ids'].cuda()
            text_attention_mask = batch['text_attention_mask'].cuda()
            label_input_ids = batch['label_input_ids'].cuda()

            output = model.generate(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
                max_length=512,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0
            )

            prediction = [tokenizer.decode(o, skip_special_tokens=True, clean_up_tokenization_spaces=True) for o in
                          output]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in
                      label_input_ids]

            predictions.extend(prediction)
            targets.extend(target)

    rouge_score = metric(predictions, targets)

    return predictions, targets, rouge_score


def train(config):

    if config.seed is not None:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True

    device = torch.device('cuda:0')

    tokenizer = T5Tokenizer.from_pretrained(config.model_path)
    # model = T5ForConditionalGeneration.from_pretrained(config.model_config).to(device)
    model = T5ForConditionalGeneration.from_pretrained(config.model_path).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.init_lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, config.lr_decay_step, config.lr_decay_rate)
    metric = ROUGEScore()

    train_loader, val_loader = data_helper(config)

    for epoch in range(config.num_epochs):
        train_epoch(config, epoch, train_loader, val_loader, model, optimizer, metric, tokenizer)
        scheduler.step()
        torch.save(model.state_dict(), config.save_path + "/model_{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # default settings
    parser.add_argument('--save', type=str, default='MuP')
    parser.add_argument('--save_path', type=str, default='flan-t5-book')
    parser.add_argument('--dataset', type=str, default='kmfoda/booksum')
    parser.add_argument('--model_path', type=str, default="/path/to/model")
    # training
    parser.add_argument('--num_epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--init_lr', type=float, default=1e-5,
                        help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--train_checkpoint', type=int, default=100)
    parser.add_argument('--eval_checkpoint', type=int, default=1000)
    parser.add_argument('--dropout_p', type=float, default=0.5)
    parser.add_argument('--lr_decay_step', type=int, default=1)

    # pretrain model settings
    parser.add_argument('--max_length', type=int, default=512)

    # distributed training
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=42, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', type=bool, default=False,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    config = parser.parse_args()
    train(config)


