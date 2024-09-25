'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import random, numpy as np, argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask

import itertools

TQDM_DISABLE=False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Pretrain mode does not require updating BERT paramters.
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        # You will want to add layers here to perform the downstream tasks.
        ### TODO
        self.linear_sentiment = nn.Linear(config.hidden_size, 5)
        self.dropout_sentiment = nn.Dropout(config.hidden_dropout_prob)
        self.softmax_sentiment = nn.Softmax(dim = 1)

        self.dropout_paraphrase_1= nn.Dropout(config.hidden_dropout_prob)
        self.linear_paraphrase_1 = nn.Linear(config.hidden_size * 2, 512)
        self.cosine_similarity = nn.CosineSimilarity()

        self.dropout_paraphrase_2= nn.Dropout(config.hidden_dropout_prob)
        self.linear_paraphrase_2 = nn.Linear(512, 1)

        self.relu1 = nn.ReLU()

        self.dropout_similarity1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_similarity2 = nn.Dropout(config.hidden_dropout_prob)
        self.linear_similarity1 = nn.Linear(config.hidden_size, 10)
        self.linear_similarity2 = nn.Linear(config.hidden_size, 10)
=        self.relu_similarity3 = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        embeddings = self.bert.forward(input_ids, attention_mask)
        embeddings = embeddings['pooler_output']

        return embeddings


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        embeddings = self.forward(input_ids, attention_mask)
        logits = self.dropout_sentiment(embeddings)
        logits = self.linear_sentiment(logits)
        #logits = self.activation_sentiment(logits)
        #probabilities = self.softmax_sentiment(logits)

        return logits



    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        ### TODO
        embeddings1 = self.forward(input_ids_1, attention_mask_1)
        embeddings2 = self.forward(input_ids_2, attention_mask_2)

        concat_embeddings = torch.cat((embeddings1, embeddings2), dim = 1)

        logit1 = self.dropout_paraphrase_1(concat_embeddings)
        logit1 = self.linear_paraphrase_1(logit1)
        logit1 = self.relu1(logit1)
        self.relu_similarity1(logit1)

        logit_final = self.dropout_paraphrase_2(logit1)
        logit_final = self.linear_paraphrase_2(logit_final)
        #print('\n')
        #print(logit_final)
        #print('\n')

        return logit_final

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        embeddings1 = self.forward(input_ids_1, attention_mask_1)
        embeddings2 = self.forward(input_ids_2, attention_mask_2)

        logit1 = self.dropout_similarity1(embeddings1)
        logit1 = self.linear_similarity1(logit1)

        logit2 = self.dropout_similarity2(embeddings2)
        logit2 = self.linear_similarity2(logit2)

        cosine_similarity = self.cosine_similarity(logit1, logit2)
        logit = self.relu_similarity3(cosine_similarity)

        #return logit




def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

def nt_xent_loss(embeddings, temperature=0.1):
    """
    Calculate the NT-Xent loss for a batch of embeddings.

    embeddings: Tensor of shape (2 * actual_batch_size, embedding_dim), where
    the first half of embeddings correspond to the original embeddings, and the
    second half to the augmented embeddings (or vice versa).
    temperature: A scalar controlling the temperature of the softmax function.
    """
    actual_batch_size = embeddings.shape[0] // 2
    labels = torch.cat([torch.arange(actual_batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(embeddings.device)

    embeddings = torch.nn.functional.normalize(embeddings, dim=1)

    similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature

    # Mask to exclude self similarities
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(embeddings.device)
    labels.masked_fill_(mask, 0)

    # Compute the loss
    exp_similarities = torch.exp(similarity_matrix)
    exp_similarities.masked_fill_(mask, 0)

    # Sum of similarities for normalization
    sum_of_similarities = exp_similarities.sum(dim=1, keepdim=True)

    # Log-probabilities
    log_prob = similarity_matrix - torch.log(sum_of_similarities)

    # Compute mean of log-likelihood over positive pairs
    mean_log_prob_pos = (labels * log_prob).sum(dim=1) / labels.sum(dim=1)

    loss = -mean_log_prob_pos.mean()
    return loss

def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    #Loading datasets
    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)



    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                collate_fn=para_dev_data.collate_fn)


    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                collate_fn=sts_dev_data.collate_fn)



    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    print(args.epochs)
    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for sst_batch, para_batch, sts_batch in tqdm(zip(sst_train_dataloader, para_train_dataloader, sts_train_dataloader),  total=min([len(sst_train_dataloader), len(para_train_dataloader), len(sts_train_dataloader)]), desc=f'train-{epoch}', disable=TQDM_DISABLE):

            optimizer.zero_grad()
            #Sst
            sst_b_ids, sst_b_mask, sst_b_labels = (sst_batch['token_ids'],
                                      sst_batch['attention_mask'], sst_batch['labels'])

            sst_b_ids = sst_b_ids.to(device)
            sst_b_mask = sst_b_mask.to(device)
            sst_b_labels = sst_b_labels.to(device)

            sst_logits = model.predict_sentiment(sst_b_ids, sst_b_mask)
            loss = F.cross_entropy(sst_logits, sst_b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            train_loss += loss.item()
            num_batches += 1

            #Para
            para_b_ids1, para_b_mask1, para_b_ids2, para_b_mask2, para_b_labels = (para_batch['token_ids_1'],
                                      para_batch['attention_mask_1'], para_batch['token_ids_2'],
                                                                para_batch['attention_mask_2'], para_batch['labels'])

            para_b_ids1 = para_b_ids1.to(device)
            para_b_mask1 = para_b_mask1.to(device)
            para_b_ids2 = para_b_ids2.to(device)
            para_b_mask2 = para_b_mask2.to(device)
            para_b_labels = para_b_labels.to(device)

            #embeddings1 = model.forward(para_b_ids1, para_b_mask1)
            #embeddings2 = model.forward(para_b_ids2, para_b_mask2)
            #embeddings = torch.cat((embeddings1, embeddings2), dim=0)

            para_logits = model.predict_paraphrase(para_b_ids1, para_b_mask1, para_b_ids2, para_b_mask2)
            loss = F.binary_cross_entropy_with_logits(para_logits.squeeze(), para_b_labels.float(), reduction='sum') / args.batch_size
            #loss += nt_xent_loss(embeddings)
            loss.backward()
            train_loss += loss.item()
            num_batches += 1
            #Sts
            sts_b_ids1, sts_b_mask1, sts_b_ids2, sts_b_mask2, sts_b_labels = (sts_batch['token_ids_1'],
                                      sts_batch['attention_mask_1'], sts_batch['token_ids_2'], sts_batch['attention_mask_2'],
                                      sts_batch['labels'])

            sts_b_ids1 = sts_b_ids1.to(device)
            sts_b_mask1 = sts_b_mask1.to(device)
            sts_b_ids2 = sts_b_ids2.to(device)
            sts_b_mask2 = sts_b_mask2.to(device)
            sts_b_labels = sts_b_labels.to(device)

            #embeddings1 = model.forward(sts_b_ids1, sts_b_mask1)
            #embeddings2 = model.forward(sts_b_ids2, sts_b_mask2)
            #embeddings = torch.cat((embeddings1, embeddings2), dim=0)

            sts_logits = model.predict_similarity(sts_b_ids1, sts_b_mask1, sts_b_ids2, sts_b_mask2)
            #I multiplied by 5 because when checking sts_train csv file, the similarity scores were between 0 and 5. The cosin_similarity index
            #is between 0 and 1. So multipying by 5 will get the logits in the rquired range.

            sts_logits = sts_logits * 5
            sts_logits.requires_grad = True
            loss = F.mse_loss(sts_logits, sts_b_labels.view(-1).float(), reduction='sum') / args.batch_size
            #loss = F.cross_entropy(sts_logits, sts_b_labels.view(-1).float(), reduction='sum') / args.batch_size
            #loss = F.binary_cross_entropy_with_logits(sts_logits.squeeze(), sts_b_labels.float(), reduction='sum') / args.batch_size
            #loss += nt_xent_loss(embeddings)

            loss.backward()
            train_loss += loss.item()
            num_batches += 1

            optimizer.step()

        if num_batches != 0:
            train_loss = train_loss / (num_batches)

        sentiment_train_accuracy,sst_y_pred, sst_sent_ids, paraphrase_train_accuracy, para_y_pred, para_sent_ids, sts_train_corr, sts_y_pred, sts_sent_ids = model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, device)

        sentiment_dev_accuracy,sst_y_pred, sst_sent_ids, paraphrase_dev_accuracy, para_y_pred, para_sent_ids, sts_dev_corr, sts_y_pred, sts_sent_ids = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)


        average_train_accuracy = (sentiment_train_accuracy + paraphrase_train_accuracy + sts_train_corr) / 3
        average_dev_accuracy = (sentiment_dev_accuracy + paraphrase_dev_accuracy + sts_dev_corr) / 3
        #train_acc, train_f1, *_ = model_eval_sst(sst_train_dataloader, model, device)
        #dev_acc, dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)

        if (average_dev_accuracy >= best_dev_acc):
            best_dev_acc = average_dev_accuracy
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, Sst train acc :: {sentiment_train_accuracy :.3f}, Sst dev acc :: {sentiment_dev_accuracy :.3f}, Para train acc :: {paraphrase_train_accuracy :.3f}, Para dev acc :: {paraphrase_dev_accuracy :.3f}, Sts train corr :: {sts_train_corr :.3f}, Sts dev  corr :: {sts_dev_corr :.3f}")


def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train_multitask(args)
    test_multitask(args)
