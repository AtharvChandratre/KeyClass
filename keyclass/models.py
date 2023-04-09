import logging
from typing import List, Iterable, Union, Optional
import utils
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import sentence_transformers.util
import torch
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model.baselines import MajorityLabelVoter
from snorkel.labeling.model.label_model import LabelModel
from sentence_transformers import SentenceTransformer
from tqdm.autonotebook import trange

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
class CustomEncoder(torch.nn.Module):

    def __init__(self,
                 pretrained_model_name_or_path:
                 str = 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12',
                 device: str = "cuda"):
        super(CustomEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path)
        logging = "Creating Model"
        logger.log(logging)
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path)
        logging = "Beginning Training"
        logger.log(logging)
        self.model.train()
        logging = "Training completed"
        logger.log(logging)
        self.device = device
        self.to(device)

    def encode(self,
               sentences: Union[str, List[str]],
               batch_size: int = 32,
               show_progress_bar: Optional[bool] = False,
               normalize_embeddings: bool = False):
        self.model.eval()  # Set model in evaluation mode.
        with torch.no_grad():
            forward = self.forward(sentences,
                                   batch_size=batch_size,
                                   show_progress_bar=show_progress_bar,
                                   normalize_embeddings=normalize_embeddings
                                   )
            detached = forward.detach()
            returned_tensor = detached.cpu()
            embeddings = returned_tensor.numpy()

        self.model.train()
        return embeddings

    def forward(self,
                sentences: Union[str, List[str]],
                batch_size: int = 32,
                show_progress_bar: Optional[bool] = None,
                normalize_embeddings: bool = False):

        all_embeddings = []

        length_sorted_idx = np.argsort(
            [-utils._text_length(sen) for sen in sentences])
        # length_sorted_idx = np.argsort([-self.model._text_length(sen) for sen in sentences])

        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0,
                                  len(sentences),
                                  batch_size,
                                  desc="Batches",
                                  disable=not show_progress_bar):
            # for start_index in range(0, len(sentences), batch_size):
            sentences_batch = sentences_sorted[start_index:start_index +
                                                           batch_size]

            features = self.tokenizer(sentences_batch,
                                      return_tensors='pt',
                                      truncation=True,
                                      max_length=512,
                                      padding=True)
            features = features.to(self.device)
            out_features = self.model.forward(**features)
            embeddings = utils.mean_pooling(out_features,
                                            features['attention_mask'])

            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.extend(embeddings)

        return torch.stack([all_embeddings[idx] for idx in np.argsort(length_sorted_idx)])

class Encoder(torch.nn.Module):

    def __init__(self,
                 model_name: str = 'all-mpnet-base-v2',
                 device: str = "cuda"):
        super(Encoder, self).__init__()

        self.model_name = model_name
        self.model = SentenceTransformer(model_name_or_path=model_name,
                                         device=device)
        self.device = device
        self.to(device)

    def encode(self,
               sentences: Union[str, List[str]],
               batch_size: int = 32,
               show_progress_bar: Optional[bool] = False,
               normalize_embeddings: bool = False):
        self.model.eval()  # Set model in evaluation mode.
        with torch.no_grad():
            forward = self.forward(sentences,
                                   batch_size=batch_size,
                                   show_progress_bar=show_progress_bar,
                                   normalize_embeddings=normalize_embeddings
                                   )
            detached = forward.detach()
            returned_tensor = detached.cpu()
            embeddings = returned_tensor.numpy()
        self.model.train()
        return embeddings

    def forward(self,
                sentences: Union[str, List[str]],
                batch_size: int = 32,
                show_progress_bar: Optional[bool] = False,
                normalize_embeddings: bool = False):

        all_embeddings = []
        sentences_sorted = [sentences[idx] for idx in np.argsort([-utils._text_length(sen) for sen in sentences])]

        for start_index in trange(0,
                                  len(sentences),
                                  batch_size,
                                  desc="Batches",
                                  disable=not show_progress_bar):
            features = sentence_transformers.util.batch_to_device(
                self.model.tokenize(sentences_sorted[start_index:start_index + batch_size]), self.device)
            out_features = self.model.forward(features)
            embeddings = out_features['sentence_embedding']
            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.extend(embeddings)

        length_sorted_idx = np.argsort([-utils._text_length(sen) for sen in sentences])
        all_embeddings = [
            all_embeddings[idx] for idx in np.argsort(length_sorted_idx)
        ]
        all_embeddings = torch.stack(all_embeddings)  # Converts to tensor

        return all_embeddings


class FeedForwardFlexible(torch.nn.Module):

    def __init__(self,
                 encoder_model: torch.nn.Module,
                 h_sizes: Iterable[int] = [768, 256, 64, 2],
                 activation: torch.nn.Module = torch.nn.LeakyReLU(),
                 device: str = "cuda"):
        super(FeedForwardFlexible, self).__init__()

        self.encoder_model = encoder_model
        self.device = device
        self.layers = torch.nn.ModuleList()
        len_h_sizes = len(h_sizes)
        for k in range(len_h_sizes - 1):
            self.layers.append(torch.nn.Linear(h_sizes[k], h_sizes[k + 1]))
            self.layers.append(activation)
            self.layers.append(torch.nn.Dropout(p=0.5))

        self.to(device)

    def forward(self, x, mode='inference', raw_text=True):
        if raw_text:
            x = self.encoder_model.forward(x)

        for layer in self.layers:
            x = layer(x)

        if mode == 'inference':
            x = torch.nn.Softmax(dim=-1)(x)
        elif mode == 'self_train':
            x = torch.nn.LogSoftmax(dim=-1)(x)

        return x

    def predict(self, x_test, batch_size=128, raw_text=True):
        return np.argmax(self.predict_proba(x_test,
                                            batch_size=batch_size,
                                            raw_text=raw_text), axis=1)

    def predict_proba(self, x_test, batch_size=128, raw_text=True):
        with torch.no_grad():
            self.eval()
            probs_list = []
            # for i in trange(0, N, batch_size, unit='batches'):
            for i in range(0, len(x_test), batch_size):
                test_batch = x_test[i:i + batch_size]
                if raw_text == False:
                    test_batch = test_batch.to(self.device)
                forward = self.forward(test_batch,
                                       mode='inference',
                                       raw_text=raw_text)
                output_tensor = forward.cpu()
                probs = output_tensor.numpy()
                probs_list.append(probs)
            self.train()
        return np.concatenate(probs_list, axis=0)


class LabelModelWrapper:

    def __init__(self,
                 label_matrix,
                 y_train=None,
                 n_classes=2,
                 device='cuda',
                 model_name='data_programming'):
        if not isinstance(label_matrix, pd.DataFrame):
            raise ValueError(f'label_matrix must be a DataFrame.')

        _VALID_LABEL_MODELS = ['data_programming', 'majority_vote']
        if model_name not in _VALID_LABEL_MODELS:
            raise ValueError(
                f'model_name must be one of {_VALID_LABEL_MODELS} but passed {model_name}.'
            )

        self.label_matrix = label_matrix.to_numpy()
        self.y_train = y_train
        self.n_classes = n_classes
        self.LF_names = list(label_matrix.columns)
        self.learned_weights = None  # learned weights of the labeling functions
        self.trained = False  # The label model is not trained yet
        self.device = device
        self.model_name = model_name

    def display_LF_summary_stats(self):
        df_LFAnalysis = LFAnalysis(L=self.label_matrix).lf_summary(Y=self.y_train, est_weights=self.learned_weights)
        df_LFAnalysis.index = self.LF_names
        return df_LFAnalysis

    def train_label_model(self,
                          n_epochs=500,
                          class_balance=None,
                          log_freq=100,
                          lr=0.01,
                          seed=13,
                          cuda=False):
        print(f'==== Training the label model ====')
        if self.model_name == 'data_programming':
            self.label_model = LabelModel(cardinality=self.n_classes,
                                          device=self.device)
            if cuda == True:
                self.label_model = self.label_model.cuda()
            self.label_model.fit(self.label_matrix,
                                 n_epochs=n_epochs,
                                 class_balance=class_balance,
                                 log_freq=log_freq,
                                 lr=lr,
                                 seed=seed,
                                 optimizer='sgd')
            self.trained = True
            self.learned_weights = self.label_model.get_weights()
        elif self.model_name == 'majority_vote':
            self.label_model = MajorityLabelVoter(cardinality=self.n_classes)
            self.trained = True

    def predict_proba(self):
        if not self.trained:
            print(
                "Model must be trained before predicting probabilistic labels")
            return

        y_proba = pd.DataFrame(
            self.label_model.predict_proba(L=self.label_matrix),
            columns=[f'Class {i}' for i in range(self.n_classes)])
        return y_proba

    def predict(self, tie_break_policy='random'):
        if not self.trained:
            print("Model must be trained before predicting labels")
            return 0

        y_pred = self.label_model.predict(L=self.label_matrix,
                                          tie_break_policy=tie_break_policy)
        return y_pred
