from transformers import ElectraForPreTraining, ElectraTokenizerFast, EarlyStoppingCallback
from transformers.integrations import TensorBoardCallback
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer,\
    AutoFeatureExtractor
from utils import prepare_text_dataset, compute_metrics
import os


class RETrainer(Trainer):
    """
    custom Trainer
    -- optimizer is by default AdamW
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("label", None)
        if labels is None:
            labels = inputs.get("labels", None)
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        print('labels', labels)
        print('pred', logits)
        #print('inputs', inputs)

        loss_fct = nn.MSELoss()  # or we can use nn.L1Loss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1, self.model.config.num_labels))

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        pass


if __name__ == '__main__':
    # https://huggingface.co/Seznam/small-e-czech
    # https://huggingface.co/docs/transformers/training  ---> finetuning tutorial
    # https://towardsdatascience.com/how-to-fine-tune-an-nlp-regression-model-with-transformers-and-huggingface-94b2ed6f798f

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    tokenizer = ElectraTokenizerFast.from_pretrained("Seznam/small-e-czech")

    datasets = prepare_text_dataset('/home/fratrik/real_estate/data/dataset.csv', split_on='Kč',
                                    use_price_m2=True, use_gp_residual=True)

    datasets = datasets.rename_column('description', 'text')

    tokenized_datasets = datasets.map(lambda x: tokenizer(x["text"], max_length=512, padding="max_length",
                                                          truncation=True),
                                      batched=True)

    # num_labels=1 --> regression
    model = AutoModelForSequenceClassification.from_pretrained("Seznam/small-e-czech", num_labels=1).to("cuda")

    # to use as feature extractor
    #feature_extractor = AutoFeatureExtractor.from_pretrained("Seznam/small-e-czech")

    # https://stackoverflow.com/questions/69087044/early-stopping-in-bert-trainer-instances
    training_args = TrainingArguments(output_dir="test_trainer",
                                      logging_strategy="epoch",
                                      logging_dir='logs',
                                      evaluation_strategy="epoch",
                                      #eval_steps=200,
                                      #save_steps=200,
                                      learning_rate=2e-1,
                                      weight_decay=0.001,
                                      per_device_train_batch_size=32,
                                      per_device_eval_batch_size=32,
                                      num_train_epochs=100,
                                      save_total_limit=5,
                                      save_strategy='epoch',
                                      metric_for_best_model='mae',
                                      load_best_model_at_end=True
                                      )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10), TensorBoardCallback()]
    )

    trainer.train()

    print('trained')

    # TODO at model.eval() stage predictions are constant WTF
    # TODO try train on `price_m2`
    # TODO try train on residual price/price_m2 from XGB/GP/TargetEncoding
    # TODO use standardization !!!
