from transformers import ElectraForPreTraining, ElectraTokenizerFast
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from utils import prepare_text_dataset

# https://huggingface.co/Seznam/small-e-czech
# https://huggingface.co/docs/transformers/training  ---> finetuning tutorial
# https://towardsdatascience.com/how-to-fine-tune-an-nlp-regression-model-with-transformers-and-huggingface-94b2ed6f798f

discriminator = ElectraForPreTraining.from_pretrained("Seznam/small-e-czech")
tokenizer = ElectraTokenizerFast.from_pretrained("Seznam/small-e-czech")

# num_labels=1 --> regression
model = AutoModelForSequenceClassification.from_pretrained("Seznam/small-e-czech", num_labels=1)

datasets = prepare_text_dataset('/home/fratrik/real_estate/data/dataset.csv')

tokenized_datasets = datasets.map(lambda x: tokenizer(x["description"], padding="max_length", truncation=True),
                                  batched=True)

training_args = TrainingArguments(output_dir="test_trainer",
                                  logging_strategy="epoch",
                                  evaluation_strategy="epoch",
                                  per_device_train_batch_size=8,
                                  per_device_eval_batch_size=8,
                                  num_train_epochs=3,
                                  save_total_limit = 2,
                                  save_strategy = 'no',
                                  load_best_model_at_end=False
                                  )


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"]
)

sentence = "Za hory, za doly, mé zlaté parohy"
fake_sentence = "Za hory, za doly, kočka zlaté parohy"

fake_sentence_tokens = ["[CLS]"] + tokenizer.tokenize(fake_sentence) + ["[SEP]"]
fake_inputs = tokenizer.encode(fake_sentence, return_tensors="pt")
outputs = discriminator(fake_inputs)
predictions = torch.nn.Sigmoid()(outputs[0]).cpu().detach().numpy()

for token in fake_sentence_tokens:
    print("{:>7s}".format(token), end="")
print()

for prediction in predictions.squeeze():
    print("{:7.1f}".format(prediction), end="")
print()
