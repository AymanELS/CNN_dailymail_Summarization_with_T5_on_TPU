from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
from transformers import T5ForConditionalGeneration, AdamW, set_seed
from accelerate import Accelerator
from tqdm.notebook import tqdm
import datasets
import transformers
from accelerate import notebook_launcher
import torch

## load cnn dailymail dataset
train_dataset, val_dataset, test_dataset = load_dataset("cnn_dailymail", '3.0.0', split=['train[:10000]', 'validation[:2000]', 'test[:2000]'])
print(train_dataset.features)

## load T5 tokenizer
tokenizer = AutoTokenizer.from_pretrained('t5-base')

## set up prefix for summarization
prefix= "Summarize: "


## preprocessing
max_target_length = 64

def preprocessing(dataset):
  articles = dataset['article']
  summaries = dataset['highlights']
  # add prefix to articles
  inputs = [prefix + article for article in articles]
  #encode inputs and labels(summaries)
  tokenized_inputs = tokenizer(inputs, padding='max_length', max_length= 512, truncation=True)
  tokenized_summaries = tokenizer(summaries, padding='max_length', max_length=64, truncation=True).input_ids

  # replace index of padding tokens to -100 to be ignored by CrossEntropyLoss
  tokenized_summaries_ignore_index=[]
  for summary in tokenized_summaries:
    new_summary = [token if token!=0 else -100 for token in summary]
    tokenized_summaries_ignore_index.append(new_summary)

  tokenized_inputs['labels']= tokenized_summaries_ignore_index
  return tokenized_inputs

encoded_train_dataset= train_dataset.map(preprocessing, batched=True, remove_columns=train_dataset.column_names)
encoded_val_dataset = val_dataset.map(preprocessing, batched=True, remove_columns=val_dataset.column_names)
encoded_test_dataset = test_dataset.map(preprocessing, batched=True, remove_columns=test_dataset.column_names)
# print(encoded_train_dataset.features)
# print(tokenizer.decode(encoded_train_dataset[0]['input_ids']))
# print(tokenizer.decode([token for token in encoded_train_dataset[0]['labels'] if token!=-100]))

## change data type to torch tensors
encoded_train_dataset.set_format(type='torch')
encoded_val_dataset.set_format(type='torch')
encoded_test_dataset.set_format(type='torch')

## setup dataloader
train_loader = DataLoader(encoded_train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(encoded_val_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(encoded_test_dataset, batch_size=4, shuffle=True)

## init training variables
learning_rate= 0.0001
epochs= 50 # set to very high number
train_batch_size= 2 # Actual batch size will this x 8 (was 8 before but can cause OOM)
eval_batch_size= 2 # Actual batch size will this x 8 (was 32 before but can cause OOM)
seed = 4
patience = 3 # early stopping
output_dir = "/output/"
epochs_no_improvement = 0
min_val_loss = float('inf')

# define training function for TPUs
def train():
  accelerator = Accelerator()

  # To have only one message (and not 8) per logs of Transformers or Datasets, we set the logging verbosity
  # to INFO for the main process only.
  if accelerator.is_main_process:
      datasets.utils.logging.set_verbosity_warning()
      transformers.utils.logging.set_verbosity_info()
  else:
      datasets.utils.logging.set_verbosity_error()
      transformers.utils.logging.set_verbosity_error()
  set_seed(seed)
  model = T5ForConditionalGeneration.from_pretrained('t5-base')
  optimizer = AdamW(model.parameters(), lr=learning_rate)
  #prepare data with accelerator
  model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

  for epoch in epochs:
    progress_bar = tqdm(range(len(train_loader)), disable=not accelerator.is_main_process)
    progress_bar.set_description(f"Epoch: {epoch}")
    model.train()
    for batch in train_loader:
      outputs = model(**batch)
      loss = outputs.loss
      accelerator.backward(loss)
      optimizer.step()
      optimizer.zero_grad()
      progress_bar.set_postfix({'loss': loss.item()})
      progress_bar.update(1)
    # evaluation at end of epoch
    model.eval()
    val_loss=[]
    for batch in val_loader:
      with torch.no_grad():
        outputs = model(**batch)
      loss = outputs.loss
      # get loss values from all TPU cores
      val_loss.append(accelerator.gather(loss[None]))
    avg_val_loss = torch.stack(val_loss).sum().item() / len(val_loss)

    if avg_val_loss < min_val_loss:
      epochs_no_improvement= 0
      min_val_loss = avg_val_loss
      continue
    else:
      epochs_no_improvement+=1
      if epochs_no_improvement==patience:
        accelerator.print(f"Early Stopping due to no improvement for {patience} epochs")
        break
  # save model after training
  accelerator.wait_for_everyone()
  unwrapped_model = accelerator.unwrap_model(model)
  unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)


notebook_launcher(train)

#### RETURN ERROR DUE torch_XLA compatibility