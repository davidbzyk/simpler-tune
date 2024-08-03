## Training a large language model for a specific use case: Simpler Trading Product Question and Answers
- note: I used a bleeding edge efficient in terms of speed and vram, process which uses custom Nvidia and Triton kernals so (Unsloth) so will not work with metal (Mac) at the moment.

## Notes / lessons learned / best practices before you start training
### `max_sequence_length` 
I trained locally using a 3090 and i9 14.9kf.  24gb reall isn't a lot so had to keep the `max_sequence_length` lower.  I initially did 512 on pretrain but ended up 800 on finetune.  They both worked fine.
- The longer the sequence, the more memory is required to store and process it. Memory consumption grows approximately linearly with sequence length.
- Transformers, in particular, have memory requirements that scale quadratically with the sequence length due to the self-attention mechanism. This means that doubling the sequence length can quadruple the memory usage.

### `per_device_train_batch_size` and `gradient_accumulation_steps` 
These also effect memory requirements significantly.  How it works?

The `per_device_train_batch_size` parameter specifies the number of training examples per device (GPU) in each training step. A smaller batch size can help reduce memory usage, making it possible to train larger models or fit more data into limited GPU memory. However, a smaller batch size can also lead to less stable training and noisier gradient estimates, potentially requiring more training steps to converge.

The `gradient_accumulation_steps` parameter allows you to accumulate gradients over multiple forward passes before performing a backward pass and an optimization step. This effectively increases the batch size without requiring additional memory for storing larger batches of data.
- Larger batch sizes tend to provide more stable gradient estimates, leading to smoother convergence. Try not to go as low as 1.  If you run into memory constraints just use a google collab.
- Smaller batch sizes with higher gradient accumulation steps can simulate this stability while managing memory constraints.
A rule of thumb I tend to use is multiples of 2.
Adjusting in multiples or divisors of 2 is common because it aligns well with binary computing systems, ensuring efficient memory allocation and usage. For example:
Batch Size:
If you reduce per_device_train_batch_size by a factor of 2 (e.g., from 8 to 4), you need to double gradient_accumulation_steps to maintain the same effective batch size.
Gradient Accumulation Steps:
Similarly, if you increase per_device_train_batch_size by a factor of 2 (e.g., from 4 to 8), you can halve gradient_accumulation_steps to keep the effective batch size constant.


### Setup Environment
- Install Miniconda for the virtual environment
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init
```
- Install python 3.10 and dependencies
```bash
conda create --name unsloth_env python=3.10 -y
conda activate unsloth_env
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install nvidia/label/cuda-12.1.0::cuda-toolkit -y
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
conda install -c conda-forge jupyter -y
conda install -c anaconda ipykernel -y
pip install wandb
```

### First step is to acquire and clean data (can skip ahead as finished results are in folder)
I find pretraining and finetuning are great combination when trying to train a model for a specific task.
For our example we used the Product Question and Answers in the Step-1-Data-Preproccessing --> raw-data folder.

I added two columns one called question and one called answer, I may have deleted a few before that, I can't recall, but in order to make it a repeatable pipeline where you "press play" and it cleans, trains, deploys there would need to be a set format so when changes are made for whatever reason the template is intact.  

I first used google sheets and basic formulas to determine what the text should look like and then used simple python string manipulation like so:
```python
import csv
import json
def process_csv_to_jsonl(input_csv, output_jsonl):
    def format_platforms(platforms):
        platforms_list = platforms.split('|')
        if len(platforms_list) == 1:
            return platforms
        elif len(platforms_list) == 2:
            return ' and '.join(platforms_list)
        else:
            return ', '.join(platforms_list[:-1]) + ', and ' + platforms_list[-1]

    print(f"Opening CSV file: {input_csv}")
    with open(input_csv, 'r', encoding='utf-8') as csv_file, open(output_jsonl, 'w', encoding='utf-8') as jsonl_file:
        csv_reader = csv.reader(csv_file)
        print("Skipping first 3 rows...")
        next(csv_reader)  # Skip the first 3 rows
        next(csv_reader)
        next(csv_reader)
        product_names = next(csv_reader)[4:]  # Get product names from row 4, starting from column E
        print(f"Found {len(product_names)} product names: {product_names[:5]}...")

        print("Searching for the platforms row...")
        for row_num, row in enumerate(csv_reader, start=5):
            print(f"Checking row {row_num}: {row[:5]}...")
            if row and len(row) > 2 and "What platforms can" in row[2]:
                print(f"Found platforms row: {row[:5]}...")
                for i, product in enumerate(product_names):
                    if i + 4 < len(row) and row[i+4]:  # Check if there's a value for this product
                        question = f"What platforms can {product} be used on?"
                        answer = f"{product} can be used on {format_platforms(row[i+4])}"
                        json_line = json.dumps({"question": question, "answer": answer})
                        jsonl_file.write(json_line + '\n')
                        print(f"Wrote entry for {product}")
                print("Finished processing platforms row")
                break  # We've found the row we need, no need to continue
        else:
            print("WARNING: Did not find a row containing platform information!")

    print("JSONL file creation process completed.")

```
There is a handful of steps in fine-tuning-data.ipynb, I left one out as for some reason I moved directories around and did not save.  feel free to adjust or enhance, just remember Python index starts at 0.  

- fine tuning data is first processed to the processed folder, then I manually look at it to determine what needs to be cleaned then run a cleaning function to the cleaned folder.  Then delete jsonl file in the processed folder after manually confirming nothing else is going on.

### Continued Pre training Data
Basically all I did here was concatenate all rows for each product that made sense.  Then did some cleaning for nans, etc.

