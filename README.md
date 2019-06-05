# A Pipline Of Pretraining Bert On Google TPU

A tutorial of pertaining Bert on your own dataset using google TPU

## Introduction

Bert, which is also known as Bidirectional Encoder Representations from Transformers, is a powerful nerual network model presented by Google in 2018. There exist a bunch of pretrained models that can be fine tuned for the downstreaming tasks to achieve good performances. Though the pretrained model provided by Google is good enough, you may still want to tune the pretrained model provided by Google on your own domain-specific corpus for several additional iterations. That is, give our bert model a chance to be farimiliar with the jargons on your domain, and therefore we can expect better performance on the fine-tune process. 

Nevertheless, as I observed, such a tuning (pretraining) process really takes a long time even on a 1080Ti GPU. The batchsize is limited, and the loss decreases really slow. One promising way to solve this problem is to use TPU, which is provided in Google Cloud Platform. So, in this tutorial, We will go over a pipline of pretraining the Bert on TPU. 

## Prerequest
1. A Google account
2. A bank card (No worry! Google won't charge you any money! At least this time :P)
3. Your data

## Data preparation
Prepare your data as your are told at [here](https://github.com/google-research/bert#pre-training-with-bert). Now you should get a .txt file. This time, let's simply use the sample_text.txt, which can be downloaded from [here](https://github.com/google-research/bert.git). 

Download the pretrained bert model at [here](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip), make sure you unzip it. Now you get a folder named "multi_cased_L-12_H-768_A-12".

## Data upload
First, go to the [google cloud platform](https://cloud.google.com) and sign in. Create your project, and you should see this interface:

<p>
    <img src="image/1.png"/>
</p>

Then, click the storage button on the left bar:

<p>
    <img src="image/2.png"/>
</p>

Click Create bucket, then give it a name. For example the "sample_bucket_test". Make sure this name is not used by any other people. 

<p>
    <img src="image/3.png"/>
</p>

<p>
    <img src="image/4.png"/>
</p>

Ok! Now it's time to upload the data (sample_text.txt) and the pretrained model from Google (multi_cased_L-12_H-768_A-12) to the bucket!

<p>
    <img src="image/5.png"/>
</p>

Click the Upload folder button, and select the folder "multi_cased_L-12_H-768_A-12" to upload the pretrained model. Click the Upload files button, and select the file "sample_text.txt" to upload your data. Then you should get something like this:

<p>
    <img src="image/6.png"/>
</p>

## Create a VM & TPU
Click the button on the right top to open the console:

<p>
    <img src="image/7.png"/>
</p>

Here is something you should see:

<p>
    <img src="image/8.png"/>
</p>

Then, we are going to start the VM & TPU now! simply run the following code:

```
ctup up --name=test_tpu
```

However, if you want to use the newest tpu, you should tell google about this (Google! Give me your best V3-8 TPU!). But wait, the new GPU is more expensive (8.00$/hour). That's why I added the "--preemptible" in the following command. Basicly, it means that google can stop your training process whenever it wants. Nevertheless, it's much cheap: 2.40$/hour. This should not be a problem if your program save the model frequently. 

```
ctpu up --name=test-tpu --tpu-size=v3-8 --preemptible  
```

<p>
    <img src="image/9.png"/>
</p>

Press "y" and "Enter" to continue. It may take a while, so just wait. By the way, if you are asked to set a password about ssh, just set it. 

<p>
    <img src="image/10.png"/>
</p>

Now, you can run to check the status of your VM and TPU:

```
ctpu status
```

Here is something I got:

<p>
    <img src="image/11.png"/>
</p>

## Fetch bert program
Previously, we have got the model. Since we would like to train the model for an additional timesteps, we need to get the tensorflow code. Simply run:

```
git clone https://github.com/google-research/bert.git
```

You should see a folder named bert under your root directory:

<p>
    <img src="image/12.png"/>
</p>

We are almost there! Enter the folder "bert", and run the following code to generate the data in tensorflow style:

```
python create_pretraining_data.py \
  --input_file=gs://sample_bucket_test/sample_text.txt \
  --output_file=gs://sample_bucket_test/tmp/tf_examples.tfrecord \
  --vocab_file=gs://sample_bucket_test/multi_cased_L-12_H-768_A-12/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
```
Just like this:

<p>
    <img src="image/13.png"/>
</p>

Please, notice the usage of "gs://". It connects the google storage bucket with your virtual machine. I think this is actually the most valuable part of this tutorail... Anyway, the process should finish really quickly:

<p>
    <img src="image/14.png"/>
</p>

Now, we can train the model! Run the following code:

```
  python run_pretraining.py \
  --input_file=gs://sample_bucket_test/tmp/tf_examples.tfrecord \
  --output_dir=gs://sample_bucket_test/tmp/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=gs://sample_bucket_test/multi_cased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=gs://sample_bucket_test/multi_cased_L-12_H-768_A-12/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5 \
  --use_tpu=True \
  --tpu_name=test-tpu
```
Like this:

<p>
    <img src="image/15.png"/>
</p>

After a while, you can see the following result:

<p>
    <img src="image/16.png"/>
</p>

Bravo! You did it!



















