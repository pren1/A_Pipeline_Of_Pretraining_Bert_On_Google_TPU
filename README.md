# A Pipline Of Pretraining Bert On Google TPU

A tutorial of pertaining Bert on your own dataset using google TPU

#### Introduction

Bert, which is also known as Bidirectional Encoder Representations from Transformers, is a powerful nerual network model presented by Google in 2018. There exist a bunch of pretrained models that can be fine tuned for the downstreaming tasks to achieve good performances. Though the pretrained model provided by Google is good enough, you may still want to tune the pretrained model provided by Google on your own domain-specific corpus for several additional iterations. That is, give our bert model a chance to be farimiliar with the jargons on your domain, and therefore we can expect better performance on the fine-tune process. 

Nevertheless, as I observed, such a tuning (pretraining) process really takes a long time even on a 1080Ti GPU. The batchsize is limited, and the loss decreases really slow. One promising way to solve this problem is to use TPU, which is provided in Google Cloud Platform. So, in this tutorial, We will go over a pipline of pretraining the Bert on TPU. 

#### Prerequest
1. A Google account
2. A bank card (No worry! Google won't charge you any money! At least this time :P)
3. Your data

#### Data preparation
Prepare your data as your are told at [here](https://github.com/google-research/bert#pre-training-with-bert). Now you should get a .txt file. This time, let's simply use the sample_text.txt, which can be downloaded from [here](https://github.com/google-research/bert.git). 

Download the pretrained bert model at [here](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip), make sure you unzip it. Now you get a folder named "multi_cased_L-12_H-768_A-12".

#### Data upload
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
Click the Upload folder button, and select the folder "multi_cased_L-12_H-768_A-12" to upload the pretrained model. 
Click the Upload files button, and select the file "sample_text.txt" to upload your data. Then you should get something like this:
<p>
    <img src="image/6.png"/>
</p>

#### Create a VM & TPU
Click the button on the right top to open the console:
<p>
    <img src="image/7.png"/>
</p>

















