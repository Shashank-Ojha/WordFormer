# WordFormer
Implements a GPT architecture to predict next letter in word.

We test this specifically by training and testing on Shakespeare text. After
training, this model is able to generate text that resembles Shakespeare.

## How to run
Just type `jupyter notebook` into your terminal and open the link it gives you.
After that open the file `trainer.ipynb` and run the blocks. 

## Results
After training the model for about 15000 iters on T4 GPUs through Google Colab,
we get the train and validation losses of:

```
step 0: train loss 1.9695, val loss 2.0415
step 1000: train loss 1.9431, val loss 2.0218
step 2000: train loss 1.9187, val loss 2.0036
step 3000: train loss 1.8996, val loss 1.9954
step 4000: train loss 1.8816, val loss 1.9857
step 4999: train loss 1.8646, val loss 1.9732
```

If we generate from the model at this point, we get:

```

NESINGM:
Be'Tink his had, mater thatiless
That yhint, by when and inter of thou it with'ds?

KING ORCHARWICHINT:
How to has unciment?
ISal douch old dry,
Whath and netencoound we irou,
I could
And the creasmberss jurrourb
To artt, yoplay'd slomas ds-
you, worsess bellod,
And I friece, Papurght
Go-sard in'd Dothe's misecal or and goood,
And that the
Gh 'ts boooot botieloh thy is dishse wil-morgene horgue.

STEirintil.

CLINUR ONGRUET:
I powans at theurs; his thempser; lovelod.
Ten i't will then m
```

The model is starting to look like Shakespeare, but it is still not there yet.
There's a lot of words that are just gibberish.

To improve these results, we would need to scale up the hyperparameters and
train for longer. However, this requires more time and/or better GPUs, so we
leave it here for now.


# References
This toy project was heavily inspired by Andrej Karpathy's course:
https://www.youtube.com/watch?v=kCc8FmEb1nY&ab_channel=AndrejKarpathy

Also I'd like to cite:

Attention Is All You Need Paper: https://arxiv.org/abs/1706.03762

Self Attention Blog Post: https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html
