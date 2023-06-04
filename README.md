# WordFormer
Implements transformer architecture to predict next letter in word

## How to run
Just type `jupyter notebook` into your terminal and open the link it gives you.
After that open the file `trainer.ipynb` and run the blocks. 

## Results
After training the model for about 15000 iters on T4 GPUs through Google Colab,
we get the train and validation losses of:

```
step <built-in function iter>: train loss 1.9695, val loss 2.0415
step <built-in function iter>: train loss 1.9431, val loss 2.0218
step <built-in function iter>: train loss 1.9187, val loss 2.0036
step <built-in function iter>: train loss 1.8996, val loss 1.9954
step <built-in function iter>: train loss 1.8816, val loss 1.9857
step <built-in function iter>: train loss 1.8646, val loss 1.9732
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