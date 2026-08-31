# So… What Can 128GB Actually Do?

A long story again. Maybe not as long as the last one. But this is the part nobody told me when I started.

When I received the DGX Spark, my first thought was simple. Finally. No more experimenting around memory limits. No more reducing context length. No more calculating whether a model will fit before even downloading it. I had 128GB of unified memory sitting on my desk.

Like most people, my first instinct was to think bigger. Maybe now I can train larger models. Maybe now I can fine-tune everything. Maybe now I can run models that were impossible before.

But something unexpected happened. For almost the first few weeks, I didn't train anything. Instead, I found myself staring at monitoring tools, memory graphs, Hugging Face model pages, quantization formats, and a question that sounded embarrassingly basic:

**Where is all this memory actually going?**

Because the moment you start looking at modern AI systems closely, nothing behaves quite the way you expect.

## The first assumption

I assumed model size was easy. A 70 billion parameter model should take around 70GB. A 32 billion parameter model should take around 32GB. Simple.

Well, not quite.

I was forgetting about precision.

In FP16 or BF16, each parameter takes roughly 2 bytes. So a 70 billion parameter model is already around 140GB just for the weights. A 32 billion parameter model is around 64GB.

Then I started looking at Hugging Face and seeing the same model available in multiple formats. FP16, BF16, GPTQ, AWQ, GGUF, EXL2. Same model name, same parameter count, but the required memory could be completely different.

A model that needed well over 100GB in one format could suddenly become a 35GB model in another.

At first it felt confusing.

Then I realized I wasn't really learning models.

**I was learning inference.**

## Hugging Face changed the way I look at models

Before local AI, I mostly thought about models in terms of capability — which model is smarter, which one scores higher.

Once you start running models locally, a different set of questions shows up.

Will it fit?

What precision is it using?

How much context can it handle?

How much memory will remain after loading?

Suddenly the model itself is only one part of the equation. The infrastructure around it matters just as much.

A good AI engineer cannot think only about models. They also need to think about systems.

## The day quantization finally made sense

For a long time, quantization felt like one of those AI words everyone uses but nobody properly explains.

Then I started looking at it differently.

Imagine taking a RAW image from a professional camera and converting it into a compressed JPEG. The image becomes smaller, some information is lost, but for most people it still looks perfectly usable.

Quantization is somewhat similar.

The neural network weights get stored at lower precision. The model becomes smaller, memory requirements go down, and inference **can** become faster depending on the hardware and inference engine. The challenge is making sure the quality doesn't drop too much.

And suddenly, that "128GB machine" started feeling much bigger than it did on paper.

## Then I met the real memory consumer

Model weights are only part of the story.

The part nobody talks about enough is **KV Cache**.

I kept running models and noticed something strange.

The model would load fine. Memory usage looked normal.

Then I would start chatting, and memory kept climbing.

More questions. More tokens. More context. More memory.

The model was building and holding on to a KV Cache — storing the key and value states from previous tokens so they don't have to be recomputed during subsequent generation.

The longer the conversation gets, the bigger the cache grows.

A model that comfortably fits in memory can still end up using a lot more once you actually start using it.

This was probably the biggest surprise for me.

Not because it was hidden, but because nobody mentions it when they talk about model sizes.

## Then I started watching everything

Instead of training models, I started observing them.

How much memory belongs to the model?

How much to KV Cache?

What happens when context grows from 8K to 32K?

Why do two inference engines give different throughput on the same model?

Why is GPU utilization sometimes high and sometimes surprisingly low?

The deeper I went, the more interesting it got.

AI isn't only about neural networks.

It's also about scheduling, memory management, caching, batching, GPU utilization — all the engineering that has to work together underneath.

## Why I didn't start with training

A lot of people buy powerful hardware and jump straight into fine-tuning.

I almost did the same.

But then I stopped myself.

Training felt like the final chapter.

Inference felt like the beginning.

Before changing a model, I wanted to understand how it behaves. Before fine-tuning anything, I wanted to know where every gigabyte was disappearing.

**Training shapes the model. Inference is where the model actually delivers something.**

## What 128GB actually gave me

The biggest benefit wasn't running a bigger model.

It was **freedom**.

Freedom to observe, to experiment, to understand the system without constantly fighting memory limits.

On smaller machines, every experiment starts with a compromise.

Smaller model.

Shorter context.

Lower precision.

Smaller batches.

With 128GB, a lot of those trade-offs move further away.

For the first time, I could focus on understanding how AI systems actually behave, instead of just trying to make them fit.

And oddly enough, that has turned out to be far more valuable than training a model.

## What's next

Quantization led me to model sizing.

Model sizing led me to memory consumption.

Memory consumption led me to KV Cache.

And KV Cache is now leading me towards inference engines, batching strategies, and throughput optimization.

So before I touch LoRA, fine-tuning, or model training, I want to spend some more time understanding how modern inference systems actually work.

Long story long — I bought a 128GB AI workstation expecting to learn training.

Instead, it pushed me towards understanding inference first.

And honestly, **that has turned out to be the more interesting journey.**
