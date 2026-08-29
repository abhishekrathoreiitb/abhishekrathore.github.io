---
layout: post
title: "Know Your GPU as an Experiment: My DGX Spark Journey"
date: 2026-08-29
categories: [AI, Machine Learning, GPU]
tags: [dgx-spark, gpu, llm, ai-engineering, nvidia, inference]
---
A long story. Very long. But every part of it matters, so bear with me.

## Where it started: a laptop and a lot of hope

I have been doing a lot of experiments related to Machine Learning, mostly on my Acer Intel Evo laptop. It was a time when I was just experimenting with ML, and things were smooth — until I started working on Neural Networks.

That's when I really understood the importance of a GPU.

![CPU vs GPU parallel processing](/assets/img/dgx-spark/CPU_VS_GPU.png)

CPUs are powerful, but they are built around a relatively small number of powerful cores. Neural networks don't work quite like that. The real work in a neural network is heavy matrix multiplication, done again and again, at scale. A CPU has limited cores, while a GPU has thousands of simpler cores, all designed to handle many operations in parallel. GPUs are built to finish many small tasks at once, even if each individual task is relatively simple.

Neural net training is throughput-bound. You don't care how fast one multiplication is. You care about how many operations you can push through per second. That's where Tensor Cores come in — specialized hardware designed to accelerate matrix multiply-accumulate operations.

On my laptop, a CNN experiment used to take around 2 hours. Then I moved to Colab, which gave me a T4 GPU (16GB GDDR6) — fast, but demanding. The same task that took 2 hours now finished in 5 to 15 minutes, and 15 minutes only if I hadn't optimized my epochs or shortened my batch size.

But session timeouts and out-of-memory errors became my new normal. The real issue wasn't speed. It was control. I didn't have control over my own environment.

## Bigger isn't always better

When I moved into Gen AI and transformer architectures, the only real option available was APIs.

For a long time, I believed every problem needed an optimized, best-possible model. That's just how I was trained to think — bigger model, better result.

But AI is a much bigger set than that. Not every problem needs a bigger model.

During one of my experiments — building a tool to merge code from one repo to another — I noticed something. A normal embedding of code chunks using a smaller model like MiniLM worked just fine. Even for retrieval, a smaller AI model was enough to find similar code and make the right changes. In another task, where I just needed to extract token names and country names, a small NER model did the job perfectly.

Around the time people were building tools on Claude Code — say, for simple DB query optimization — I started thinking the same way: a normal agent with a smaller GenAI model is often enough.

That thought took me to Hugging Face, and from there my interest in LLMs grew — specifically local LLMs. I started looking into optimization, into QLoRA techniques. I kept coming back to one idea: train a smaller model, optimized for a specific piece of knowledge, so it can help with one specific task.

But for that, I needed a bigger GPU. And more control.

## The control problem

AWS and GCP were options, but they came with a steep learning curve and a cost I could never fully predict. That risk of a surprise bill always sat at the back of my mind.

There was another layer to this. I work with Banks and Financial Technology. These institutions operate in controlled environments. Data sensitivity and information security are always their biggest concern. That's exactly why they still rely on local infrastructure.

In AI, their only real option was a private cloud. But even there, I kept seeing the same pattern — uncontrolled token spending, unstructured approaches, no real regulation, and very little focus on building an AI Engineering mindset first, or actually understanding the LLM before throwing money at it.

Companies were spending millions. The same companies calling themselves "AI First," posting heavily on LinkedIn about it.

Watching this made one thing very clear to me by late 2025: I needed a machine with at least 32GB of GPU memory. My own machine. My own control.

## Chasing the right hardware

Some hope arrived when the RTX 5090 launched. I thought this could be my starting point — run Unsloth, use LoRA, get my hands dirty. But the GPU disappeared from the market almost instantly, and the price shot up to 4-5 Lakhs.

I looked at a MacBook next, but I already knew MLX inferencing is still far behind CUDA. Then I considered the AMD Strix Halo — 128GB of unified memory, which sounded great on paper, but it came with its own set of limitations.

![My NVIDIA DGX Spark](/assets/img/dgx-spark/Search_of_machine.png)

Then, in April, at an event in Mumbai, I saw a firm selling the NVIDIA DGX Spark to corporates and smaller educational institutes. I reached out to them, followed up for quite some time, asking if I could get one for personal use. Somehow, they made it happen. I received it by the end of May.

## Now, the elephant in the room: the DGX Spark

![Large Hero](/assets/img/dgx-spark/ASUS_Ascent_GX10.png)

Here's what makes it worth the wait.

It has 128GB of unified memory — memory that the GPU can share directly — along with up to 1 petaFLOP of AI performance. That's a serious amount of computation sitting on your desk.

It moves easily between being a desktop and an AI server. With NVIDIA Sync, I can connect it straight to my laptop. Physically, it's about the size of a Mac Mini, and it consumes roughly the same power as a laptop — which makes it a genuinely sustainable product to run.

The one drawback I found is data speed. Because it uses LPDDR5 RAM, it runs at around 273GB/sec, which makes it slower than the newest consumer GPUs like the RTX 5090.

But the real advantage is the NVIDIA CUDA ecosystem. PyTorch, TensorFlow, JAX — all of it, working the way it's meant to. TensorRT brings much better LLM optimization, and there's a whole line of NVIDIA tools around LLM inferencing, AI Ops, Guardrails, and Agentic workflows.

That ecosystem alone was a huge addition. While I was getting into inferencing first, I found that the best quantization techniques are still built around the NVIDIA architecture. My interest has now moved toward production-grade inference engines — ones that give far better configuration and optimization than a simple Ollama or LM Studio setup.

## What's next

Before I move into model training, I want to share a few more posts on the research side first. Up next: inferencing, quantization techniques, and optimizing the model itself. After that, KV cache and faster token generation.

So keep an eye out.
