# Chainmind demo: voiceover and post

The video (`demo/out/chainmind-demo.mp4`, 1920×1080, about 95 seconds) has no audio. It works muted on X with its on-screen captions; this script is for recording a voiceover over it if you want one. Times are approximate; each line fits its scene at a normal speaking pace.

## Voiceover

**0:00 · Intro**
> This is Chainmind. A language model that lives inside the NEAR blockchain.

**0:05 · Real testnet transaction**
> Here's a real transaction on NEAR testnet. I ask the contract for a story with seed one. Every validator runs the full model inside the transaction, and agrees on the same words. It cost 194 teragas, about two cents.

**0:17 · Landing page hero**
> The page you're looking at runs the exact same model in your browser. Every token it writes fires through the model's five layers.

**0:24 · Why**
> Today, if a smart contract wants an AI's opinion, it has to trust someone: an oracle, an API company, or a secure server. Chainmind removes the someone.

**0:33 · One transaction**
> Inside one transaction, the contract loads its sealed weights, reads the prompt, runs the transformer token by token, and samples with the block's randomness. Here, 194 teragas for 48 tokens.

**0:47 · Proof**
> And it's deterministic. The NEAR VM, a laptop, and your browser produce the same 32 tokens, bit for bit. Anyone can check any answer.

**0:55 · Use cases**
> That makes it a building block: fair game AI, NFTs with provable origin, risk scoring for DeFi, safety checks for AI agents.

**1:01 · Live demo**
> Try it yourself. Type a prompt, pick a seed, and you get exactly what the chain would write.

**1:08 · Developers**
> Call it from the command line for free, on chain as a transaction, or from your own contract.

**1:15 · Limits and roadmap**
> We measured everything: up to seventy tokens per transaction, and it stops safely before the gas limit. Next, we split bigger models across NEAR's shards, and use this engine to check big off-chain models on chain.

**1:30 · Outro**
> Chainmind. Verifiable AI for NEAR smart contracts. Live on testnet now.

## X post

Required by the hackathon: say you built it at NEAR HACKATHON, tag @NEARProtocol and #NEARHackathon, and attach the video.

> I built Chainmind at NEAR HACKATHON: a language model that lives inside a NEAR smart contract.
>
> Every validator runs the model in the transaction and gets the same words. No oracle, no API key, no trusted hardware.
>
> Try it in your browser: https://chainmind-theta.vercel.app
> Live on testnet: mind-mwdofvuu.leston.testnet
>
> @NEARProtocol #NEARHackathon

Add the GitHub link once the repo is public. If the post is over the length limit, drop the second paragraph first.
