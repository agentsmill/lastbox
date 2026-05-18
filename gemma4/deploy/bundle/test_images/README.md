# LastBox vision test images

Three CC-licensed images from Wikimedia Commons for end-to-end vision smoke test
through the deployed llama-server (`mmproj-F16.gguf` SigLIP encoder).

## Files

| File | Source | Use case | Expected LastBox answer (lora source ≤150 B) |
|------|--------|----------|---------------------------------------------|
| `rowan_berry.jpg` | [Rowan article](https://en.wikipedia.org/wiki/Rowan), Wikimedia Commons | "Is this safe to eat?" — **edible** berry, bitter raw | "Rowan berry. Edible cooked (jelly), bitter raw. Cluster of small orange-red fruits." |
| `yew_berry_toxic.jpg` | [Taxus baccata article](https://en.wikipedia.org/wiki/Taxus_baccata), Wikimedia Commons | "Is this safe to eat?" — **TOXIC** (seeds lethal) | "Yew (Taxus baccata). Seeds + leaves lethal — taxine alkaloid. Do not eat. Red flesh only is non-toxic." |
| `bleeding_finger.jpg` | [Bleeding article](https://en.wikipedia.org/wiki/Bleeding), Wikimedia Commons | "What's wrong, first aid?" — minor cut | "Minor cut. Apply pressure 2-5 min, rinse with clean water, cover with sterile dressing." |

## Demo invocation

```bash
LASTBOX_ENDPOINT=http://lastbox:11436/v1 python ../demo.py \
    --source lora \
    --image test_images/rowan_berry.jpg \
    --query "is this safe to eat?"
```

The orchestrator base64-encodes the image, sends it in the OpenAI-compatible
`messages[].content[].image_url` format, llama-server's mmproj path runs the
SigLIP encoder, and the model produces a text answer respecting the byte cap.

Expected latency on RPi 5 CPU: 10-15 s for vision encoding + 1-2 s/answer.

## License

Each image retains its Wikimedia license (CC-BY-SA or public domain — see the
linked file pages on Wikimedia Commons). They are included here for evaluation
and demo purposes only, with attribution preserved.
