# Method Description: Qwen3.5-4B + LoRA-SFT for ORena FOCUS SEGMENT

## Model and Adaptation

- Base model: `Qwen/Qwen3.5-4B`
- Adaptation: LoRA supervised fine-tuning
- LoRA rank `8`, alpha `16`, dropout `0.05`
- Vision backbone frozen; LoRA is applied to language attention, Gated
  DeltaNet, and MLP linear projections
- BF16 model execution on the official GPU

## Training Data

- Official SEGMENT TRAIN samples from both HeiCo and LapChole
- `13,746` total official training questions
- deterministic split with seed `20260707`: `12,372` train and `1,374`
  validation questions
- all clip windows passed the media/path audit
- official TEST samples were held out

## Training

- five epochs on four NVIDIA L20 GPUs using DDP
- global batch size `4`
- learning rate `1e-4`
- prompt tokens masked so supervision is applied only to assistant answers
- timestamp-overlay videos used for temporal grounding

## Inference Alignment

- deterministic non-thinking Qwen3.5 chat template
- videos decoded once with Decord and passed to the processor as in-memory RGB
  frames
- `1 FPS` target sampling, `4` minimum frames, `64` maximum frames
- frames resized to `640 x 360`
- explicit frame-index metadata supplied for Qwen3.5 timestamp tokens
- fully offline base model and LoRA adapter under the Docker `resources/`
