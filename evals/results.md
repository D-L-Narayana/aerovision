# AeroVision eval — openai/clip-vit-base-patch32 on UC Merced (2100 images, 21 classes)

| method | top-1 | notes |
|---|---|---|
| zero-shot, single prompt | 58.1% | 'an aerial photo of {class}.' |
| zero-shot, 4-template prompt ensemble | 67.3% (top-3 87.6%) | test split 66.1% |
| linear probe on frozen CLIP features | **93.0%** | 50 labelled images / class |

Few-shot probe accuracy (images per class → test top-1, mean of 3 seeds): 1→66.7%, 2→71.5%, 5→81.8%, 10→86.9%, 20→90.6%, 50→93.0%

Thermal detector vs ground truth (200 synthetic radiometric frames, ≥10 °C hotspots): precision 100.0%, recall 93.6%, severity agreement 95.5%, p50 0.60 ms/frame.

CLIP image encoding: 62 ms/image on cpu (2 vCPU).
