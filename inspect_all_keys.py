import torch, sys
for name in ["weights/skeletongait++.pt", "weights/deepgaitv2.pt"]:
    ckpt = torch.load(name, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    print(f"\n===== {name} ({len(sd)} keys) =====", flush=True)
    for k, v in sd.items():
        if hasattr(v, 'shape'):
            print(f"{k}  {list(v.shape)}", flush=True)
        else:
            print(f"{k}  {v}", flush=True)
    sys.stdout.flush()
