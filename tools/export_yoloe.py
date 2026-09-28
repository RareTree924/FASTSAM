"""Exports YOLOE-11L-seg for the CamPipe app as two Core ML models + tokenizer data.

  YOLOE-seg.mlpackage   image [1,3,480,480] (RGB, 0-255) + text [1,1,512]
                        -> det   [1, 4+2+32, 4725]  cx, cy, w, h (pixels), text score, object score, 32 mask coefficients
                        -> proto [1, 32, 120, 120]  mask prototypes
                        "text score": how well each spot matches the typed words (YOLOE text prompt).
                        "object score": YOLOE's prompt-free score (best match over its 4,585-name
                        vocabulary) - used when nothing was typed, to outline everything.
                        Both come from one pass: the prompt-free checkpoint shares the backbone,
                        box, mask and class-feature layers with the text one (checked below).
  YOLOE-text.mlpackage  tokens [1,77] (CLIP BPE ids) -> text [1,1,512]
                        MobileCLIP-B(LT) text encoder + YOLOE's text adapter (reprta), normalized.
  clip_merges.txt       the BPE merge list the app's tokenizer needs
  tokenizer_tests.json  sample texts and their token ids, to test the app's tokenizer

Run on macOS for the Core ML part (`--no-coreml` does just the PyTorch checks).
"""

import argparse
import gzip
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ultralytics import YOLOE
from ultralytics.nn.text_model import build_text_model
from ultralytics.utils import ASSETS
from ultralytics.utils.tal import dist2bbox, make_anchors

SIZE = 480


class YOLOEOutlines(nn.Module):
    """Backbone + head of the text-prompt model, plus the prompt-free vocabulary as a second score."""

    def __init__(self, tp, pf):
        super().__init__()
        head, pf_head = tp.model[-1], pf.model[-1]
        self.layers = tp.model[:-1]
        self.save = set(tp.save)
        self.f = list(head.f)
        self.nl = head.nl
        self.reg_max = head.reg_max
        self.cv2, self.cv5, self.dfl, self.proto = head.cv2, head.cv5, head.dfl, head.proto
        self.cls_feat = nn.ModuleList(c[:-1] for c in head.cv3)      # shared by both scores
        self.cls_embed = nn.ModuleList(c[-1] for c in head.cv3)      # 256 -> 512 region embedding
        self.bn = nn.ModuleList(h.norm for h in head.cv4)
        self.register_buffer("scale", torch.stack([h.logit_scale.exp().reshape(()) for h in head.cv4]))
        self.register_buffer("bias", torch.stack([h.bias.reshape(()) for h in head.cv4]))
        self.vocab = nn.ModuleList(l.vocab for l in pf_head.lrpc)     # 256 -> 4585 (Linear, or Conv2d when not enabled)

        with torch.no_grad():
            feats = self._feats(torch.zeros(1, 3, SIZE, SIZE))
        anchors, strides = make_anchors(feats, head.stride, 0.5)
        self.register_buffer("anchors", anchors.transpose(0, 1).unsqueeze(0))  # [1, 2, A]
        self.register_buffer("strides", strides.transpose(0, 1))              # [1, A]

    def _feats(self, x):
        y = []
        for m in self.layers:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return [y[j] for j in self.f]

    def forward(self, image, text):
        feats = self._feats(image)
        b = image.shape[0]
        box = torch.cat([self.cv2[i](feats[i]).view(b, 4 * self.reg_max, -1) for i in range(self.nl)], 2)
        dbox = dist2bbox(self.dfl(box), self.anchors, xywh=True, dim=1) * self.strides

        w = F.normalize(text, dim=-1, p=2)                            # [b, 1, 512]
        tp_scores, pf_scores = [], []
        for i in range(self.nl):
            c = self.cls_feat[i](feats[i])                            # [b, 256, H, W]
            e = self.bn[i](self.cls_embed[i](c)).flatten(2)           # [b, 512, N]
            tp_scores.append(torch.einsum("bcn,bkc->bkn", e, w) * self.scale[i] + self.bias[i])
            v = self.vocab[i]
            if isinstance(v, nn.Linear):
                logits = v(c.flatten(2).transpose(1, 2)).amax(-1, keepdim=True).transpose(1, 2)
            else:
                logits = v(c).flatten(2).amax(1, keepdim=True)
            pf_scores.append(logits)
        scores = torch.cat([torch.cat(tp_scores, 2), torch.cat(pf_scores, 2)], 1).sigmoid()

        mc = torch.cat([self.cv5[i](feats[i]).view(b, -1, feats[i].shape[2] * feats[i].shape[3])
                        for i in range(self.nl)], 2)
        return torch.cat([dbox, scores, mc], 1), self.proto(feats[0])


class TextEncoder(nn.Module):
    """Tokens -> YOLOE text prompt embedding (what YOLOEModel.get_text_pe returns)."""

    def __init__(self, enc, head):
        super().__init__()
        self.enc = enc
        self.reprta = head.reprta

    def forward(self, tokens):
        t = self.enc(tokens)                                           # MobileCLIP output, already normalized
        return F.normalize(self.reprta(t), dim=-1, p=2).unsqueeze(1)   # [b, 1, 512]


def load_image():
    """bus.jpg centre-cropped to a 480x480 square, like the photos from the camera."""
    from PIL import Image
    im = Image.open(ASSETS / "bus.jpg").convert("RGB")
    s = min(im.size)
    im = im.crop(((im.width - s) // 2, (im.height - s) // 2, (im.width + s) // 2, (im.height + s) // 2))
    return im.resize((SIZE, SIZE))


def top(det, ch, k=3):
    s = det[0, 4 + ch]
    idx = s.argsort(descending=True)[:k]
    return [(round(s[i].item(), 3), [round(v, 1) for v in det[0, :4, i].tolist()]) for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="export")
    ap.add_argument("--no-coreml", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tp_yolo, pf_yolo = YOLOE("yoloe-11l-seg.pt"), YOLOE("yoloe-11l-seg-pf.pt")
    tp, pf = tp_yolo.model.eval(), pf_yolo.model.eval()

    # The one-pass trick relies on these being identical between the two checkpoints.
    tsd, psd = tp.state_dict(), pf.state_dict()
    nh = f"model.{len(tp.model) - 1}."
    shared = [k for k in tsd if not k.startswith(nh)]
    shared += [k for k in tsd if k.startswith(nh) and k.split(".")[2] in ("cv5", "proto", "dfl")]
    shared += [k for k in tsd if k.startswith(nh + "cv3.") and k.split(".")[4] != "2"]
    bad = [k for k in shared if k not in psd or not torch.equal(tsd[k], psd[k])]
    assert not bad, f"prompt-free checkpoint differs from the text one: {bad[:5]}"
    for i in range(tp.model[-1].nl):   # pf keeps the box head's last conv inside lrpc.loc
        a, b = tp.model[-1].cv2[i], pf.model[-1].cv2[i]
        assert all(torch.equal(x, y) for x, y in zip(a[:-1].state_dict().values(), b.state_dict().values()))
        assert torch.equal(a[-1].weight, pf.model[-1].lrpc[i].loc.weight)
    print(f"checkpoints share {len(shared)} tensors - one model can give both scores")

    # Same preparation as Ultralytics' own Core ML export: fold BatchNorms into the
    # convs, and give the attention blocks their Core ML-friendly path.
    from ultralytics.nn.modules.block import Attention
    tp.fuse(verbose=False, imgsz=SIZE)
    net = YOLOEOutlines(tp, pf).eval()
    for m in net.modules():
        if isinstance(m, Attention):
            m.format = "coreml"
    text_model = build_text_model("mobileclip:blt", device="cpu")
    text_net = TextEncoder(text_model.encoder, tp.model[-1]).eval()

    im = load_image()
    x = torch.from_numpy(np.asarray(im)).permute(2, 0, 1)[None].float() / 255
    prompt = "bus"
    tokens = text_model.tokenize([prompt])
    with torch.no_grad():
        tpe = text_net(tokens)
        ref_tpe = tp.get_text_pe([prompt])
        print("text embedding vs ultralytics: max diff", (tpe - ref_tpe).abs().max().item())
        det, proto = net(x, tpe)
    print("det", tuple(det.shape), "proto", tuple(proto.shape))
    print("ours, text 'bus':  ", top(det, 0))
    print("ours, everything:  ", top(det, 1))

    tp_yolo.set_classes([prompt], ref_tpe)
    r = tp_yolo.predict(np.asarray(im)[..., ::-1], imgsz=SIZE, conf=0.25, verbose=False)[0]
    print("ultralytics text:  ", [(round(c, 3), [round(v, 1) for v in b]) for c, b in
                                  zip(r.boxes.conf.tolist()[:3], r.boxes.xywh.tolist()[:3])])
    r = pf_yolo.predict(np.asarray(im)[..., ::-1], imgsz=SIZE, conf=0.25, verbose=False)[0]
    print("ultralytics pf:    ", [(round(c, 3), [round(v, 1) for v in b]) for c, b in
                                  zip(r.boxes.conf.tolist()[:3], r.boxes.xywh.tolist()[:3])])

    # Tokenizer data for the app, plus test cases.
    import clip
    bpe = os.path.join(os.path.dirname(clip.__file__), "bpe_simple_vocab_16e6.txt.gz")
    merges = gzip.open(bpe).read().decode("utf-8").split("\n")[1:49152 - 256 - 2 + 1]
    with open(os.path.join(args.out, "clip_merges.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(merges))
    tests = ["bus", "Coffee Mug", "screw-driver", "a red apple!", "  lots   of   spaces ", "x", "#7 bolt", "don't",
             "USB-C cable", "Rubik's cube", "3.5mm jack", "~weird~ [chars] {ok}"]
    toks = text_model.tokenize(tests)
    with open(os.path.join(args.out, "tokenizer_tests.json"), "w") as f:
        json.dump([{"text": t, "ids": [int(v) for v in row if v != 0]} for t, row in zip(tests, toks.tolist())], f)
    print(f"wrote {len(merges)} merges and {len(tests)} tokenizer tests")

    if args.no_coreml:
        return

    import coremltools as ct

    with torch.no_grad():
        traced = torch.jit.trace(net, (x, tpe), strict=False, check_trace=False)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, SIZE, SIZE), scale=1 / 255, color_layout=ct.colorlayout.RGB),
                ct.TensorType(name="text", shape=(1, 1, 512))],
        outputs=[ct.TensorType(name="det"), ct.TensorType(name="proto")],
        minimum_deployment_target=ct.target.iOS16,
    )
    mlmodel.short_description = "YOLOE-11L-seg: text-prompted + prompt-free boxes and masks, 480x480"
    mlmodel.save(os.path.join(args.out, "YOLOE-seg.mlpackage"))

    with torch.no_grad():
        traced_t = torch.jit.trace(text_net, (tokens,), strict=False, check_trace=False)
    tmodel = ct.convert(
        traced_t,
        inputs=[ct.TensorType(name="tokens", shape=(1, 77), dtype=np.int32)],
        outputs=[ct.TensorType(name="text")],
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT32,   # small model, run once per prompt: keep it exact
    )
    # 8-bit weights keep it under GitHub's 100 MB file limit. Only the big matrices: the
    # 77x77 causal attention mask (5,929 values, full of -inf) must stay as it is.
    tmodel = ct.optimize.coreml.linear_quantize_weights(
        tmodel, ct.optimize.coreml.OptimizationConfig(
            ct.optimize.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", weight_threshold=20000)))
    tmodel.short_description = "MobileCLIP-B(LT) text encoder + YOLOE text adapter"
    tmodel.save(os.path.join(args.out, "YOLOE-text.mlpackage"))

    # Core ML (on this Mac) against PyTorch.
    if sys.platform == "darwin":
        out = mlmodel.predict({"image": im, "text": tpe.numpy()})
        cdet = torch.from_numpy(out["det"])
        print("coreml, text 'bus':", top(cdet, 0))
        print("coreml, everything:", top(cdet, 1))
        # FP16 on the phone: compare where it matters, the 100 most confident spots of each score.
        for ch, name in ((0, "text"), (1, "everything")):
            idx = det[0, 4 + ch].argsort(descending=True)[:100]
            d = (cdet[0][:, idx] - det[0][:, idx]).abs()
            print(f"coreml vs PyTorch, top-100 {name}: box max diff {d[:4].max():.2f} px, "
                  f"score max diff {d[4:6].max():.3f}, mask coeff max diff {d[6:].max():.3f}")
        print("proto mean abs diff:", (torch.from_numpy(out["proto"]) - proto).abs().mean().item())
        ct_tpe = tmodel.predict({"tokens": tokens.numpy().astype(np.int32)})["text"]
        cos = F.cosine_similarity(torch.from_numpy(ct_tpe).flatten(), tpe.flatten(), dim=0).item()
        print("coreml text embedding cosine vs PyTorch:", cos)
        assert cos > 0.99, "quantized text encoder drifted too far"
        for name in ("YOLOE-seg.mlpackage", "YOLOE-text.mlpackage"):
            total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(os.path.join(args.out, name)) for f in fs)
            print(f"{name}: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
