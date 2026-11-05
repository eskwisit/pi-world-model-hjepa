"""Run an H-JEPA release script (eval.py) on a machine without an NVIDIA GPU.

Usage, from the release's h_jepa/ directory, with its Python:

    HJEPA_DEVICE=mps|cpu|auto python hjepa_local.py eval.py --config-name ant_l3 seed=42 policy=...

The pinned release assumes CUDA in three places. This launcher adapts them in memory and leaves the
clone untouched:

1. `model.to("cuda")` and any other move to "cuda" are sent to HJEPA_DEVICE instead
   (auto: Apple GPU "mps" when available, else "cpu").
2. Every `device: cuda` in the selected eval config is overridden on the command line, unless the
   caller already set that key.
3. MPS has no float64. Tensors moved to MPS as float64 are cast to float32.
4. eval.py forces MUJOCO_GL=egl (Linux headless rendering). When MUJOCO_GL is already set (the
   runner sets glfw on macOS), that assignment is ignored.

Results are not expected to match the paper bit for bit (other hardware, float32 on MPS).
Shipped by the World Model (H-JEPA) PI-Desktop plugin.
"""

import os
import runpy
import sys


def pick_device():
    import torch

    want = os.environ.get("HJEPA_DEVICE", "auto").lower()
    if want == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if want == "mps" and not torch.backends.mps.is_available():
        sys.exit("hjepa_local: HJEPA_DEVICE=mps but torch reports no MPS device")
    if want not in ("mps", "cpu"):
        sys.exit(f"hjepa_local: HJEPA_DEVICE must be auto, mps or cpu, not {want!r}")
    return want


def patch_torch(device):
    import torch

    def is_cuda(d):
        return (isinstance(d, str) and d.split(":")[0] == "cuda") or (isinstance(d, torch.device) and d.type == "cuda")

    def remap(d):
        return device if is_cuda(d) else d

    tensor_to = torch.Tensor.to

    def to(self, *args, **kwargs):
        dev, dtype, rest = kwargs.pop("device", None), kwargs.pop("dtype", None), []
        for a in args:
            if a is None:  # Module.to passes dtype=None positionally
                continue
            if isinstance(a, (str, torch.device)):
                dev = a
            elif isinstance(a, torch.dtype):
                dtype = a
            elif torch.is_tensor(a):
                dev, dtype = a.device, dtype or a.dtype
            else:
                rest.append(a)
        dev = remap(dev)
        if dev is not None and torch.device(dev).type == "mps" and (dtype or self.dtype) == torch.float64:
            dtype = torch.float32
        if rest:
            kwargs.setdefault("non_blocking", rest[0])
        if len(rest) > 1:
            kwargs.setdefault("copy", rest[1])
        return tensor_to(self, device=dev, dtype=dtype, **kwargs)

    torch.Tensor.to = to

    module_to = torch.nn.Module.to

    def mod_to(self, *args, **kwargs):
        args = tuple(remap(a) for a in args)
        if "device" in kwargs:
            kwargs["device"] = remap(kwargs["device"])
        return module_to(self, *args, **kwargs)

    torch.nn.Module.to = mod_to

    def cuda(self, *args, **kwargs):  # Tensor.cuda() / Module.cuda()
        return self.to(device)

    torch.Tensor.cuda = cuda
    torch.nn.Module.cuda = cuda

    # Tensors created on "cuda" directly, e.g. torch.zeros(..., device="cuda") in droid_eval.py.
    def factory(fn):
        def wrapped(*args, **kwargs):
            if kwargs.get("device") is not None:
                kwargs["device"] = remap(kwargs["device"])
                if torch.device(kwargs["device"]).type == "mps" and kwargs.get("dtype") == torch.float64:
                    kwargs["dtype"] = torch.float32
            return fn(*args, **kwargs)

        wrapped.__wrapped__ = fn
        return wrapped

    for name in FACTORIES:
        fn = getattr(torch, name, None)
        if fn is not None:
            setattr(torch, name, factory(fn))


FACTORIES = (
    "zeros", "ones", "empty", "full", "rand", "randn", "randint", "randperm", "arange", "linspace",
    "logspace", "eye", "tensor", "as_tensor", "zeros_like", "ones_like", "empty_like", "full_like",
    "rand_like", "randn_like", "randint_like",
)


def keep_renderer():
    if not os.environ.get("MUJOCO_GL"):
        return
    env_cls = type(os.environ)
    setitem = env_cls.__setitem__

    def guarded(self, key, value):
        if self is os.environ and key in ("MUJOCO_GL", "PYOPENGL_PLATFORM"):
            return
        setitem(self, key, value)

    env_cls.__setitem__ = guarded


def device_overrides(argv, device):
    """`<key>=<device>` for every `device: cuda` of the selected eval config the caller did not set."""
    from omegaconf import OmegaConf

    name = None
    for i, a in enumerate(argv):
        if a == "--config-name" and i + 1 < len(argv):
            name = argv[i + 1]
        elif a.startswith("--config-name="):
            name = a.split("=", 1)[1]
    if not name:
        return []
    path = os.path.join("config", "eval", f"{name}.yaml")
    if not os.path.exists(path):
        return []
    cfg = OmegaConf.load(path)
    found = []

    def walk(node, prefix):
        for key in node.keys():
            child = node._get_node(key)
            if OmegaConf.is_dict(child):
                walk(child, f"{prefix}{key}.")
            elif key == "device" and str(child._value()) == "cuda":
                found.append(f"{prefix}{key}")

    walk(cfg, "")
    given = {a.split("=", 1)[0].lstrip("+") for a in argv if "=" in a}
    return [f"{k}={device}" for k in found if k not in given]


def main():
    if len(sys.argv) < 2 or not sys.argv[1].endswith(".py"):
        sys.exit("usage: hjepa_local.py eval.py [hydra args...]")
    script, args = sys.argv[1], sys.argv[2:]
    device = pick_device()
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    patch_torch(device)
    keep_renderer()
    extra = device_overrides(args, device)
    print(f"hjepa_local: device={device} MUJOCO_GL={os.environ.get('MUJOCO_GL', '')} overrides={extra}", file=sys.stderr, flush=True)
    sys.argv = [script, *args, *extra]
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
