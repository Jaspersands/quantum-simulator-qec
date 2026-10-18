# %% [markdown]
# # Real time, and the GPU
#
# A decoder in an experiment is a service: syndromes arrive, corrections must come back, and
# how long each took matters as much as whether it was right. This package's decoders run in
# the background, behind a queue with statistics, and behind a server another process or
# machine talks to (CUDA-Q QEC's `decode_async`, `enqueue_syndromes` and `get_corrections`,
# on this package's decoders). And its samplers and min-sum BP run on a GPU through wgpu,
# bit for bit the CPU's.

# %%
import threading
import time

import numpy as np
import stabilizer_qec as sq
from stabilizer_qec.realtime import DecoderClient, DecoderQueue, serve

c = sq.memory_circuit(distance=5, rounds=5, p=0.003)
dem = c.detector_error_model(decompose_errors=True)
dets, obs = c.compile_detector_sampler(seed=1).sample(20_000, separate_observables=True)
decoder = sq.Matching(dem)

# %% [markdown]
# ## A decode in the background
#
# Every detector-error-model decoder has `decode_async`: the batch runs on a worker thread with the GIL released in
# the engine, and the result is an `AsyncResult` you ask for when you need it.

# %%
job = decoder.decode_async(dets)
busy = 0
while not job.ready():
    busy += 1  # Python keeps working while the engine decodes
pred = job.get()
print(f"{(pred != obs).any(axis=1).mean():.4%} logical error; Python looped {busy} times meanwhile; "
      f"latency {job.latency * 1e3:.1f} ms, of which decoding {job.service * 1e3:.1f} ms")

# %% [markdown]
# ## A queue with statistics
#
# `DecoderQueue` takes batches with `enqueue_syndromes` and gives corrections back in order with
# `get_corrections`. Its `stats()` is what an experiment needs: latency from enqueue to result
# (mean, median, 99th percentile), the decode's own time, shots per second, and how deep the
# queue got. Fed faster than it decodes, the latency shows it; two workers decode two batches at
# once.

# %%
for workers in (1, 2):
    with DecoderQueue(decoder, workers=workers) as queue:
        tickets = [queue.enqueue_syndromes(dets[i:i + 1000]) for i in range(0, 20_000, 1000)]
        out = np.concatenate([queue.get_corrections(t) for t in tickets])
        s = queue.stats()
    assert np.array_equal(out, pred)
    print(f"{workers} worker(s): {s.batches} batches, p50 latency {s.p50_latency_s * 1e3:.1f} ms, "
          f"p99 {s.p99_latency_s * 1e3:.1f} ms, service {s.mean_service_s * 1e3:.1f} ms, "
          f"{s.throughput_per_s / 1e3:.0f}k shots/s, queue depth {s.queue_depth_max}")

# %% [markdown]
# ## A decoder server
#
# `stabilizer-qec serve --dem model.dem --decoder matching --tcp host:port` serves a decoder
# over a small framed protocol; `DecoderClient` decodes through it. Here the server runs on a
# thread in this process; it could be another machine. The predictions are the local
# decoder's shot for shot.

# %%
bound = {}
ready = threading.Event()

def on_ready(server):
    bound["address"] = server.server_address[:2]
    bound["server"] = server
    ready.set()

thread = threading.Thread(target=serve, args=(decoder,), kwargs=dict(tcp="127.0.0.1:0", workers=2, ready=on_ready), daemon=True)
thread.start()
ready.wait(10)
host, port = bound["address"]
with DecoderClient.connect(host, port) as client:
    print(client)
    remote = client.decode_batch(dets[:5000])
    print("equal to the local decoder:", np.array_equal(remote, pred[:5000]))
    print("the server's statistics:", client.stats()._asdict()["decoded"], "shots decoded")
bound["server"].shutdown()

# %% [markdown]
# ## Sampling on the GPU
#
# The batch sampler packs 64 shots into every machine word and draws its noise by geometric
# skipping in double precision — nothing a GPU repeats bit for bit. So the GPU runs a second
# program: the same circuit flattened into 32-bit words, 32 shots a word, every random number an
# integer compared with a threshold. That program also runs on the CPU (`device="cpu-lanes"`),
# and the two give the same bytes for every seed, shot count and batching. `gpu_device()` names
# the GPU, or is `None` in a build or on a machine without one (then `device="gpu"` raises).

# %%
print("GPU:", sq.gpu_device())
lanes = c.compile_detector_sampler(seed=7, device="cpu-lanes").sample(32_000, bit_packed=True)
if sq.gpu_device():
    t0 = time.perf_counter()
    gpu = c.compile_detector_sampler(seed=7, device="gpu").sample(32_000, bit_packed=True)
    print(f"GPU equals the CPU lanes byte for byte: {np.array_equal(gpu, lanes)}, in {time.perf_counter() - t0:.2f} s")
words = c.compile_detector_sampler(seed=7).sample(32_000, bit_packed=True)
print("the word sampler draws different shots from the same seed, at the same rate:",
      f"{np.unpackbits(lanes, axis=1).mean():.5f} vs {np.unpackbits(words, axis=1).mean():.5f} detection events per detector")

# %% [markdown]
# ## Min-sum BP on the GPU
#
# `BpDecoder(..., device="cpu-f32")` runs ldpc's parallel min-sum schedule in single precision,
# the operation order the GPU kernel follows; `device="gpu"` is the same decisions bit for bit.
# With damped min-sum (`ms_scaling_factor=0.625`) the f32 decoder gives the f64 one's answer on
# almost every syndrome, at a fraction of the time (undamped or adaptive min-sum grows messages
# past single precision on unconverged shots, where the decisions part and the error rates
# agree); a GPU pays off on small graphs only, where a whole shot's Tanner graph is a short walk.

# %%
from stabilizer_qec import BpDecoder
from stabilizer_qec.beliefmatching import detector_error_model_to_check_matrices

m = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)
H = np.asarray(m.check_matrix.todense() if hasattr(m.check_matrix, "todense") else m.check_matrix, dtype=np.uint8)
f64 = BpDecoder(H, error_channel=np.asarray(m.priors), max_iter=20, bp_method="minimum_sum", ms_scaling_factor=0.625)
f32 = BpDecoder(H, error_channel=np.asarray(m.priors), max_iter=20, bp_method="minimum_sum", ms_scaling_factor=0.625, device="cpu-f32")
syndromes = dets[:2000].astype(np.uint8)
t0 = time.perf_counter(); a = np.stack([f64.decode(s) for s in syndromes]); t64 = time.perf_counter() - t0
t0 = time.perf_counter(); b = f32.decode_batch(syndromes); t32 = time.perf_counter() - t0
print(f"f32 agrees with f64 on {(a == b).all(axis=1).mean():.2%} of syndromes; {t64 / t32:.0f}× faster")
if sq.gpu_device():
    g = BpDecoder(H, error_channel=np.asarray(m.priors), max_iter=20, bp_method="minimum_sum", ms_scaling_factor=0.625, device="gpu")
    print("GPU equals f32 bit for bit:", np.array_equal(g.decode_batch(syndromes), b))
