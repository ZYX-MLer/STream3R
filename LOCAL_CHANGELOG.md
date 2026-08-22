# Local STream3R Changes

This file records local changes made to the upstream STream3R checkout. The
original implementation is retained as comments next to each replacement so
that the changes can be reviewed or reverted without consulting this document.

## 2026-07-24 — Enable Lightning validation with `CausalLoss`

### Context

Training is launched with:

```text
/home/boe/Work/ProjectBOE_syn/vSLAM/3RGSv3/train_stream3r.py
```

and the experiment configuration:

```text
config/experiment/stream3r_my_train_val.yaml
```

That configuration enables a `MegaDepth_Multi` validation dataset and uses
`CausalLoss` for both training and validation.

The run reached Lightning's validation sanity check, loaded the validation
dataset successfully, and then failed before the first training iteration with:

```text
AttributeError: 'list' object has no attribute 'shape'
```

The traceback ended at `stream3r/models/stream3r.py`, where
`STream3R.forward()` evaluated `images.shape`.

### Change 1: convert view dictionaries during validation

File:

```text
stream3r/models/stream3r.py
```

Location:

```text
STream3R.forward(), immediately after the docstring
```

Upstream behavior:

```python
if self.training:
    images = torch.stack([view["img"] for view in images], dim=1)
    images = (images + 1.) / 2.
```

Problem:

- The Lightning module passes a list of view dictionaries to the network in
  both `training_step()` and `validation_step()`.
- Lightning sets the network to evaluation mode during validation, making
  `self.training` false.
- The upstream condition therefore leaves the validation input as a Python
  list.
- The following `images.shape` access fails.

Local behavior:

```python
input_is_views = isinstance(images, (list, tuple))
if input_is_views:
    images = torch.stack([view["img"] for view in images], dim=1)
    images = (images + 1.) / 2.
```

Reason:

Input conversion depends on the input representation, not on whether the model
is training. Tensor inputs used by the normal inference launchers remain
unchanged. Validation remains in evaluation mode and under Lightning's
no-gradient evaluation context.

### Change 2: expose iterative camera predictions to validation loss

File:

```text
stream3r/models/stream3r.py
```

Location:

```text
STream3R.forward(), camera-head prediction block
```

Upstream behavior:

```python
if self.training:
    predictions["pose_enc_list"] = pose_enc_list
```

Problem:

`stream3r/loss/losses.py::CausalLoss.compute_loss()` unconditionally reads:

```python
preds["pose_enc_list"]
```

The upstream network only exposes that value in training mode. After fixing
the input-list error, validation would consequently fail with a missing
`pose_enc_list` key.

Local behavior:

```python
if self.training or input_is_views:
    predictions["pose_enc_list"] = pose_enc_list
```

Reason:

A list/tuple input identifies the training-data view-dictionary interface used
by both Lightning training and validation. It makes the iterative camera
predictions available to `CausalLoss` without changing tensor-based inference
outputs.

### Compatibility and scope

- Training with view dictionaries retains the original input normalization and
  output keys.
- Validation now receives the same normalized image tensor representation as
  training while the model itself remains in evaluation mode.
- Tensor-based inference is unchanged: it does not perform the view-dictionary
  conversion and does not retain `pose_enc_list`.
- No CUT3R source file was changed.
- Dataset files and preprocessing outputs were not changed.

### Verification

The change was verified in the server's `vSLAM` environment in two stages:

1. A lightweight eval-mode unit test confirmed that a list of view
   dictionaries becomes a `[B, S, 3, H, W]` tensor, exposes
   `pose_enc_list`, and leaves the tensor-based inference interface unchanged.
2. A real no-gradient validation batch used one `MegaDepth_Multi` sample with
   10 views at `518 x 336`, the real `STream3R` network, and the real
   `CausalLoss`. The complete forward and loss computation passed, returned
   four camera-pose iterations and nine loss-detail values, and used about
   5.5 GiB peak allocated GPU memory.

### Related pre-existing local validation guard

The checkout already contains a separate local change in:

```text
stream3r/data/multiview_dust3r_datamodule.py
```

It replaces an unconditional `dataset.set_ratio(1.0)` call with an
`hasattr(dataset, "set_ratio")` guard. This is required because not every
validation dataset implements `set_ratio()`. It is related to enabling
validation but is not the cause of the `images.shape` failure addressed above.

### Reverting

To revert only the changes documented in this entry:

1. Remove the `input_is_views` assignment and its replacement conversion block.
2. Uncomment the preserved upstream `if self.training` conversion block.
3. Remove the `if self.training or input_is_views` camera-output block.
4. Uncomment the preserved upstream `if self.training` camera-output block.

After reverting, Lightning validation with the current view-list input and
`CausalLoss` configuration will reproduce the original failures.
