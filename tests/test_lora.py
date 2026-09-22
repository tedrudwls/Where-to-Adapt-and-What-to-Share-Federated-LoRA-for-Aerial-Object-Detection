"""Unit tests for exact LoRA behavior and strict adapter state routing."""

from __future__ import annotations

import copy
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel, RTDETRDetectionModel

import models.rtdetr_lora as rtdetr_lora_module
from models.lora import (
    LoRAConv2d,
    LoRALinear,
    LoRAMultiheadAttention,
    get_lora_AB_state_dict,
    get_lora_A_state_dict,
    get_lora_B_state_dict,
    inject_lora_conv2d,
    inject_lora_linear,
    inject_lora_mha,
    set_lora_AB_state_dict,
    set_lora_A_state_dict,
    set_lora_B_state_dict,
)
from models.rtdetr_lora import _validate_pretrained_rtdetr_source
from trainers.trainer import _preserve_eval_state


def _assert_finite_gradient(test_case: unittest.TestCase, parameter, *, nonzero: bool):
    test_case.assertIsNotNone(parameter.grad)
    test_case.assertTrue(torch.isfinite(parameter.grad).all().item())
    if nonzero:
        test_case.assertGreater(float(parameter.grad.abs().max()), 0.0)


class LoRALinearTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_initial_adapter_is_exact_noop_and_base_is_frozen(self):
        base = nn.Linear(5, 3, bias=True)
        reference = copy.deepcopy(base)
        adapter = LoRALinear(base, rank=2, alpha=4.0, dropout=0.0)
        inputs = torch.randn(4, 5)

        expected = reference(inputs)
        actual = adapter(inputs)

        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(adapter.original_layer.weight.requires_grad)
        self.assertFalse(adapter.original_layer.bias.requires_grad)
        self.assertTrue(adapter.lora_A.requires_grad)
        self.assertTrue(adapter.lora_B.requires_grad)

    def test_first_backward_updates_B_then_nonzero_B_unlocks_A_gradient(self):
        adapter = LoRALinear(nn.Linear(5, 3), rank=2, alpha=4.0, dropout=0.0)
        inputs = torch.randn(6, 5)

        adapter(inputs).square().mean().backward()
        _assert_finite_gradient(self, adapter.lora_B, nonzero=True)
        _assert_finite_gradient(self, adapter.lora_A, nonzero=False)
        self.assertTrue(torch.equal(adapter.lora_A.grad, torch.zeros_like(adapter.lora_A.grad)))
        self.assertIsNone(adapter.original_layer.weight.grad)

        adapter.zero_grad(set_to_none=True)
        with torch.no_grad():
            adapter.lora_B.normal_(mean=0.0, std=0.1)
        adapter(inputs).square().mean().backward()
        _assert_finite_gradient(self, adapter.lora_A, nonzero=True)
        _assert_finite_gradient(self, adapter.lora_B, nonzero=True)

    def test_rejects_nonpositive_rank(self):
        with self.assertRaisesRegex(ValueError, "rank must be positive"):
            LoRALinear(nn.Linear(2, 2), rank=0)


class LoRAConv2dTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def test_stride_dilation_and_nonzero_padding_mode_preserve_geometry_and_noop(self):
        cases = [
            nn.Conv2d(4, 6, kernel_size=3, stride=2, padding=1, dilation=1, bias=True),
            nn.Conv2d(4, 6, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.Conv2d(
                4, 6, kernel_size=3, stride=2, padding=1, dilation=1,
                padding_mode="reflect", bias=True,
            ),
        ]
        inputs = torch.randn(2, 4, 17, 19)
        for base in cases:
            with self.subTest(stride=base.stride, dilation=base.dilation,
                              padding_mode=base.padding_mode):
                reference = copy.deepcopy(base)
                adapter = LoRAConv2d(base, rank=3, alpha=6.0, dropout=0.0)
                expected = reference(inputs)
                actual = adapter(inputs)
                self.assertEqual(actual.shape, expected.shape)
                self.assertTrue(torch.equal(actual, expected))

    def test_first_backward_has_connected_finite_B_gradient(self):
        adapter = LoRAConv2d(
            nn.Conv2d(4, 6, kernel_size=3, stride=2, padding=1),
            rank=3,
            alpha=6.0,
            dropout=0.0,
        )
        adapter(torch.randn(2, 4, 15, 13)).square().mean().backward()
        _assert_finite_gradient(self, adapter.lora_B, nonzero=True)
        _assert_finite_gradient(self, adapter.lora_A, nonzero=False)
        self.assertIsNone(adapter.original_layer.weight.grad)

    def test_rejects_grouped_convolution_and_nonpositive_rank(self):
        with self.assertRaisesRegex(ValueError, "groups=1"):
            LoRAConv2d(nn.Conv2d(4, 4, kernel_size=3, groups=4), rank=2)
        with self.assertRaisesRegex(ValueError, "rank must be positive"):
            LoRAConv2d(nn.Conv2d(4, 4, kernel_size=3), rank=0)


class LoRAMultiheadAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)

    def _compare_noop(self, batch_first: bool):
        base = nn.MultiheadAttention(
            embed_dim=8,
            num_heads=2,
            dropout=0.0,
            bias=True,
            batch_first=batch_first,
        )
        reference = copy.deepcopy(base)
        adapter = LoRAMultiheadAttention(base, rank=2, alpha=4.0, dropout=0.0)
        shape = (3, 5, 8) if batch_first else (5, 3, 8)
        query = torch.randn(*shape)
        key_padding_mask = torch.tensor(
            [[False, False, False, True, True],
             [False, False, False, False, True],
             [False, False, False, False, False]],
            dtype=torch.bool,
        )

        expected_output, expected_weights = reference(
            query, query, query,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        actual_output, actual_weights = adapter(
            query, query, query,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )

        torch.testing.assert_close(actual_output, expected_output, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(actual_weights, expected_weights, rtol=1e-6, atol=1e-7)
        self.assertFalse(adapter.original_layer.in_proj_weight.requires_grad)

    def test_initial_adapter_is_noop_for_sequence_first_attention(self):
        self._compare_noop(batch_first=False)

    def test_initial_adapter_is_noop_for_batch_first_attention(self):
        self._compare_noop(batch_first=True)

    def test_first_backward_connects_fused_qkv_B_and_then_A(self):
        adapter = LoRAMultiheadAttention(
            nn.MultiheadAttention(8, 2, dropout=0.0, batch_first=True),
            rank=2,
            alpha=4.0,
            dropout=0.0,
        )
        inputs = torch.randn(3, 5, 8)
        adapter(inputs, inputs, inputs, need_weights=False)[0].square().mean().backward()
        _assert_finite_gradient(self, adapter.lora_B, nonzero=True)
        _assert_finite_gradient(self, adapter.lora_A, nonzero=False)
        for projection, label in enumerate(("Q", "K", "V")):
            with self.subTest(projection=label):
                self.assertGreater(float(adapter.lora_B.grad[projection].abs().max()), 0.0)
        self.assertIsNone(adapter.original_layer.in_proj_weight.grad)

        adapter.zero_grad(set_to_none=True)
        with torch.no_grad():
            adapter.lora_B.normal_(mean=0.0, std=0.1)
        adapter(inputs, inputs, inputs, need_weights=False)[0].square().mean().backward()
        _assert_finite_gradient(self, adapter.lora_A, nonzero=True)
        _assert_finite_gradient(self, adapter.lora_B, nonzero=True)
        for projection, label in enumerate(("Q", "K", "V")):
            with self.subTest(projection=label):
                self.assertGreater(float(adapter.lora_A.grad[projection].abs().max()), 0.0)

    def test_rejects_dropout_nonfused_attention_and_nonpositive_rank(self):
        with self.assertRaisesRegex(ValueError, "dropout"):
            LoRAMultiheadAttention(nn.MultiheadAttention(8, 2), rank=2, dropout=0.1)
        with self.assertRaisesRegex(ValueError, "rank must be positive"):
            LoRAMultiheadAttention(nn.MultiheadAttention(8, 2), rank=0)
        separate = nn.MultiheadAttention(8, 2, kdim=4, vdim=4)
        with self.assertRaisesRegex(ValueError, "equal-dimension"):
            LoRAMultiheadAttention(separate, rank=2)


class StrictStateTests(unittest.TestCase):
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(5, 4)
            self.conv = nn.Conv2d(4, 6, kernel_size=3, padding=1)
            self.attn = nn.MultiheadAttention(8, 2, batch_first=True)

    def _adapted_model(self):
        model = self.ToyModel()
        inject_lora_linear(model, ["linear"], rank=2, alpha=4.0)
        inject_lora_conv2d(model, ["conv"], rank=2, alpha=4.0, min_channels=1)
        inject_lora_mha(model, ["attn"], rank=2, alpha=4.0)
        return model

    def test_roundtrip_restores_every_A_and_B_tensor(self):
        torch.manual_seed(23)
        source = self._adapted_model()
        with torch.no_grad():
            for name, parameter in source.named_parameters():
                if name.endswith(".lora_A") or name.endswith(".lora_B"):
                    parameter.normal_()
        state = get_lora_AB_state_dict(source)

        target = self._adapted_model()
        set_lora_AB_state_dict(target, state)
        restored = get_lora_AB_state_dict(target)
        self.assertEqual(set(restored), set(state))
        for key in state:
            self.assertTrue(torch.equal(restored[key], state[key]), key)
            self.assertEqual(restored[key].device.type, "cpu")

    def test_state_loading_rejects_missing_unexpected_and_wrong_shape(self):
        model = self._adapted_model()
        state = get_lora_A_state_dict(model)
        first_key = sorted(state)[0]

        missing = dict(state)
        missing.pop(first_key)
        with self.assertRaisesRegex(RuntimeError, "missing"):
            set_lora_A_state_dict(model, missing)

        unexpected = dict(state)
        unexpected["not_a_module.lora_A"] = torch.zeros(1)
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            set_lora_A_state_dict(model, unexpected)

        wrong_role = dict(state)
        wrong_role["unrelated.lora_B"] = torch.zeros(1)
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            set_lora_A_state_dict(model, wrong_role)

        wrong_shape = dict(state)
        wrong_shape[first_key] = torch.zeros(1)
        with self.assertRaisesRegex(RuntimeError, "Shape mismatch"):
            set_lora_A_state_dict(model, wrong_shape)

        ab_state = get_lora_AB_state_dict(model)
        ab_state["unrelated.tensor"] = torch.zeros(1)
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            set_lora_AB_state_dict(model, ab_state)


    def test_B_only_state_roundtrip_is_strict(self):
        torch.manual_seed(29)
        source = self._adapted_model()
        with torch.no_grad():
            for name, parameter in source.named_parameters():
                if name.endswith(".lora_B"):
                    parameter.normal_()
        state = get_lora_B_state_dict(source)

        target = self._adapted_model()
        set_lora_B_state_dict(target, state)
        restored = get_lora_B_state_dict(target)
        self.assertEqual(set(restored), set(state))
        for key in state:
            self.assertTrue(torch.equal(restored[key], state[key]), key)

        first_key = sorted(state)[0]
        missing = dict(state)
        missing.pop(first_key)
        with self.assertRaisesRegex(RuntimeError, "missing"):
            set_lora_B_state_dict(target, missing)

        wrong_role = dict(state)
        wrong_role["unrelated.lora_A"] = torch.zeros(1)
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            set_lora_B_state_dict(target, wrong_role)

    def test_exact_target_injection_rejects_missing_linear_conv_and_mha(self):
        model = self.ToyModel()
        with self.assertRaisesRegex(RuntimeError, "target"):
            inject_lora_linear(model, ["does.not.exist"])
        with self.assertRaisesRegex(RuntimeError, "target"):
            inject_lora_conv2d(model, ["does.not.exist"], min_channels=1)
        with self.assertRaisesRegex(RuntimeError, "target"):
            inject_lora_mha(model, ["does.not.exist"])


class EvaluationStateRegressionTests(unittest.TestCase):
    class CacheHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(3, 2)
            self.shapes = [[5, 5]]
            self.anchors = torch.randn(1, 4)
            self.valid_mask = torch.ones(1, 1, dtype=torch.bool)

    class ToyDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(EvaluationStateRegressionTests.CacheHead())

        def fuse(self, *args, **kwargs):
            return "original-fuse"

    def test_eval_device_dtype_preparation_prevents_inference_parameter_poisoning(self):
        model = self.ToyDetector().double().train()
        # Mirror LoRA routing: some adapter/head parameters train while base
        # parameters remain frozen, and both roles must survive validation.
        model.model[-1].projection.bias.requires_grad_(False)
        head = model.model[-1]
        original_parameters = {
            name: parameter for name, parameter in model.named_parameters()
        }
        original_grad_flags = {
            name: parameter.requires_grad
            for name, parameter in original_parameters.items()
        }
        self.assertTrue(any(original_grad_flags.values()))
        self.assertTrue(any(not flag for flag in original_grad_flags.values()))
        original_cache = {
            "shapes": head.shapes,
            "anchors": head.anchors,
            "valid_mask": head.valid_mask,
        }
        ultralytics_wrapper = SimpleNamespace(model=model)
        wrapper = SimpleNamespace(
            model=model,
            ultralytics_model=ultralytics_wrapper,
            enforce_frozen_norm_eval=lambda: None,
        )

        with _preserve_eval_state(wrapper, eval_device="cpu"):
            # The conversion needed by AutoBackend has already happened outside
            # inference_mode, so its own float() is a no-op on normal tensors.
            self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
            self.assertIs(ultralytics_wrapper.model, model)
            self.assertIs(model.fuse(), model)
            with torch.inference_mode():
                model.float()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                head.shapes = [[9, 9]]
                head.anchors = torch.randn(1, 4)
                head.valid_mask = torch.zeros(1, 1, dtype=torch.bool)
                self.assertTrue(head.anchors.is_inference())

        self.assertIs(wrapper.model, model)
        self.assertIs(ultralytics_wrapper.model, model)
        self.assertEqual(model.fuse(), "original-fuse")
        self.assertTrue(model.training)
        for name, parameter in model.named_parameters():
            self.assertIs(parameter, original_parameters[name])
            self.assertEqual(parameter.dtype, torch.float64)
            self.assertEqual(parameter.requires_grad, original_grad_flags[name])
            self.assertFalse(parameter.is_inference())
        for name, value in original_cache.items():
            self.assertIs(getattr(head, name), value)

    def test_eval_state_restores_after_validator_exception(self):
        model = self.ToyDetector().double().train()
        ultralytics_wrapper = SimpleNamespace(model=model)
        wrapper = SimpleNamespace(
            model=model,
            ultralytics_model=ultralytics_wrapper,
            enforce_frozen_norm_eval=lambda: None,
        )

        with self.assertRaisesRegex(RuntimeError, "validator failed"):
            with _preserve_eval_state(wrapper, eval_device="cpu"):
                with torch.inference_mode():
                    model.float()
                    for parameter in model.parameters():
                        parameter.requires_grad_(False)
                raise RuntimeError("validator failed")

        self.assertIs(ultralytics_wrapper.model, model)
        self.assertEqual(model.fuse(), "original-fuse")
        self.assertTrue(all(p.dtype == torch.float64 for p in model.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertTrue(all(not p.is_inference() for p in model.parameters()))


class RTDETRCheckpointCompatibilityTests(unittest.TestCase):
    class FakeRTDETRDecoder(nn.Module):
        def __init__(self, nc: int = 80):
            super().__init__()
            self.nc = nc

    @staticmethod
    def _source(model_type, head: nn.Module):
        source = model_type.__new__(model_type)
        nn.Module.__init__(source)
        source.model = nn.Sequential(nn.Identity(), head)
        source.yaml = {
            "nc": 80,
            "backbone": [[-1, 1, "Conv", [64, 3, 2]]],
            "head": [[-1, 1, "RTDETRDecoder", [80]]],
        }
        source.task = "detect"
        return source

    def test_accepts_official_detection_model_container_with_rtdetr_head(self):
        source = self._source(DetectionModel, self.FakeRTDETRDecoder())
        with mock.patch.object(
            rtdetr_lora_module, "RTDETRDecoder", self.FakeRTDETRDecoder
        ):
            descriptor = _validate_pretrained_rtdetr_source(
                source, "rtdetr-l.pt"
            )
        self.assertEqual(
            descriptor["container_representation"],
            "detection_model_with_rtdetr_decoder",
        )

    def test_accepts_native_rtdetr_detection_model_container(self):
        source = self._source(
            RTDETRDetectionModel, self.FakeRTDETRDecoder()
        )
        with mock.patch.object(
            rtdetr_lora_module, "RTDETRDecoder", self.FakeRTDETRDecoder
        ):
            descriptor = _validate_pretrained_rtdetr_source(
                source, "native-rtdetr.pt"
            )
        self.assertEqual(
            descriptor["container_representation"],
            "rtdetr_detection_model",
        )

    def test_rejects_ordinary_detection_model_without_rtdetr_head(self):
        source = self._source(DetectionModel, nn.Identity())
        with self.assertRaisesRegex(TypeError, "expected RTDETRDecoder"):
            _validate_pretrained_rtdetr_source(source, "yolo-detect.pt")

    def test_rejects_inconsistent_rtdetr_class_metadata(self):
        source = self._source(DetectionModel, self.FakeRTDETRDecoder(nc=79))
        with mock.patch.object(
            rtdetr_lora_module, "RTDETRDecoder", self.FakeRTDETRDecoder
        ):
            with self.assertRaisesRegex(TypeError, "inconsistent RT-DETR class counts"):
                _validate_pretrained_rtdetr_source(source, "broken-rtdetr.pt")

    def test_target_validation_requires_synchronized_outer_nc_for_loss(self):
        target = nn.Module()
        target.model = nn.Sequential(self.FakeRTDETRDecoder(nc=4))
        target.yaml = {"nc": 4}
        self.assertFalse(hasattr(target, "nc"))

        wrapper = object.__new__(rtdetr_lora_module.RTDETRLoRA)
        wrapper.model = target
        wrapper.num_classes = 4
        with mock.patch.object(
            rtdetr_lora_module, "RTDETRDecoder", self.FakeRTDETRDecoder
        ):
            with self.assertRaisesRegex(RuntimeError, "yaml.nc/model.nc"):
                wrapper._validate_detection_head()

            target.nc = 4
            wrapper._validate_detection_head()

            target.yaml["nc"] = 3
            with self.assertRaisesRegex(RuntimeError, "class rebuild failed"):
                wrapper._validate_detection_head()


if __name__ == "__main__":
    unittest.main()
