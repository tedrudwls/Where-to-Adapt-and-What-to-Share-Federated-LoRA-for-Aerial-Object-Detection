"""Unit tests for fixed A/B factor-sharing routes and checkpoint compatibility."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

from main import _runtime_arguments
from models.lora import inject_lora_linear
from models.rtdetr_lora import RTDETRLoRA
from trainers.fl_server import (
    FEDERATED_CHECKPOINT_SCHEMA,
    FLServer,
    _checkpoint_local_personalized_states,
    _experiment_manifest,
)


class _ToyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Linear(5, 4, bias=False)
        self.score_head = nn.Linear(4, 4)


def _wrapper(method: str) -> RTDETRLoRA:
    model = _ToyDetector()
    inject_lora_linear(model, ["feature"], rank=2, alpha=4.0, dropout=0.0)
    wrapper = object.__new__(RTDETRLoRA)
    wrapper.ft_mode = method
    wrapper.model = model
    return wrapper


class FactorSharingRoutingTests(unittest.TestCase):
    def test_fedsa_and_fixed_share_b_route_complementary_factors(self):
        cases = (
            (
                "fedsa_lora",
                "A",
                "B",
                "global_A_plus_global_task_head__local_B",
            ),
            (
                "fixed_share_b_lora",
                "B",
                "A",
                "global_B_plus_global_task_head__local_A",
            ),
        )
        for method, shared_role, local_role, policy in cases:
            with self.subTest(method=method):
                wrapper = _wrapper(method)
                shared = wrapper.get_aggregation_state()
                local_before = wrapper.get_local_personalized_state()

                shared_adapter_keys = {
                    key for key in shared if key.endswith((".lora_A", ".lora_B"))
                }
                self.assertTrue(shared_adapter_keys)
                self.assertTrue(
                    all(key.endswith(f".lora_{shared_role}") for key in shared_adapter_keys)
                )
                self.assertTrue(local_before)
                self.assertTrue(
                    all(key.endswith(f".lora_{local_role}") for key in local_before)
                )
                self.assertTrue(any("score_head" in key for key in shared))
                self.assertEqual(wrapper.shared_lora_factor_role(), shared_role)
                self.assertEqual(wrapper.local_personalized_factor_role(), local_role)
                self.assertEqual(wrapper.federated_payload_policy(), policy)
                if local_role == "A":
                    self.assertEqual(
                        set(wrapper.get_local_A_state()), set(local_before)
                    )
                    with self.assertRaisesRegex(RuntimeError, "does not retain LoRA B"):
                        wrapper.get_local_B_state()
                else:
                    self.assertEqual(
                        set(wrapper.get_local_B_state()), set(local_before)
                    )
                    with self.assertRaisesRegex(RuntimeError, "does not retain LoRA A"):
                        wrapper.get_local_A_state()

                changed_shared = {
                    key: value + torch.ones_like(value) for key, value in shared.items()
                }
                wrapper.set_aggregation_state(changed_shared)
                local_after = wrapper.get_local_personalized_state()
                self.assertEqual(set(local_before), set(local_after))
                for key in local_before:
                    self.assertTrue(torch.equal(local_before[key], local_after[key]), key)

                counts = wrapper.count_params()
                self.assertEqual(
                    counts["communication_params"],
                    sum(tensor.numel() for tensor in changed_shared.values()),
                )

    def test_shared_lora_has_no_personalized_state(self):
        wrapper = _wrapper("lora")
        self.assertEqual(wrapper.get_local_personalized_state(), {})
        self.assertIsNone(wrapper.local_personalized_factor_role())
        with self.assertRaisesRegex(RuntimeError, "no client-local"):
            wrapper.set_local_personalized_state({"unexpected": torch.zeros(1)})

    def test_server_accepts_fixed_share_b_method(self):
        server = FLServer("fixed_share_b_lora", num_clients=2)
        self.assertEqual(server.fl_method, "fixed_share_b_lora")


class FactorSharingCheckpointCompatibilityTests(unittest.TestCase):
    def test_schema4_fedsa_local_B_is_normalized(self):
        states = [{"adapter.lora_B": torch.zeros(2, 2)} for _ in range(3)]
        role, restored = _checkpoint_local_personalized_states(
            {"schema_version": 4, "local_B_states": states},
            SimpleNamespace(fl_method="fedsa_lora"),
        )
        self.assertEqual(role, "B")
        self.assertIs(restored, states)

    def test_schema4_cannot_represent_fixed_share_b(self):
        with self.assertRaisesRegex(ValueError, "schema 5"):
            _checkpoint_local_personalized_states(
                {"schema_version": 4, "local_B_states": None},
                SimpleNamespace(fl_method="fixed_share_b_lora"),
            )

    def test_schema5_preserves_generic_local_A_role(self):
        states = [{"adapter.lora_A": torch.zeros(2, 2)} for _ in range(3)]
        role, restored = _checkpoint_local_personalized_states(
            {
                "schema_version": FEDERATED_CHECKPOINT_SCHEMA,
                "local_lora_factor_role": "A",
                "local_personalized_states": states,
            },
            SimpleNamespace(fl_method="fixed_share_b_lora"),
        )
        self.assertEqual(role, "A")
        self.assertIs(restored, states)


class ExperimentManifestTests(unittest.TestCase):
    @staticmethod
    def _args(partition: str) -> SimpleNamespace:
        return SimpleNamespace(
            partition=partition,
            dirichlet_alpha=0.4,
            fl_rounds=20,
            local_epochs=5,
            close_mosaic_epochs=10,
        )

    def test_iid_manifest_does_not_record_argparse_dirichlet_default(self):
        manifest = _experiment_manifest(self._args("iid"))
        self.assertIsNone(manifest["dirichlet_alpha"])

    def test_dirichlet_manifest_retains_configured_alpha(self):
        manifest = _experiment_manifest(self._args("dirichlet"))
        self.assertEqual(manifest["dirichlet_alpha"], 0.4)

    def test_iid_runtime_arguments_do_not_record_unused_alpha(self):
        arguments = _runtime_arguments(self._args("iid"))
        self.assertIsNone(arguments["dirichlet_alpha"])

    def test_dirichlet_runtime_arguments_retain_configured_alpha(self):
        arguments = _runtime_arguments(self._args("dirichlet"))
        self.assertEqual(arguments["dirichlet_alpha"], 0.4)


if __name__ == "__main__":
    unittest.main()
