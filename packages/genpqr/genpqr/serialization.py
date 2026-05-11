"""Serialization helpers for GenPQR results."""

from __future__ import annotations

import json
import math
import pickle
import platform
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from genpqr._version import __version__
from genpqr.action_head_fqe import (
    ActionHeadNeuralFQEConfig,
    ActionHeadNeuralFQEFunction,
    _network_from_numpy_state,
    _state_dict_to_numpy,
)
from genpqr.api import GenPQRConfig, GenPQRResult
from genpqr.deeppqr import DeepPQRStratifiedQFunction
from genpqr.normalization import DiscreteNormalizationPolicy
from genpqr.policies import NativeDiscretePolicy, NativeGaussianPolicy
from genpqr.q_estimators import ConstantFittedQFunction
from genpqr.recovery import GenPQRRewardFunction
from genpqr.types import ActionSpaceSpec


SERIALIZATION_VERSION = 1


def save_genpqr_result(result: Any, path: str | Path, *, allow_pickle_fallback: bool = True) -> None:
    """Save a GenPQR result as a manifest plus pickle payload."""

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    try:
        manifest, arrays = _portable_payload(result)
    except TypeError:
        if not allow_pickle_fallback:
            raise
        manifest = _base_manifest(result, serialization_mode="unsafe_pickle")
        manifest["warnings"].append(
            "Result contains objects without portable serializers; loading requires allow_pickle=True."
        )
        with (target / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        try:
            with (target / "result.pkl").open("wb") as handle:
                pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            raise TypeError(
                "Result is neither portable nor pickle-serializable; use native GenPQR backends or "
                "provide top-level importable custom classes."
            ) from exc
        return
    with (target / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
    np.savez(target / "arrays.npz", **arrays)


def load_genpqr_result(path: str | Path, *, allow_pickle: bool = False) -> Any:
    """Load a result saved by :func:`save_genpqr_result`."""

    source = Path(path)
    with (source / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("serialization_version") != SERIALIZATION_VERSION:
        raise ValueError("Unsupported GenPQR serialization version.")
    mode = manifest.get("serialization_mode", "unsafe_pickle")
    if mode == "portable":
        arrays = np.load(source / "arrays.npz", allow_pickle=False)
        return _load_portable(manifest, arrays)
    if not allow_pickle:
        raise ValueError("This GenPQR result uses unsafe pickle serialization; pass allow_pickle=True to load it.")
    with (source / "result.pkl").open("rb") as handle:
        return pickle.load(handle)


def save_deep_genpqr_result(result: Any, path: str | Path, *, allow_pickle_fallback: bool = True) -> None:
    """Save a DeepGenPQR result with a DeepGenPQR manifest.

    Portable DeepGenPQR saves embed the underlying portable GenPQR payload plus
    DeepGenPQR mode/config metadata. Nonportable objects fall back to an unsafe
    pickle payload only when ``allow_pickle_fallback`` is true.
    """

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    try:
        manifest, arrays = _portable_deep_payload(result)
    except TypeError:
        if not allow_pickle_fallback:
            raise
        manifest = _deep_base_manifest(result, serialization_mode="unsafe_pickle")
        manifest["warnings"].append(
            "DeepGenPQR result contains objects without portable serializers; loading requires allow_pickle=True."
        )
        with (target / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        try:
            with (target / "result.pkl").open("wb") as handle:
                pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            raise TypeError("DeepGenPQR result is neither portable nor pickle-serializable.") from exc
        return
    with (target / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
    np.savez(target / "arrays.npz", **arrays)


def load_deep_genpqr_result(path: str | Path, *, allow_pickle: bool = False) -> Any:
    """Load a result saved by :func:`save_deep_genpqr_result`."""

    source = Path(path)
    with (source / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("serialization_version") != SERIALIZATION_VERSION:
        raise ValueError("Unsupported GenPQR serialization version.")
    if manifest.get("result_kind") != "DeepGenPQRResult":
        raise ValueError("Manifest does not describe a DeepGenPQR result.")
    mode = manifest.get("serialization_mode", "unsafe_pickle")
    if mode == "portable":
        arrays = np.load(source / "arrays.npz", allow_pickle=False)
        genpqr_result = _load_portable(manifest["genpqr_payload"], arrays)
        from genpqr.deepgenpqr import DeepGenPQRConfig, DeepGenPQRResult

        deep_config_payload = manifest.get("deepgenpqr_config", {})
        config = DeepGenPQRConfig(
            policy=deep_config_payload.get("policy", "portable_loaded"),
            q_mode=deep_config_payload.get("q_mode", manifest.get("deepgenpqr_mode", "pooled_fqe")),
            q_backend=deep_config_payload.get("q_backend", "portable_loaded"),
            anchor_backend=deep_config_payload.get("anchor_backend", "portable_loaded"),
            anchor_action=deep_config_payload.get("anchor_action", 0),
            anchor_tolerance=float(deep_config_payload.get("anchor_tolerance", 1e-8)),
            min_anchor_count=int(deep_config_payload.get("min_anchor_count", 5)),
            seed=int(deep_config_payload.get("seed", genpqr_result.config.seed)),
            n_action_samples=int(deep_config_payload.get("n_action_samples", genpqr_result.config.n_action_samples)),
            policy_config=dict(deep_config_payload.get("policy_config", {})),
            q_config=dict(deep_config_payload.get("q_config", {})),
            normalization_config=dict(deep_config_payload.get("normalization_config", {})),
            anchor_fallback=deep_config_payload.get("anchor_fallback", "error"),
        )
        diagnostics = dict(manifest.get("diagnostics", genpqr_result.diagnostics))
        genpqr_result.diagnostics.update(diagnostics)
        return DeepGenPQRResult(
            genpqr_result=genpqr_result,
            config=config,
            q_mode=config.q_mode,
            policy_backend=manifest.get("policy_backend", "portable_loaded"),
            q_backend=manifest.get("q_backend", genpqr_result.diagnostics.get("q_backend", "portable_loaded")),
            diagnostics=genpqr_result.diagnostics,
        )
    if not allow_pickle:
        raise ValueError("This DeepGenPQR result uses unsafe pickle serialization; pass allow_pickle=True to load it.")
    with (source / "result.pkl").open("rb") as handle:
        return pickle.load(handle)


def _base_manifest(result: Any, *, serialization_mode: str) -> dict[str, Any]:
    return {
        "serialization_version": SERIALIZATION_VERSION,
        "serialization_mode": serialization_mode,
        "genpqr_version": __version__,
        "python_version": platform.python_version(),
        "package": "genpqr",
        "action_space": _action_space_to_dict(result.action_space),
        "policy_class": type(result.policy).__name__,
        "q_function_class": type(result.q_function).__name__,
        "normalization_policy_class": type(result.normalization_policy).__name__,
        "backend_ids": {
            "q_backend": getattr(result.diagnostics_report, "q_backend", None)
            if getattr(result, "diagnostics_report", None) is not None
            else result.diagnostics.get("q_backend"),
        },
        "warnings": [
            "Pickle payloads are Python-environment dependent; reload with compatible optional backends installed.",
        ],
    }


def _deep_base_manifest(result: Any, *, serialization_mode: str) -> dict[str, Any]:
    return {
        "serialization_version": SERIALIZATION_VERSION,
        "serialization_mode": serialization_mode,
        "genpqr_version": __version__,
        "python_version": platform.python_version(),
        "package": "genpqr",
        "result_kind": "DeepGenPQRResult",
        "action_space": _action_space_to_dict(result.action_space),
        "deepgenpqr_mode": result.q_mode,
        "policy_backend": result.policy_backend,
        "q_backend": result.q_backend,
        "warnings": [
            "Pickle payloads are Python-environment dependent; reload with compatible optional backends installed.",
        ],
    }


def _portable_deep_payload(result: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not hasattr(result, "genpqr_result"):
        raise TypeError("save_deep_genpqr_result requires a DeepGenPQRResult.")
    gen_manifest, arrays = _portable_payload(result.genpqr_result)
    manifest = _deep_base_manifest(result, serialization_mode="portable")
    manifest["warnings"] = ["Portable DeepGenPQR serialization is supported for native portable objects only."]
    manifest["genpqr_payload"] = gen_manifest
    manifest["deepgenpqr_config"] = _deep_config_payload(result.config)
    manifest["diagnostics"] = _json_safe(result.diagnostics)
    return manifest, arrays


def _deep_config_payload(config: Any) -> dict[str, Any]:
    return {
        "q_mode": getattr(config, "q_mode", None),
        "anchor_fallback": getattr(config, "anchor_fallback", None),
        "seed": int(getattr(config, "seed", 123)),
        "n_action_samples": int(getattr(config, "n_action_samples", 32)),
        "policy": _json_safe(getattr(config, "policy", None)),
        "q_backend": _json_safe(getattr(config, "q_backend", None)),
        "anchor_backend": _json_safe(getattr(config, "anchor_backend", None)),
        "anchor_action": _json_safe(getattr(config, "anchor_action", None)),
        "anchor_tolerance": float(getattr(config, "anchor_tolerance", 1e-8)),
        "min_anchor_count": int(getattr(config, "min_anchor_count", 0)),
        "policy_config": _json_safe(getattr(config, "policy_config", {})),
        "q_config": _json_safe(getattr(config, "q_config", {})),
        "normalization_config": _json_safe(getattr(config, "normalization_config", {})),
    }


def _portable_payload(result: GenPQRResult) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrays: dict[str, np.ndarray] = {}
    manifest = _base_manifest(result, serialization_mode="portable")
    manifest["warnings"] = ["Portable serialization is supported for native GenPQR objects only."]
    manifest["config"] = {
        "n_action_samples": int(result.config.n_action_samples),
        "seed": int(result.config.seed),
    }
    manifest["diagnostics"] = _json_safe(result.diagnostics)
    manifest["policy"] = _save_policy(result.policy, arrays)
    manifest["normalization_policy"] = _save_normalization(result.normalization_policy, arrays)
    manifest["q_function"] = _save_q_function(result.q_function, arrays)
    anchor = result.reward_function.anchor_function
    if callable(anchor):
        raise TypeError("callable anchor functions are not portable.")
    manifest["anchor_function"] = float(anchor)
    return manifest, arrays


def _save_policy(policy: Any, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    if isinstance(policy, NativeDiscretePolicy):
        arrays["policy_theta"] = np.asarray(policy.theta)
        arrays["policy_input_mean"] = np.asarray(policy.input_mean)
        arrays["policy_input_std"] = np.asarray(policy.input_std)
        return {
            "class": "NativeDiscretePolicy",
            "action_space": _action_space_to_dict(policy.action_space),
            "prob_clip_min": float(policy.prob_clip_min),
            "prob_clip_max": float(policy.prob_clip_max),
        }
    if isinstance(policy, NativeGaussianPolicy):
        arrays["policy_beta"] = np.asarray(policy.beta)
        arrays["policy_input_mean"] = np.asarray(policy.input_mean)
        arrays["policy_input_std"] = np.asarray(policy.input_std)
        arrays["policy_covariance_diag"] = np.asarray(policy.covariance_diag)
        return {
            "class": "NativeGaussianPolicy",
            "action_space": _action_space_to_dict(policy.action_space),
        }
    raise TypeError(f"Policy type {type(policy).__name__} is not portable.")


def _save_normalization(policy: Any, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    if isinstance(policy, DiscreteNormalizationPolicy) and not callable(policy.probabilities):
        arrays["normalization_probabilities"] = np.asarray(policy.probabilities, dtype=np.float64)
        return {"class": "DiscreteNormalizationPolicy", "n_actions": int(policy.n_actions)}
    raise TypeError(f"Normalization policy type {type(policy).__name__} is not portable.")


def _save_q_function(q_function: Any, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    if isinstance(q_function, ConstantFittedQFunction):
        return {
            "class": "ConstantFittedQFunction",
            "action_space": _action_space_to_dict(q_function.action_space),
            "value": float(q_function.value),
            "backend": q_function.backend,
            "diagnostics": _json_safe(q_function.diagnostics),
        }
    if isinstance(q_function, DeepPQRStratifiedQFunction):
        arrays["q_coef"] = np.asarray(q_function.coef)
        arrays["q_input_mean"] = np.asarray(q_function.input_mean)
        arrays["q_input_std"] = np.asarray(q_function.input_std)
        return {
            "class": "DeepPQRStratifiedQFunction",
            "action_space": _action_space_to_dict(q_function.action_space),
            "anchor_action": int(q_function.anchor_action),
            "alpha": float(q_function.alpha),
            "diagnostics": _json_safe(q_function.diagnostics),
        }
    if isinstance(q_function, ActionHeadNeuralFQEFunction):
        arrays["q_input_mean"] = np.asarray(q_function.input_mean)
        arrays["q_input_std"] = np.asarray(q_function.input_std)
        state_keys = []
        for key, value in _state_dict_to_numpy(q_function.network):
            array_key = "q_action_head_state_" + key.replace(".", "__")
            state_keys.append((key, array_key))
            arrays[array_key] = np.asarray(value)
        return {
            "class": "ActionHeadNeuralFQEFunction",
            "action_space": _action_space_to_dict(q_function.action_space),
            "config": _action_head_config_payload(q_function.config),
            "state_keys": state_keys,
            "diagnostics": _json_safe(q_function.diagnostics),
        }
    if type(q_function).__name__ == "NeuralDeepPQRStratifiedQFunction":
        torch = q_function.torch_module
        if torch is None:
            import torch as torch_import

            torch = torch_import
        arrays["q_input_mean"] = np.asarray(q_function.input_mean)
        arrays["q_input_std"] = np.asarray(q_function.input_std)
        state_keys = []
        for key, value in q_function.model.state_dict().items():
            array_key = "q_state_" + key.replace(".", "__")
            state_keys.append((key, array_key))
            arrays[array_key] = value.detach().cpu().numpy()
        return {
            "class": "NeuralDeepPQRStratifiedQFunction",
            "action_space": _action_space_to_dict(q_function.action_space),
            "anchor_action": _serialize_anchor_action(q_function.anchor_action),
            "hidden_dims": list(q_function.hidden_dims),
            "state_keys": state_keys,
            "diagnostics": _json_safe(q_function.diagnostics),
        }
    raise TypeError(f"Q function type {type(q_function).__name__} is not portable.")


def _load_portable(manifest: dict[str, Any], arrays: Any) -> GenPQRResult:
    policy = _load_policy(manifest["policy"], arrays)
    mu = _load_normalization(manifest["normalization_policy"], arrays)
    q_function = _load_q_function(manifest["q_function"], arrays, policy)
    reward_function = GenPQRRewardFunction(
        q_function=q_function,
        normalization_policy=mu,
        anchor_function=float(manifest.get("anchor_function", 0.0)),
        n_action_samples=int(manifest.get("config", {}).get("n_action_samples", 32)),
        seed=int(manifest.get("config", {}).get("seed", 123)),
    )
    config = GenPQRConfig(
        policy="portable_loaded",
        q="portable_loaded",
        n_action_samples=int(manifest.get("config", {}).get("n_action_samples", 32)),
        seed=int(manifest.get("config", {}).get("seed", 123)),
    )
    return GenPQRResult(
        policy=policy,
        q_function=q_function,
        reward_function=reward_function,
        config=config,
        action_space=_action_space_from_dict(manifest["action_space"]),
        normalization_policy=mu,
        diagnostics=dict(manifest.get("diagnostics", {})),
        diagnostics_report=None,
    )


def _load_policy(payload: dict[str, Any], arrays: Any) -> Any:
    action_space = _action_space_from_dict(payload["action_space"])
    if payload["class"] == "NativeDiscretePolicy":
        return NativeDiscretePolicy(
            theta=arrays["policy_theta"],
            input_mean=arrays["policy_input_mean"],
            input_std=arrays["policy_input_std"],
            action_space=action_space,
            prob_clip_min=float(payload["prob_clip_min"]),
            prob_clip_max=float(payload["prob_clip_max"]),
        )
    if payload["class"] == "NativeGaussianPolicy":
        return NativeGaussianPolicy(
            beta=arrays["policy_beta"],
            input_mean=arrays["policy_input_mean"],
            input_std=arrays["policy_input_std"],
            covariance_diag=arrays["policy_covariance_diag"],
            action_space=action_space,
        )
    raise ValueError(f"Unsupported portable policy class {payload['class']}.")


def _load_normalization(payload: dict[str, Any], arrays: Any) -> Any:
    if payload["class"] == "DiscreteNormalizationPolicy":
        return DiscreteNormalizationPolicy(
            n_actions=int(payload["n_actions"]),
            probabilities=arrays["normalization_probabilities"],
        )
    raise ValueError(f"Unsupported portable normalization policy class {payload['class']}.")


def _load_q_function(payload: dict[str, Any], arrays: Any, policy: Any) -> Any:
    if payload["class"] == "ConstantFittedQFunction":
        return ConstantFittedQFunction(
            action_space=_action_space_from_dict(payload["action_space"]),
            value=float(payload["value"]),
            backend=payload.get("backend", "constant_q"),
            diagnostics=dict(payload.get("diagnostics", {})),
        )
    action_space = _action_space_from_dict(payload["action_space"])
    if payload["class"] == "DeepPQRStratifiedQFunction":
        return DeepPQRStratifiedQFunction(
            coef=arrays["q_coef"],
            input_mean=arrays["q_input_mean"],
            input_std=arrays["q_input_std"],
            policy=policy,
            action_space=action_space,
            anchor_action=int(payload["anchor_action"]),
            alpha=float(payload["alpha"]),
            diagnostics=dict(payload.get("diagnostics", {})),
        )
    if payload["class"] == "ActionHeadNeuralFQEFunction":
        config = ActionHeadNeuralFQEConfig(**dict(payload["config"]))
        state_items = [
            (key, np.asarray(arrays[array_key]))
            for key, array_key in payload["state_keys"]
        ]
        network = _network_from_numpy_state(
            state_items,
            action_space=action_space,
            state_dim=int(np.asarray(arrays["q_input_mean"]).shape[0]),
            config=config,
        )
        target_network = _network_from_numpy_state(
            state_items,
            action_space=action_space,
            state_dim=int(np.asarray(arrays["q_input_mean"]).shape[0]),
            config=config,
        )
        return ActionHeadNeuralFQEFunction(
            network=network,
            target_network=target_network,
            action_space=action_space,
            input_mean=arrays["q_input_mean"],
            input_std=arrays["q_input_std"],
            config=config,
            diagnostics=dict(payload.get("diagnostics", {})),
            policy=policy,
        )
    if payload["class"] == "NeuralDeepPQRStratifiedQFunction":
        import torch
        from torch import nn
        from genpqr.neural_deeppqr import NeuralDeepPQRStratifiedQFunction, _build_mlp

        hidden_dims = tuple(int(width) for width in payload["hidden_dims"])
        model = _build_mlp(nn, arrays["q_input_mean"].shape[0], hidden_dims)
        state_dict = {}
        for key, array_key in payload["state_keys"]:
            state_dict[key] = torch.as_tensor(arrays[array_key])
        model.load_state_dict(state_dict)
        model.eval()
        return NeuralDeepPQRStratifiedQFunction(
            model=model,
            input_mean=arrays["q_input_mean"],
            input_std=arrays["q_input_std"],
            policy=policy,
            action_space=action_space,
            anchor_action=payload["anchor_action"],
            hidden_dims=hidden_dims,
            diagnostics=dict(payload.get("diagnostics", {})),
            torch_module=torch,
        )
    raise ValueError(f"Unsupported portable Q function class {payload['class']}.")


def _action_space_to_dict(action_space: ActionSpaceSpec) -> dict[str, int | str | None]:
    return {
        "kind": action_space.kind,
        "n_actions": action_space.n_actions,
        "action_dim": action_space.action_dim,
    }


def _serialize_anchor_action(anchor_action: Any) -> int | float | list[float] | list[list[float]]:
    if callable(anchor_action):
        raise TypeError("Callable neural DeepPQR anchor actions are not portable.")
    arr = np.asarray(anchor_action, dtype=np.float64)
    if arr.ndim == 0:
        value = float(arr.item())
        return int(value) if np.isclose(value, int(value)) else value
    if arr.ndim <= 2:
        return arr.tolist()
    raise TypeError("Neural DeepPQR anchor_action must be scalar, 1D, or 2D to serialize portably.")


def _action_head_config_payload(config: ActionHeadNeuralFQEConfig) -> dict[str, Any]:
    payload = {field.name: getattr(config, field.name) for field in fields(config)}
    return _json_safe(payload)


def _action_space_from_dict(payload: dict[str, Any]) -> ActionSpaceSpec:
    if payload["kind"] == "discrete":
        return ActionSpaceSpec.discrete(int(payload["n_actions"]))
    return ActionSpaceSpec.continuous(int(payload["action_dim"]))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(val) for val in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)
