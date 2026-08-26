# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lightweight counters for selective activation offload diagnostics."""

from dataclasses import dataclass

from ...utils import logging


logger = logging.get_logger(__name__)


@dataclass
class ActivationOffloadStats:
    """Runtime counters reported via :meth:`log_summary`."""

    num_matched_module_calls: int = 0
    num_offloaded_tensors: int = 0
    num_prefetch_hits: int = 0
    num_checkpoint_prefetch_groups: int = 0
    num_ondemand_restores: int = 0
    num_parameter_views_skipped: int = 0
    num_threshold_fallback_offloads: int = 0
    num_threshold_keep_on_gpu: int = 0
    num_ignored_tensors: int = 0
    offloaded_bytes: int = 0
    restored_bytes: int = 0
    parameter_view_bytes_skipped: int = 0
    peak_pinned_bytes: int = 0

    def log(self) -> None:
        if self.num_offloaded_tensors == 0 and self.num_threshold_fallback_offloads == 0:
            logger.info_rank0("Selective activation offload: no tensors were offloaded.")
            return

        logger.info_rank0(
            "Selective activation offload summary: "
            f"matched_module_calls={self.num_matched_module_calls}, "
            f"offloaded_tensors={self.num_offloaded_tensors}, "
            f"offloaded_bytes={self.offloaded_bytes}, "
            f"prefetch_hits={self.num_prefetch_hits}, "
            f"checkpoint_prefetch_groups={self.num_checkpoint_prefetch_groups}, "
            f"ondemand_restores={self.num_ondemand_restores}, "
            f"parameter_views_skipped={self.num_parameter_views_skipped}, "
            f"parameter_view_bytes_skipped={self.parameter_view_bytes_skipped}, "
            f"threshold_fallback_offloads={self.num_threshold_fallback_offloads}, "
            f"threshold_keep_on_gpu={self.num_threshold_keep_on_gpu}, "
            f"ignored_tensors={self.num_ignored_tensors}, "
            f"restored_bytes={self.restored_bytes}, "
            f"peak_pinned_bytes={self.peak_pinned_bytes}"
        )
