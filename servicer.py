# Copyright 2022 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
from abc import ABC
from typing import Optional

from dlrover.python.common import comm
from dlrover.python.common.comm import BaseRequest, BaseResponse
from dlrover.python.common.constants import (
    CustomMetricKeys,
    TrainingExceptionLevel,
)
from dlrover.python.common.event.context import JobEventContext
from dlrover.python.common.event.reporter import get_event_reporter
from dlrover.python.common.event.train_event import TrainEventName
from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.diagnosis_data import DiagnosisData
from dlrover.python.master.stats.job_collector import JobMetricCollector
from dlrover.python.unified.backend.elastic.manager import ElasticManager

_event_context = JobEventContext.singleton_instance()


class MasterServicer(ABC):
    """Master service base class."""

    def __init__(
        self,
        job_manager: ElasticManager,
        job_metric_collector: Optional[JobMetricCollector] = None,
    ):
        self._core = job_manager
        self._job_metric_collector = job_metric_collector
        self._event_reporter = get_event_reporter()

        # preload module for class reflection
        self._diagnosis_data_module = importlib.import_module(
            "dlrover.python.diagnosis.common.diagnosis_data"
        )

    def get(self, request, _=None):
        node_type = request.node_type
        node_id = request.node_id
        req_message = comm.deserialize_message(request.data)

        response = BaseResponse()
        if not req_message:
            return response
        message = None
        if isinstance(req_message, comm.TaskRequest):
            raise NotImplementedError("deprecated, TF backend only")
        elif isinstance(req_message, comm.ShardCheckpointRequest):
            raise NotImplementedError("deprecated, TF backend only")
        elif isinstance(req_message, comm.ClusterVersionRequest):
            raise NotImplementedError("deprecated, TF backend only")
        elif isinstance(req_message, comm.RunningNodesRequest):
            raise NotImplementedError("deprecated, new RDZV based on Ray")
        elif isinstance(req_message, comm.JoinRendezvousRequest):
            raise NotImplementedError("deprecated, new RDZV based on Ray")
        elif isinstance(req_message, comm.WaitingNodeNumRequest):
            raise NotImplementedError("deprecated, new RDZV based on Ray")
        elif isinstance(req_message, comm.NetworkReadyRequest):
            raise NotImplementedError("deprecated, new unified Node-check")
        elif isinstance(req_message, comm.StragglerExistRequest):
            raise NotImplementedError("deprecated, new unified Node-check")
        elif isinstance(req_message, comm.CommWorldRequest):
            raise NotImplementedError("deprecated, new RDZV based on Ray")
        elif isinstance(req_message, comm.KeyValuePair):
            raise NotImplementedError("deprecated, new RDZV based on Ray")
        elif isinstance(req_message, comm.KeyValuePairs):
            raise NotImplementedError("deprecated, new RDZV based on Ray")
        elif isinstance(req_message, comm.PsNodesRequest):
            raise NotImplementedError("deprecated, TF backend only")
        elif isinstance(req_message, comm.TrainingStatusRequest):
            raise NotImplementedError("deprecated, TF backend only")
        elif isinstance(req_message, comm.ParallelConfigRequest):
            raise NotImplementedError(
                "Currently not support AutoTunning/ElasticDataLoader"
            )
        elif isinstance(req_message, comm.CheckHardwareResetRequest):
            raise NotImplementedError("deprecated, useless in unified")
        elif isinstance(req_message, comm.SyncTrainingPort):
            raise NotImplementedError("deprecated, sync hccl port for NPU")
        elif isinstance(req_message, comm.ElasticRunConfigRequest):
            raise NotImplementedError("deprecated, useless in unified")
        elif isinstance(req_message, comm.PreCheckRequest):
            raise NotImplementedError("deprecated, useless in unified")
        elif isinstance(req_message, comm.HeartBeat):
            raise NotImplementedError("deprecated, useless in unified")

        if message:
            response.data = message.serialize()
        return response

    def report(self, request, _=None):
        node_type = request.node_type
        node_id = request.node_id
        message = comm.deserialize_message(request.data)

        response = BaseResponse()
        if not message:
            return response

        success = False
        if isinstance(message, comm.DatasetShardParams):
            raise NotImplementedError("deprecated, TF backend only")
        elif isinstance(message, comm.ResourceStats):
            success = self._update_node_resource_usage(
                node_type, node_id, message
            )
        elif isinstance(message, comm.ModelInfo):
            raise NotImplementedError("deprecated, TF backend only")
        elif isinstance(message, comm.GlobalStep):
            raise NotImplementedError("deprecated, TF backend only")
        elif isinstance(message, comm.ShardCheckpoint):
            raise NotImplementedError("deprecated, TF backend only")
        elif isinstance(message, comm.TaskResult):
            raise NotImplementedError("deprecated, TF backend only")
        elif isinstance(message, comm.ClusterVersion):
            raise NotImplementedError("deprecated, TF backend only")
        elif isinstance(message, comm.NodeAddress):
            raise NotImplementedError("deprecated, TF backend only")
        elif isinstance(message, comm.NodeEvent):
            raise NotImplementedError("deprecated, useless in unified")
        elif isinstance(message, comm.AtorchEvent):
            success = self._handle_reported_atorch_event(message)  # metric
        elif isinstance(message, comm.SyncJoin):
            raise NotImplementedError("deprecated, TF backend only")
        elif isinstance(message, comm.SyncFinish):
            raise NotImplementedError("deprecated, TF backend only")
        elif isinstance(message, comm.SyncBarrier):
            raise NotImplementedError("deprecated, TF backend only")
        elif isinstance(message, comm.NodeFailure):
            success = self._report_failure(
                node_type, node_id, message
            )  # diagnosis
        elif isinstance(message, comm.RendezvousParams):
            raise NotImplementedError("deprecated, new RDZV based on Ray")
        elif isinstance(message, comm.PsReady):
            raise NotImplementedError("deprecated, TF backend only")
        elif isinstance(message, comm.KeyValuePair):
            raise NotImplementedError("deprecated, new RDZV based on Ray")
        elif isinstance(message, comm.KeyValuePairs):
            raise NotImplementedError("deprecated, new RDZV based on Ray")
        elif isinstance(message, comm.ParallelConfig):
            raise NotImplementedError(
                "Currently not support AutoTunning/ElasticDataLoader"
            )
        elif isinstance(message, comm.NodeCheckpointState):
            raise NotImplementedError("Currently not support AsyncCheckpoint")
        elif isinstance(message, comm.DiagnosisReportData):
            success = self._report_node_diagnosis_data(message)  # XPU Timer
        elif isinstance(message, comm.Event):
            success = self._report_event(message)  # metric

        response.success = success
        return response

    def _update_node_resource_usage(
        self, node_type, node_id, metrics: comm.ResourceStats
    ):
        logger.debug(
            f"Update resource usage for {node_type}-{node_id},"
            f"cpu={metrics.cpu}, memory={metrics.memory},"
            f"gpu_stats={metrics.gpu_stats}"
        )
        if self._core:
            self._core.update_node_resource_usage(
                node_type,
                node_id,
                metrics.cpu,
                metrics.memory,
                metrics.gpu_stats,
            )
        return True

    def _handle_reported_atorch_event(self, message: comm.AtorchEvent):
        if message.name == TrainEventName.TRAIN_EVT_STEP:
            logger.debug(f"Add step event: {message}")
            _event_context.train_steps.add_step_event(message)
        elif message.name == TrainEventName.TRAIN_EVT_FLASH_CKPT:
            logger.debug(f"Add ckpt event: {message}")
            _event_context.ckpt_steps.add_ckpt_event(message)

        return True

    def _report_failure(self, node_type, node_id, message: comm.NodeFailure):
        self._core.handle_training_failure(
            node_type,
            node_id,
            message.restart_count,
            message.error_data,
            message.level,
        )
        if message.level == TrainingExceptionLevel.RDZV_ERROR:
            custom_data = {
                CustomMetricKeys.TRAINING_ERROR_LEVEL: message.level,
                CustomMetricKeys.ERROR_CONTENT: message.error_data,
            }
            if self._job_metric_collector:
                self._job_metric_collector.collect_custom_data(custom_data)
        return True

    def _report_node_diagnosis_data(self, message: comm.DiagnosisReportData):
        if True:
            data_cls: Optional[DiagnosisData] = getattr(
                self._diagnosis_data_module,
                message.data_cls,
            )
            if data_cls is None:
                logger.warning(
                    f"Invalid diagnosis report data type: {message.data_cls}"
                )
                return False
            data_obj = data_cls.from_json(message.data_content)
            self._core.diagnosis.collect_diagnosis_data(data_obj)
        return True

    def _report_event(self, message: comm.Event):
        if self._event_reporter:
            self._event_reporter.report(
                message.event_type,
                message.instance,
                message.action,
                message.msg,
                message.labels,
            )
        return True


class RayMasterServicer(MasterServicer):
    """Master service with ray implementation."""

    def agent_report(self, request):
        return self.report(BaseRequest.from_json(request))

    def agent_get(self, request):
        return self.get(BaseRequest.from_json(request))
