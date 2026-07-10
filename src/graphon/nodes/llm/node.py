from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import time
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, assert_never, override

from graphon.entities.graph_init_params import GraphInitParams
from graphon.enums import (
    BuiltinNodeTypes,
    WorkflowNodeExecutionMetadataKey,
    WorkflowNodeExecutionStatus,
)
from graphon.file import file_manager
from graphon.file.enums import FileType
from graphon.file.models import File
from graphon.http import HttpClientProtocol
from graphon.model_runtime.entities.llm_entities import (
    LLMPollingConfig,
    LLMPollingResult,
    LLMPollingStatus,
    LLMResult,
    LLMResultChunk,
    LLMResultChunkWithStructuredOutput,
    LLMResultWithStructuredOutput,
    LLMStructuredOutput,
    LLMUsage,
)
from graphon.model_runtime.entities.message_entities import (
    AssistantPromptMessage,
    ImagePromptMessageContent,
    MultiModalPromptMessageContent,
    PromptMessage,
    PromptMessageContentType,
    PromptMessageContentUnionTypes,
    PromptMessageRole,
    SystemPromptMessage,
    TextPromptMessageContent,
    UserPromptMessage,
)
from graphon.model_runtime.entities.model_entities import ModelPropertyKey
from graphon.model_runtime.memory.prompt_message_memory import PromptMessageMemory
from graphon.model_runtime.utils.encoders import jsonable_encoder
from graphon.node_events.base import (
    NodeEventBase,
    NodeRunResult,
)
from graphon.node_events.node import (
    ModelInvokeCompletedEvent,
    ModelPollingProgressEvent,
    RunRetrieverResourceEvent,
    StreamChunkEvent,
    StreamCompletedEvent,
    StreamReasoningEvent,
)
from graphon.nodes.base.entities import VariableSelector
from graphon.nodes.base.node import Node
from graphon.nodes.base.variable_template_parser import VariableTemplateParser
from graphon.nodes.llm.reasoning import (
    FilterPiece,
    ThinkStreamFilter,
    extract_stream_reasoning,
    split_reasoning,
)
from graphon.nodes.llm.runtime_protocols import (
    LLMPollingCapableProtocol,
    LLMProtocol,
    PromptMessageSerializerProtocol,
    RetrieverAttachmentLoaderProtocol,
)
from graphon.prompt_entities import MemoryConfig
from graphon.runtime.graph_runtime_state import GraphRuntimeState
from graphon.runtime.variable_pool import VariablePool
from graphon.template_rendering import Jinja2TemplateRenderer, TemplateRenderError
from graphon.variables.segments import (
    ArrayFileSegment,
    ArraySegment,
    FileSegment,
    NoneSegment,
    ObjectSegment,
    StringSegment,
)
from graphon.variables.template_resolution import convert_template

from . import llm_utils
from .entities import (
    LLMNodeChatModelMessage,
    LLMNodeCompletionModelPromptTemplate,
    LLMNodeData,
)
from .exc import (
    InvalidContextStructureError,
    InvalidVariableTypeError,
    LLMNodeError,
    MemoryRolePrefixRequiredError,
    NoPromptFoundError,
    TemplateTypeNotSupportError,
    VariableNotFoundError,
)
from .file_saver import LLMFileSaver


logger = logging.getLogger(__name__)


_MULTIMODAL_OUTPUT_FILE_TYPES: Mapping[PromptMessageContentType, FileType] = {
    PromptMessageContentType.IMAGE: FileType.IMAGE,
    PromptMessageContentType.VIDEO: FileType.VIDEO,
    PromptMessageContentType.AUDIO: FileType.AUDIO,
    PromptMessageContentType.DOCUMENT: FileType.DOCUMENT,
}


@dataclass
class _CollectedRunContext:
    context: str | None = None
    context_files: list[File] = field(default_factory=list)


@dataclass
class _PreparedRunPrompt:
    prompt_messages: Sequence[PromptMessage] = field(default_factory=tuple)
    stop: Sequence[str] | None = None
    model_instance: LLMProtocol | None = None


@dataclass
class _StreamingInvokeState:
    usage: LLMUsage = field(default_factory=LLMUsage.empty_usage)
    finish_reason: str | None = None
    full_text_buffer: io.StringIO = field(default_factory=io.StringIO)
    start_time: float = 0.0
    first_token_time: float | None = None
    has_content: bool = False
    structured_output: dict[str, Any] | None = None
    # None in "tagged" mode (stream raw tokens); a filter in "separated" mode.
    text_filter: ThinkStreamFilter | None = None
    # Set once reasoning has streamed, to emit one terminal marker at stream end.
    reasoning_started: bool = False


class LLMNode(Node[LLMNodeData]):
    node_type = BuiltinNodeTypes.LLM

    # Instance attributes specific to LLMNode.
    # Output variable for file
    _file_outputs: list[File]

    _llm_file_saver: LLMFileSaver
    _retriever_attachment_loader: RetrieverAttachmentLoaderProtocol | None
    _prompt_message_serializer: PromptMessageSerializerProtocol
    _jinja2_template_renderer: Jinja2TemplateRenderer | None
    _model_instance: LLMProtocol
    _memory: PromptMessageMemory | None
    _default_query_selector: tuple[str, ...] | None

    @override
    def __init__(
        self,
        node_id: str,
        data: LLMNodeData,
        *,
        graph_init_params: GraphInitParams,
        graph_runtime_state: GraphRuntimeState,
        credentials_provider: object | None = None,
        model_factory: object | None = None,
        model_instance: LLMProtocol,
        http_client: HttpClientProtocol | None = None,
        memory: PromptMessageMemory | None = None,
        llm_file_saver: LLMFileSaver,
        prompt_message_serializer: PromptMessageSerializerProtocol,
        retriever_attachment_loader: RetrieverAttachmentLoaderProtocol | None = None,
        jinja2_template_renderer: Jinja2TemplateRenderer | None = None,
        default_query_selector: Sequence[str] | None = None,
    ) -> None:
        super().__init__(
            node_id=node_id,
            data=data,
            graph_init_params=graph_init_params,
            graph_runtime_state=graph_runtime_state,
        )
        # LLM file outputs, used for MultiModal outputs.
        self._file_outputs = []

        _ = credentials_provider, model_factory, http_client
        self._model_instance = model_instance
        self._memory = memory

        self._llm_file_saver = llm_file_saver
        self._prompt_message_serializer = prompt_message_serializer
        self._retriever_attachment_loader = retriever_attachment_loader
        self._jinja2_template_renderer = jinja2_template_renderer
        self._default_query_selector = (
            tuple(default_query_selector)
            if default_query_selector is not None
            else None
        )

    @classmethod
    @override
    def version(cls) -> str:
        return "1"

    @override
    def _run(self) -> Generator:
        node_inputs: dict[str, Any] = {}
        process_data: dict[str, Any] = {}
        usage_holder = {"value": LLMUsage.empty_usage()}

        try:
            prepared_prompt = _PreparedRunPrompt()
            yield from self._prepare_run_prompt(
                node_inputs=node_inputs,
                prepared_prompt=prepared_prompt,
            )
            model_instance = self._require_model_instance(
                prepared_prompt=prepared_prompt,
            )

            yield from self._yield_run_completion(
                node_inputs=node_inputs,
                process_data=process_data,
                usage_holder=usage_holder,
                prompt_messages=prepared_prompt.prompt_messages,
                stop=prepared_prompt.stop,
                model_provider=model_instance.provider,
                model_name=model_instance.model_name,
            )
        except ValueError as exc:
            yield StreamCompletedEvent(
                node_run_result=NodeRunResult(
                    status=WorkflowNodeExecutionStatus.FAILED,
                    error=str(exc),
                    inputs=node_inputs,
                    process_data=process_data,
                    error_type=type(exc).__name__,
                    llm_usage=usage_holder["value"],
                ),
            )
        except Exception as exc:
            logger.exception("error while executing llm node")
            yield StreamCompletedEvent(
                node_run_result=NodeRunResult(
                    status=WorkflowNodeExecutionStatus.FAILED,
                    error=str(exc),
                    inputs=node_inputs,
                    process_data=process_data,
                    error_type=type(exc).__name__,
                    llm_usage=usage_holder["value"],
                ),
            )

    def _prepare_run_prompt(
        self,
        *,
        node_inputs: dict[str, Any],
        prepared_prompt: _PreparedRunPrompt,
    ) -> Generator[
        NodeEventBase,
        None,
        None,
    ]:
        self.node_data.prompt_template = self._transform_chat_messages(
            self.node_data.prompt_template,
        )
        inputs = self._fetch_inputs(node_data=self.node_data)
        inputs.update(self._fetch_jinja_inputs(node_data=self.node_data))

        files = (
            llm_utils.fetch_files(
                variable_pool=self.graph_runtime_state.variable_pool,
                selector=self.node_data.vision.configs.variable_selector,
            )
            if self.node_data.vision.enabled
            else []
        )
        if files:
            node_inputs["#files#"] = [file.to_dict() for file in files]

        collected_context = _CollectedRunContext()
        yield from self._collect_run_context(
            node_inputs=node_inputs,
            collected_context=collected_context,
        )
        model_instance = self._prepare_model_instance()
        node_inputs.update(
            llm_utils.build_model_identity_inputs(model_instance=model_instance),
        )
        prompt_messages, stop = LLMNode.fetch_prompt_messages(
            sys_query=self._resolve_memory_query(),
            sys_files=files,
            context=collected_context.context or "",
            memory=self._memory,
            model_instance=model_instance,
            stop=model_instance.stop,
            prompt_template=self.node_data.prompt_template,
            memory_config=self.node_data.memory,
            vision_enabled=self.node_data.vision.enabled,
            vision_detail=self.node_data.vision.configs.detail,
            variable_pool=self.graph_runtime_state.variable_pool,
            jinja2_variables=self.node_data.prompt_config.jinja2_variables,
            context_files=collected_context.context_files,
            jinja2_template_renderer=self._jinja2_template_renderer,
        )
        prepared_prompt.prompt_messages = prompt_messages
        prepared_prompt.stop = stop
        prepared_prompt.model_instance = model_instance

    @staticmethod
    def _require_model_instance(
        *,
        prepared_prompt: _PreparedRunPrompt,
    ) -> LLMProtocol:
        if prepared_prompt.model_instance is None:
            msg = "model instance was not prepared"
            raise AssertionError(msg)
        return prepared_prompt.model_instance

    def _collect_run_context(
        self,
        *,
        node_inputs: dict[str, Any],
        collected_context: _CollectedRunContext,
    ) -> Generator[NodeEventBase, None, None]:
        context_generator = self._fetch_context(node_data=self.node_data)
        if context_generator is not None:
            for event in context_generator:
                collected_context.context = event.context
                collected_context.context_files = event.context_files or []
                yield event

        if collected_context.context:
            node_inputs["#context#"] = collected_context.context
        if collected_context.context_files:
            node_inputs["#context_files#"] = [
                file.model_dump() for file in collected_context.context_files
            ]

    def _prepare_model_instance(self) -> LLMProtocol:
        model_instance = self._model_instance
        model_instance.parameters = llm_utils.resolve_completion_params_variables(
            model_instance.parameters,
            self.graph_runtime_state.variable_pool,
        )
        return model_instance

    def _resolve_memory_query(self) -> str | None:
        if not self.node_data.memory:
            return None

        query = self.node_data.memory.query_prompt_template
        if query:
            return query
        if not self._default_query_selector:
            return None

        query_variable = self.graph_runtime_state.variable_pool.get(
            self._default_query_selector,
        )
        return query_variable.text if query_variable else None

    def _yield_run_completion(
        self,
        *,
        node_inputs: dict[str, Any],
        process_data: dict[str, Any],
        usage_holder: dict[str, LLMUsage],
        prompt_messages: Sequence[PromptMessage],
        stop: Sequence[str] | None,
        model_provider: Any,
        model_name: str,
    ) -> Generator[NodeEventBase, None, None]:
        generator = self._invoke_llm_for_run(
            prompt_messages=prompt_messages,
            stop=stop,
        )
        usage = LLMUsage.empty_usage()
        finish_reason = None
        reasoning_content = ""
        clean_text = ""
        structured_output: LLMStructuredOutput | None = None

        for event in generator:
            if isinstance(
                event,
                StreamChunkEvent | StreamReasoningEvent | ModelPollingProgressEvent,
            ):
                yield event
                continue

            if isinstance(event, LLMStructuredOutput):
                structured_output = event
                continue

            if not isinstance(event, ModelInvokeCompletedEvent):
                continue

            usage = event.usage
            usage_holder["value"] = usage
            finish_reason = event.finish_reason
            reasoning_content = event.reasoning_content or ""
            clean_text = self._extract_clean_text(event.text)
            if event.structured_output:
                structured_output = LLMStructuredOutput(
                    structured_output=event.structured_output,
                )
            break

        node_inputs.update(
            llm_utils.build_model_identity_inputs(model_instance=self._model_instance),
        )
        process_data.update(
            self._build_process_data(
                prompt_messages=prompt_messages,
                usage=usage,
                finish_reason=finish_reason,
                model_provider=model_provider,
                model_name=model_name,
            ),
        )
        outputs = self._build_run_outputs(
            clean_text=clean_text,
            usage=usage,
            finish_reason=finish_reason,
            reasoning_content=reasoning_content,
            structured_output=structured_output,
        )
        yield StreamChunkEvent(
            selector=[self._node_id, "text"],
            chunk="",
            is_final=True,
        )
        yield StreamCompletedEvent(
            node_run_result=NodeRunResult(
                status=WorkflowNodeExecutionStatus.SUCCEEDED,
                inputs=node_inputs,
                process_data=process_data,
                outputs=outputs,
                metadata={
                    WorkflowNodeExecutionMetadataKey.TOTAL_TOKENS: usage.total_tokens,
                    WorkflowNodeExecutionMetadataKey.TOTAL_PRICE: usage.total_price,
                    WorkflowNodeExecutionMetadataKey.CURRENCY: usage.currency,
                },
                llm_usage=usage,
            ),
        )

    def _invoke_llm_for_run(
        self,
        *,
        prompt_messages: Sequence[PromptMessage],
        stop: Sequence[str] | None,
    ) -> Generator[NodeEventBase | LLMStructuredOutput, None, None]:
        polling_model = self._polling_model_instance()
        if polling_model is None:
            return LLMNode.invoke_llm(
                model_instance=self._model_instance,
                prompt_messages=prompt_messages,
                stop=stop,
                structured_output_enabled=self.node_data.structured_output_enabled,
                structured_output=self.node_data.structured_output,
                file_saver=self._llm_file_saver,
                file_outputs=self._file_outputs,
                node_id=self._node_id,
                reasoning_format=self.node_data.reasoning_format,
            )

        return self._invoke_llm_with_polling(
            polling_model=polling_model,
            prompt_messages=prompt_messages,
            stop=stop,
        )

    def _polling_model_instance(self) -> LLMPollingCapableProtocol | None:
        if isinstance(self._model_instance, LLMPollingCapableProtocol):
            return self._model_instance
        return None

    def _invoke_llm_with_polling(
        self,
        *,
        polling_model: LLMPollingCapableProtocol,
        prompt_messages: Sequence[PromptMessage],
        stop: Sequence[str] | None,
    ) -> Generator[NodeEventBase | LLMStructuredOutput, None, None]:
        config = self._polling_config(polling_model)
        model_parameters = dict(self._model_instance.parameters)
        json_schema = (
            LLMNode.fetch_structured_output_schema(
                structured_output=self.node_data.structured_output or {},
            )
            if self.node_data.structured_output_enabled
            else None
        )
        self._raise_if_polling_aborted()
        request_start_time = time.perf_counter()
        polling_result = self._normalize_polling_result(
            polling_model.start_llm_polling(
                prompt_messages=prompt_messages,
                model_parameters=model_parameters,
                tools=None,
                stop=stop,
                json_schema=json_schema,
            ),
        )

        deadline = request_start_time + config.max_wait_seconds
        max_attempts = config.max_attempts
        attempts = 0

        while True:
            self._raise_if_polling_aborted()
            deadline = self._updated_polling_deadline(
                deadline=deadline,
                polling_result=polling_result,
            )
            max_attempts = self._updated_polling_max_attempts(
                max_attempts=max_attempts,
                polling_result=polling_result,
                config=config,
            )
            self._raise_if_polling_deadline_exceeded(deadline)

            match polling_result.status:
                case LLMPollingStatus.SUCCEEDED:
                    if polling_result.result is None:
                        msg = "LLM polling succeeded without a result"
                        raise LLMNodeError(msg)
                    yield from LLMNode.handle_invoke_result(
                        invoke_result=polling_result.result,
                        file_saver=self._llm_file_saver,
                        file_outputs=self._file_outputs,
                        node_id=self._node_id,
                        model_instance=self._model_instance,
                        reasoning_format=self.node_data.reasoning_format,
                        request_start_time=request_start_time,
                    )
                    return
                case LLMPollingStatus.FAILED:
                    if not polling_result.error:
                        msg = "LLM polling failed without an error"
                        raise LLMNodeError(msg)
                    raise LLMNodeError(polling_result.error)
                case LLMPollingStatus.RUNNING:
                    plugin_state = polling_result.plugin_state
                    if plugin_state is None:
                        msg = "LLM polling is running without plugin_state"
                        raise LLMNodeError(msg)
                    if attempts >= max_attempts:
                        msg = "LLM polling exceeded max attempts"
                        raise LLMNodeError(msg)

                    delay = self._polling_delay(
                        polling_result=polling_result,
                        config=config,
                    )
                    yield self._build_polling_progress_event(
                        attempt=attempts,
                        delay_seconds=delay,
                        deadline=deadline,
                    )
                    self._sleep_until_next_polling_check(
                        delay_seconds=delay,
                        deadline=deadline,
                        config=config,
                    )
                    attempts += 1
                    polling_result = self._normalize_polling_result(
                        polling_model.check_llm_polling(
                            plugin_state=plugin_state,
                        ),
                    )

    @staticmethod
    def _polling_config(
        polling_model: LLMPollingCapableProtocol,
    ) -> LLMPollingConfig:
        raw_config = getattr(polling_model, "polling_config", None)
        if isinstance(raw_config, LLMPollingConfig):
            return raw_config
        if raw_config is None:
            return LLMPollingConfig()
        return LLMPollingConfig.model_validate(raw_config)

    @staticmethod
    def _normalize_polling_result(result: object) -> LLMPollingResult:
        if isinstance(result, LLMPollingResult):
            return result
        return LLMPollingResult.model_validate(result)

    def _resolve_required_workflow_run_id(self) -> str:
        segment = self.graph_runtime_state.variable_pool.get(("sys", "workflow_run_id"))
        if segment is not None and segment.text:
            return segment.text
        run_context_value = self.get_run_context_value("workflow_run_id")
        if isinstance(run_context_value, str) and run_context_value:
            return run_context_value
        msg = "LLM polling requires workflow_run_id"
        raise LLMNodeError(msg)

    @staticmethod
    def _updated_polling_deadline(
        *,
        deadline: float,
        polling_result: LLMPollingResult,
    ) -> float:
        if polling_result.expires_after_seconds is None:
            return deadline
        return min(deadline, time.perf_counter() + polling_result.expires_after_seconds)

    @staticmethod
    def _updated_polling_max_attempts(
        *,
        max_attempts: int,
        polling_result: LLMPollingResult,
        config: LLMPollingConfig,
    ) -> int:
        if polling_result.max_attempts is None:
            return max_attempts
        return max(
            1,
            min(max_attempts, config.max_attempts, polling_result.max_attempts),
        )

    @staticmethod
    def _polling_delay(
        *,
        polling_result: LLMPollingResult,
        config: LLMPollingConfig,
    ) -> float:
        delay = polling_result.next_check_after_seconds
        if delay is None:
            delay = config.min_check_interval_seconds
        return min(
            max(delay, config.min_check_interval_seconds),
            config.max_check_interval_seconds,
        )

    @staticmethod
    def _build_polling_progress_event(
        *,
        attempt: int,
        delay_seconds: float,
        deadline: float,
    ) -> ModelPollingProgressEvent:
        checked_at = datetime.now(UTC).replace(tzinfo=None)
        remaining_seconds = deadline - time.perf_counter()
        next_check_at = (
            checked_at + timedelta(seconds=delay_seconds)
            if remaining_seconds > delay_seconds
            else None
        )
        return ModelPollingProgressEvent(
            attempt=attempt,
            last_checked_at=checked_at,
            next_check_at=next_check_at,
        )

    def _sleep_until_next_polling_check(
        self,
        *,
        delay_seconds: float,
        deadline: float,
        config: LLMPollingConfig,
    ) -> None:
        end_at = min(time.perf_counter() + delay_seconds, deadline)
        while True:
            self._raise_if_polling_aborted()
            remaining = end_at - time.perf_counter()
            if remaining <= 0:
                break
            time.sleep(min(remaining, config.wake_interval_seconds))

        if time.perf_counter() >= deadline:
            msg = "LLM polling timed out"
            raise LLMNodeError(msg)

    @staticmethod
    def _raise_if_polling_deadline_exceeded(deadline: float) -> None:
        if time.perf_counter() >= deadline:
            msg = "LLM polling timed out"
            raise LLMNodeError(msg)

    def _raise_if_polling_aborted(self) -> None:
        if self.graph_runtime_state.graph_execution.aborted:
            msg = "workflow execution was aborted"
            raise LLMNodeError(msg)

    def _extract_clean_text(self, text: str) -> str:
        if self.node_data.reasoning_format == "tagged":
            return text

        clean_text, _ = split_reasoning(text, self.node_data.reasoning_format)
        return clean_text

    def _build_process_data(
        self,
        *,
        prompt_messages: Sequence[PromptMessage],
        usage: LLMUsage,
        finish_reason: str | None,
        model_provider: Any,
        model_name: str,
    ) -> dict[str, Any]:
        return {
            "model_mode": self.node_data.model.mode,
            "prompts": self._prompt_message_serializer.serialize(
                model_mode=self.node_data.model.mode,
                prompt_messages=prompt_messages,
            ),
            "usage": jsonable_encoder(usage),
            "finish_reason": finish_reason,
            "model_provider": model_provider,
            "model_name": model_name,
        }

    def _build_run_outputs(
        self,
        *,
        clean_text: str,
        usage: LLMUsage,
        finish_reason: str | None,
        reasoning_content: str,
        structured_output: LLMStructuredOutput | None,
    ) -> dict[str, Any]:
        outputs = {
            "text": clean_text,
            "reasoning_content": reasoning_content,
            "usage": jsonable_encoder(usage),
            "finish_reason": finish_reason,
        }
        if structured_output:
            outputs["structured_output"] = structured_output.structured_output
        if self._file_outputs:
            outputs["files"] = ArrayFileSegment(value=self._file_outputs)
        return outputs

    @staticmethod
    def invoke_llm(
        *,
        model_instance: LLMProtocol,
        prompt_messages: Sequence[PromptMessage],
        stop: Sequence[str] | None = None,
        structured_output_enabled: bool,
        structured_output: Mapping[str, Any] | None = None,
        file_saver: LLMFileSaver,
        file_outputs: list[File],
        node_id: str,
        reasoning_format: Literal["separated", "tagged"] = "tagged",
    ) -> Generator[NodeEventBase | LLMStructuredOutput, None, None]:
        model_parameters = model_instance.parameters
        invoke_model_parameters = dict(model_parameters)
        invoke_result: LLMResult | Generator[LLMResultChunk, None, None]
        if structured_output_enabled:
            output_schema = LLMNode.fetch_structured_output_schema(
                structured_output=structured_output or {},
            )
            request_start_time = time.perf_counter()

            invoke_result = model_instance.invoke_llm_with_structured_output(
                prompt_messages=prompt_messages,
                json_schema=output_schema,
                model_parameters=invoke_model_parameters,
                stop=stop,
                stream=True,
            )
        else:
            request_start_time = time.perf_counter()

            invoke_result = model_instance.invoke_llm(
                prompt_messages=prompt_messages,
                model_parameters=invoke_model_parameters,
                tools=None,
                stop=stop,
                stream=True,
            )

        return LLMNode.handle_invoke_result(
            invoke_result=invoke_result,
            file_saver=file_saver,
            file_outputs=file_outputs,
            node_id=node_id,
            model_instance=model_instance,
            reasoning_format=reasoning_format,
            request_start_time=request_start_time,
        )

    @staticmethod
    def handle_invoke_result(
        *,
        invoke_result: LLMResult
        | Generator[LLMResultChunk | LLMStructuredOutput, None, None],
        file_saver: LLMFileSaver,
        file_outputs: list[File],
        node_id: str,
        model_instance: LLMProtocol,
        reasoning_format: Literal["separated", "tagged"] = "tagged",
        request_start_time: float | None = None,
    ) -> Generator[NodeEventBase | LLMStructuredOutput, None, None]:
        if isinstance(invoke_result, LLMResult):
            yield from LLMNode._yield_blocking_invoke_result(
                invoke_result=invoke_result,
                file_saver=file_saver,
                file_outputs=file_outputs,
                reasoning_format=reasoning_format,
                request_start_time=request_start_time,
            )
            return

        yield from LLMNode._yield_streaming_invoke_result(
            invoke_result=invoke_result,
            file_saver=file_saver,
            file_outputs=file_outputs,
            node_id=node_id,
            model_instance=model_instance,
            reasoning_format=reasoning_format,
            request_start_time=request_start_time,
        )

    @staticmethod
    def _yield_blocking_invoke_result(
        *,
        invoke_result: LLMResult,
        file_saver: LLMFileSaver,
        file_outputs: list[File],
        reasoning_format: Literal["separated", "tagged"] = "tagged",
        request_start_time: float | None = None,
    ) -> Generator[ModelInvokeCompletedEvent, None, None]:
        duration = None
        if request_start_time is not None:
            duration = time.perf_counter() - request_start_time
            invoke_result.usage.latency = round(duration, 3)

        yield LLMNode.handle_blocking_result(
            invoke_result=invoke_result,
            saver=file_saver,
            file_outputs=file_outputs,
            reasoning_format=reasoning_format,
            request_latency=duration,
        )

    @staticmethod
    def _yield_streaming_invoke_result(
        *,
        invoke_result: Generator[LLMResultChunk | LLMStructuredOutput, None, None],
        file_saver: LLMFileSaver,
        file_outputs: list[File],
        node_id: str,
        model_instance: LLMProtocol,
        reasoning_format: Literal["separated", "tagged"] = "tagged",
        request_start_time: float | None = None,
    ) -> Generator[NodeEventBase | LLMStructuredOutput, None, None]:
        start_time = (
            request_start_time
            if request_start_time is not None
            else time.perf_counter()
        )
        state = _StreamingInvokeState(start_time=start_time)
        if reasoning_format == "separated":
            state.text_filter = ThinkStreamFilter()

        try:
            yield from LLMNode._yield_streaming_events(
                invoke_result=invoke_result,
                state=state,
                file_saver=file_saver,
                file_outputs=file_outputs,
                node_id=node_id,
            )
        except Exception as e:
            if LLMNode._is_structured_output_parse_error(
                model_instance=model_instance,
                error=e,
            ):
                msg = f"Failed to parse structured output: {e}"
                raise LLMNodeError(msg) from e
            if type(e).__name__ == "OutputParserError":
                msg = f"Failed to parse structured output: {e}"
                raise LLMNodeError(msg) from e
            raise

        # Flush held text, then mark the reasoning stream finished (separated mode).
        if state.text_filter is not None:
            final_pieces = state.text_filter.finalize()
            has_final_reasoning = False
            for index, piece in enumerate(final_pieces):
                is_final_reasoning = (
                    piece.kind == "reasoning" and index == len(final_pieces) - 1
                )
                has_final_reasoning = has_final_reasoning or is_final_reasoning
                yield from LLMNode._yield_filter_piece(
                    piece=piece,
                    state=state,
                    node_id=node_id,
                    is_final=is_final_reasoning,
                )
            if state.reasoning_started and not has_final_reasoning:
                yield StreamReasoningEvent(
                    selector=LLMNode._reasoning_selector(node_id),
                    chunk="",
                    is_final=True,
                )

        # Extract reasoning content from <think> tags in the main text
        full_text = state.full_text_buffer.getvalue()
        clean_text, reasoning_content = extract_stream_reasoning(
            full_text=full_text,
            reasoning_format=reasoning_format,
        )
        LLMNode._finalize_streaming_usage(
            usage=state.usage,
            has_content=state.has_content,
            first_token_time=state.first_token_time,
            start_time=state.start_time,
        )

        yield ModelInvokeCompletedEvent(
            # Use clean_text for separated mode, full_text for tagged mode
            text=clean_text if reasoning_format == "separated" else full_text,
            usage=state.usage,
            finish_reason=state.finish_reason,
            # Reasoning content for workflow variables and downstream nodes
            reasoning_content=reasoning_content,
            # Pass structured output if collected from streaming chunks
            structured_output=state.structured_output,
        )

    @staticmethod
    def _yield_streaming_events(
        *,
        invoke_result: Generator[LLMResultChunk | LLMStructuredOutput, None, None],
        state: _StreamingInvokeState,
        file_saver: LLMFileSaver,
        file_outputs: list[File],
        node_id: str,
    ) -> Generator[NodeEventBase | LLMStructuredOutput, None, None]:
        for result in invoke_result:
            yield from LLMNode._handle_stream_result(
                result=result,
                state=state,
                file_saver=file_saver,
                file_outputs=file_outputs,
                node_id=node_id,
            )

    @staticmethod
    def _handle_stream_result(
        *,
        result: LLMResultChunk | LLMStructuredOutput,
        state: _StreamingInvokeState,
        file_saver: LLMFileSaver,
        file_outputs: list[File],
        node_id: str,
    ) -> Generator[NodeEventBase | LLMStructuredOutput, None, None]:
        if isinstance(result, LLMResultChunkWithStructuredOutput):
            if result.structured_output is not None:
                state.structured_output = dict(result.structured_output)
            yield result

        if isinstance(result, LLMResultChunk):
            yield from LLMNode._yield_stream_text_events(
                result=result,
                state=state,
                file_saver=file_saver,
                file_outputs=file_outputs,
                node_id=node_id,
            )
            LLMNode._update_streaming_metadata(result=result, state=state)

    @staticmethod
    def _yield_stream_text_events(
        *,
        result: LLMResultChunk,
        state: _StreamingInvokeState,
        file_saver: LLMFileSaver,
        file_outputs: list[File],
        node_id: str,
    ) -> Generator[StreamChunkEvent | StreamReasoningEvent, None, None]:
        text_parts = LLMNode._save_multimodal_output_and_convert_result_to_markdown(
            contents=result.delta.message.content,
            file_saver=file_saver,
            file_outputs=file_outputs,
        )
        for text_part in text_parts:
            yield from LLMNode._build_stream_text_events(
                text_part=text_part,
                state=state,
                node_id=node_id,
            )

    @staticmethod
    def _build_stream_text_events(
        *,
        text_part: str,
        state: _StreamingInvokeState,
        node_id: str,
    ) -> Generator[StreamChunkEvent | StreamReasoningEvent, None, None]:
        if text_part and not state.has_content:
            state.first_token_time = time.perf_counter()
            state.has_content = True

        # Keep the raw text (including <think> tags) in the buffer so the final
        # reasoning extraction still sees the full output.
        state.full_text_buffer.write(text_part)

        # "tagged" (no filter): stream the raw token unchanged, no reasoning.
        if state.text_filter is None:
            yield StreamChunkEvent(
                selector=[node_id, "text"],
                chunk=text_part,
                is_final=False,
            )
            return

        # "separated": split <think> reasoning off the answer onto its own channel.
        for piece in state.text_filter.feed(text_part):
            yield from LLMNode._yield_filter_piece(
                piece=piece,
                state=state,
                node_id=node_id,
            )

    @staticmethod
    def _yield_filter_piece(
        *,
        piece: FilterPiece,
        state: _StreamingInvokeState,
        node_id: str,
        is_final: bool = False,
    ) -> Generator[StreamChunkEvent | StreamReasoningEvent, None, None]:
        match piece.kind:
            case "text":
                yield StreamChunkEvent(
                    selector=[node_id, "text"],
                    chunk=piece.chunk,
                    is_final=False,
                )
            case "reasoning":
                state.reasoning_started = True
                yield StreamReasoningEvent(
                    selector=LLMNode._reasoning_selector(node_id),
                    chunk=piece.chunk,
                    is_final=is_final,
                )
            case _:
                assert_never(piece.kind)

    @staticmethod
    def _reasoning_selector(node_id: str) -> list[str]:
        return [node_id, "reasoning_content"]

    @staticmethod
    def _update_streaming_metadata(
        *,
        result: LLMResultChunk,
        state: _StreamingInvokeState,
    ) -> None:
        if state.usage.prompt_tokens == 0 and result.delta.usage:
            state.usage = result.delta.usage
        if state.finish_reason is None and result.delta.finish_reason:
            state.finish_reason = result.delta.finish_reason

    @staticmethod
    def _is_structured_output_parse_error(
        *,
        model_instance: LLMProtocol,
        error: Exception,
    ) -> bool:
        is_structured_output_parse_error = getattr(
            model_instance,
            "is_structured_output_parse_error",
            None,
        )
        return bool(
            callable(is_structured_output_parse_error)
            and is_structured_output_parse_error(error)
        )

    @staticmethod
    def _finalize_streaming_usage(
        *,
        usage: LLMUsage,
        has_content: bool,
        first_token_time: float | None,
        start_time: float,
    ) -> None:
        end_time = time.perf_counter()
        total_duration = end_time - start_time
        usage.latency = round(total_duration, 3)
        if not has_content or first_token_time is None:
            return

        gen_ai_server_time_to_first_token = first_token_time - start_time
        llm_streaming_time_to_generate = end_time - first_token_time
        usage.time_to_first_token = round(gen_ai_server_time_to_first_token, 3)
        usage.time_to_generate = round(llm_streaming_time_to_generate, 3)

    @staticmethod
    def _saved_file_to_markdown(file: File, /) -> str:
        if file.type == FileType.IMAGE:
            return f"![]({file.generate_url()})"
        return file.markdown

<h1>_transform_chat_messages</h1>

    def _transform_chat_messages(
        self,
        messages: Sequence[LLMNodeChatModelMessage]
        | LLMNodeCompletionModelPromptTemplate,
        /,
    ) -> Sequence[LLMNodeChatModelMessage] | LLMNodeCompletionModelPromptTemplate:
        if isinstance(messages, LLMNodeCompletionModelPromptTemplate):
            if messages.edition_type == "jinja2" and messages.jinja2_text:
                messages.text = messages.jinja2_text

            return messages

        for message in messages:
            if message.edition_type == "jinja2" and message.jinja2_text:
                message.text = message.jinja2_text

        return messages

    def _fetch_jinja_inputs(self, node_data: LLMNodeData) -> dict[str, str]:
        if not node_data.prompt_config:
            return {}

        variables: dict[str, str] = {}
        for variable_selector in node_data.prompt_config.jinja2_variables or []:
            variable = self.graph_runtime_state.variable_pool.get(
                variable_selector.value_selector,
            )
            if variable is not None:
                variables[variable_selector.variable] = self._stringify_jinja_variable(
                    variable,
                )
            else:
                variables[variable_selector.variable] = ""
        return variables
