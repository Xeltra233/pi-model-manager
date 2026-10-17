// request-pipeline.ts
//
// 统一协调 before_provider_request payload transform。每个请求内共享一次
// StateDocument；跨请求由 state-cache.ts 按配置文件签名复用状态。

import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import { normalizeAnthropicAdaptiveThinkingPayload } from "./anthropic-thinking.ts";
import { injectClaudeCodeMetadata } from "./claude-code-compat.ts";
import { formatUnknownError } from "./common.ts";
import { t, type MessageKey } from "./i18n.ts";
import { normalizeManagedOpenAIResponsesPayload } from "./openai-responses-payload.ts";
import { injectOpenAIServiceTier } from "./openai-service-tier.ts";
import { readCachedState } from "./state-cache.ts";
import type { StateDocument } from "./types.ts";


type RequestTransform = {
	id: string;
	warning: MessageKey;
	run(payload: unknown, ctx: ExtensionContext, state: StateDocument): unknown | undefined;
};


const REQUEST_TRANSFORMS: RequestTransform[] = [
	{
		// [喵喵喵]: 兜底：模型若不是本插件注册的（或旧 compat 尚未迁移），Pi 仍会发 legacy
		// thinking.type=enabled；Claude 4.7+ 会直接 400，这里按能力表改写成 adaptive。
		id: "anthropic-adaptive-thinking",
		warning: "Anthropic thinking 协议兜底改写失败，本次请求使用原始 payload",
		run(payload, ctx) {
			if (!ctx.model || ctx.model.api !== "anthropic-messages") return undefined;
			return normalizeAnthropicAdaptiveThinkingPayload(payload, ctx.model.id);
		},
	},
	{
		id: "claude-code-metadata",
		warning: "ClaudeCode metadata 注入失败，本次请求仅使用请求头",
		run(payload, ctx, state) {
			if (!ctx.model || ctx.model.api !== "anthropic-messages") return undefined;
			return injectClaudeCodeMetadata(payload, ctx.model, state);
		},
	},
	{
		id: "openai-responses-instructions",
		warning: "OpenAI Responses instructions 标准化失败，本次请求使用原始 payload",
		run(payload, ctx, state) {
			if (!ctx.model || ctx.model.api !== "openai-responses") return undefined;
			return normalizeManagedOpenAIResponsesPayload(payload, ctx.model, state);
		},
	},
	{
		id: "openai-service-tier",
		warning: "Fast mode 状态读取失败，本次请求未注入 service_tier",
		run(payload, ctx, state) {
			if (!ctx.model || ctx.model.api !== "openai-responses") return undefined;
			return injectOpenAIServiceTier(payload, ctx.model, state);
		},
	},
];

export interface RequestPipeline {
	transform(payload: unknown, ctx: ExtensionContext): Promise<unknown | undefined>;
}

export function createRequestPipeline(): RequestPipeline {
	const notifiedTransformErrors = new Set<string>();

	return {
		async transform(initialPayload, ctx) {
			let payload = initialPayload;
			let changed = false;
			if (!ctx.model || (ctx.model.api !== "anthropic-messages" && ctx.model.api !== "openai-responses")) return undefined;
			let state: StateDocument;
			try {
				state = await readCachedState();
			} catch (error) {
				if (ctx.hasUI && !notifiedTransformErrors.has("state")) {
					notifiedTransformErrors.add("state");
					ctx.ui.notify(`[pi-model-manager] ${t("请求配置读取失败，本次使用原始 payload")}: ${formatUnknownError(error)}`, "warning");
				}
				return undefined;
			}
			const provider = state.providers[ctx.model.provider];
			const storedModel = provider?.models.find((model) => model.id === ctx.model!.id);
			if (!provider?.managed || !storedModel || (storedModel.api ?? provider.api) !== ctx.model.api) return undefined;

			for (const transform of REQUEST_TRANSFORMS) {
				try {
					const nextPayload = await transform.run(payload, ctx, state);
					if (nextPayload !== undefined) {
						payload = nextPayload;
						changed = true;
					}
				} catch (error) {
					if (!notifiedTransformErrors.has(transform.id) && ctx.hasUI) {
						notifiedTransformErrors.add(transform.id);
						ctx.ui.notify(`[pi-model-manager] ${t(transform.warning)}: ${formatUnknownError(error)}`, "warning");
					}
				}
			}

			return changed ? payload : undefined;
		},
	};
}
