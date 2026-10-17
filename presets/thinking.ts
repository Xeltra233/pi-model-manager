// thinkingLevelMap 的协议默认值与持久化归一化。

import { isAnthropicThinkingDisabledRejected } from "../anthropic-thinking.ts";
import type { ApiKind, ThinkingLevelMap } from "../types.ts";

const EXTENDED_THINKING_LEVEL_MAP: ThinkingLevelMap = {
	xhigh: "xhigh",
	max: "max",
};

const OPENAI_THINKING_LEVEL_MAP: ThinkingLevelMap = {
	minimal: null,
	low: "low",
	medium: "medium",
	high: "high",
	xhigh: "xhigh",
	max: "max",
};

// [喵喵喵]: Gemini 原生 ThinkingLevel 只有 MINIMAL/LOW/MEDIUM/HIGH；pi 的 xhigh/max 必须显式禁用。(2026-07-10)
const GOOGLE_GENERATIVE_AI_THINKING_LEVEL_MAP: ThinkingLevelMap = {
	minimal: "minimal",
	low: "low",
	medium: "medium",
	high: "high",
	xhigh: null,
	max: null,
};

function isLegacyShiftedMaxLadder(map: ThinkingLevelMap): boolean {
	return map.minimal === "low"
		&& map.low === "medium"
		&& map.medium === "high"
		&& map.high === "xhigh"
		&& map.xhigh === "max"
		&& map.max === undefined;
}

function buildThinkingLevelMap(api: ApiKind, reasoning: boolean): ThinkingLevelMap | undefined {
	if (!reasoning) return undefined;
	if (api === "google-generative-ai") return { ...GOOGLE_GENERATIVE_AI_THINKING_LEVEL_MAP };
	if (api === "openai-completions" || api === "openai-responses") return { ...OPENAI_THINKING_LEVEL_MAP };
	return { ...EXTENDED_THINKING_LEVEL_MAP };
}

function applyProtocolThinkingLevelLimits(
	api: ApiKind,
	thinkingLevelMap: ThinkingLevelMap | undefined,
): ThinkingLevelMap | undefined {
	if (api === "google-generative-ai") {
		return { ...(thinkingLevelMap ?? {}), ...GOOGLE_GENERATIVE_AI_THINKING_LEVEL_MAP };
	}
	if (!thinkingLevelMap) return undefined;
	return { ...EXTENDED_THINKING_LEVEL_MAP, ...thinkingLevelMap };
}

function mergeThinkingLevelMap(
	defaultMap: ThinkingLevelMap | undefined,
	storedMap: ThinkingLevelMap | undefined,
): ThinkingLevelMap | undefined {
	const merged = { ...(defaultMap ?? {}), ...(storedMap ?? {}) };
	return Object.keys(merged).length > 0 ? merged : undefined;
}

export function normalizeThinkingLevelMap(
	api: ApiKind,
	reasoning: boolean,
	storedMap: ThinkingLevelMap | undefined,
	modelId?: string,
): ThinkingLevelMap | undefined {
	if (!reasoning) return undefined;
	const defaultMap = buildThinkingLevelMap(api, true);
	const normalized = storedMap && isLegacyShiftedMaxLadder(storedMap)
		? defaultMap
		: applyProtocolThinkingLevelLimits(api, mergeThinkingLevelMap(defaultMap, storedMap));
	if (!normalized) return undefined;
	// [喵喵喵]: Fable/Mythos 5、Opus 5.5、Sonnet 5.5 不能关闭思考；off=null 让 Pi 省掉
	// thinking.type=disabled，否则这些模型会直接 400。
	if (api === "anthropic-messages" && modelId && isAnthropicThinkingDisabledRejected(modelId)) {
		return { ...normalized, off: null };
	}
	return normalized;
}
