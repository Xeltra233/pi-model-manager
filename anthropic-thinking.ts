// Anthropic thinking 协议能力表（对齐官方 per-model 配置表，2026-10）。
//
// 官方结论：
//   - Claude 4.7 及更新模型只接受 thinking.type=adaptive；发送 thinking.type=enabled
//     + budget_tokens 会直接 400（"requires adaptive thinking"）。
//   - Claude 4.5 及更早的 thinking 模型只接受 thinking.type=enabled；发送 adaptive 会 400。
//   - Claude 4.6 两代都接受（extended 已废弃），保持用户的显式选择即可。
//   - Fable/Mythos 5、Opus 5.5、Sonnet 5.5 等模型不能关闭思考，thinking.type=disabled 会 400；
//     Pi 用 thinkingLevelMap.off = null 表达“不可关闭”，构造请求时省略 thinking。
//
// 这里只做纯函数判断，供草稿、持久化、注册和请求兜底共用同一张表。

import { isObjectRecord } from "./common.ts";
import type { AnthropicThinkingProtocol, CompatSettings } from "./types.ts";

export interface AnthropicThinkingSupport {
	/** 接受 thinking.type=adaptive。 */
	adaptive: boolean;
	/** 接受 thinking.type=enabled + budget_tokens。 */
	extended: boolean;
	/** 拒绝 thinking.type=disabled（模型始终思考）。 */
	rejectsDisabled: boolean;
}

const ADAPTIVE_ONLY: AnthropicThinkingSupport = { adaptive: true, extended: false, rejectsDisabled: false };
const ADAPTIVE_ONLY_ALWAYS_ON: AnthropicThinkingSupport = { adaptive: true, extended: false, rejectsDisabled: true };
const BOTH: AnthropicThinkingSupport = { adaptive: true, extended: true, rejectsDisabled: false };
const BOTH_ALWAYS_ON: AnthropicThinkingSupport = { adaptive: true, extended: true, rejectsDisabled: true };
const EXTENDED_ONLY: AnthropicThinkingSupport = { adaptive: false, extended: true, rejectsDisabled: false };

// 与 Pi 内置 catalog 保持一致的第三方 Anthropic 协议模型（内置表同样标记 forceAdaptiveThinking）。
const THIRD_PARTY_ADAPTIVE_ONLY_IDS = new Set(["k3", "k3-256k", "kimi-for-coding", "kimi-for-coding-highspeed"]);

const VARIANT_SUFFIXES = [
	"-thinking",
	"-reasoning",
	"-thinking-none",
	"-thinking-high",
	"-xhigh",
	"-max",
	"-high",
	"-medium",
	"-low",
	"-none",
];

const CLAUDE_GENERATION_PATTERN = /^claude-(opus|sonnet|haiku)-(\d+)(?:-(\d+))?$/;
const CLAUDE_LEGACY_NAMING_PATTERN = /^claude-3(?:-\d+)?-(?:opus|sonnet|haiku)$/;
const FABLE_MYTHOS_PATTERN = /^claude-(fable|mythos)-5(?:-\d+)?$/;

/**
 * 归一化上游常见写法：命名空间前缀、[1m] 之类标签、日期后缀、`-thinking` 等展示后缀、
 * 以及 `5.5` 这种点号版本。
 */
export function normalizeAnthropicModelId(modelId: string): string {
	let id = modelId.trim().toLowerCase();
	id = id.replace(/\[[^\]]*\]/g, "").replace(/[（(][^)）]*[)）]/g, "");
	id = id.trim();
	const lastSpace = id.lastIndexOf(" ");
	if (lastSpace >= 0) id = id.slice(lastSpace + 1);
	const lastSlash = id.lastIndexOf("/");
	if (lastSlash >= 0) id = id.slice(lastSlash + 1);
	id = id.replace(/(\d)\.(\d)/g, "$1-$2");
	id = id.replace(/-\d{8}$/, "");
	let changed = true;
	while (changed) {
		changed = false;
		for (const suffix of VARIANT_SUFFIXES) {
			if (!id.endsWith(suffix) || id.length === suffix.length) continue;
			id = id.slice(0, -suffix.length);
			changed = true;
		}
	}
	return id;
}

/** 解析模型的 thinking 能力；未知模型返回 undefined（沿用用户的显式设置）。 */
export function getAnthropicThinkingSupport(modelId: string): AnthropicThinkingSupport | undefined {
	const id = normalizeAnthropicModelId(modelId);
	if (THIRD_PARTY_ADAPTIVE_ONLY_IDS.has(id)) return ADAPTIVE_ONLY;
	if (id === "claude-mythos-preview") return BOTH_ALWAYS_ON;
	if (FABLE_MYTHOS_PATTERN.test(id)) return ADAPTIVE_ONLY_ALWAYS_ON;
	if (id === "claude-opus-5-5" || id === "claude-sonnet-5-5") return ADAPTIVE_ONLY_ALWAYS_ON;
	if (CLAUDE_LEGACY_NAMING_PATTERN.test(id)) return EXTENDED_ONLY;
	const generation = CLAUDE_GENERATION_PATTERN.exec(id);
	if (!generation) return undefined;
	const major = Number(generation[2]);
	const minor = generation[3] === undefined ? 0 : Number(generation[3]);
	if (major >= 5 || (major === 4 && minor >= 7)) return ADAPTIVE_ONLY;
	if (major === 4 && minor === 6) return BOTH;
	if (major <= 4) return EXTENDED_ONLY;
	return undefined;
}

/** 模型只接受一种协议时返回该协议；双模或未知模型返回 undefined。 */
export function resolveAnthropicThinkingProtocol(modelId: string): AnthropicThinkingProtocol | undefined {
	const support = getAnthropicThinkingSupport(modelId);
	if (!support) return undefined;
	if (support.adaptive && !support.extended) return "adaptive";
	if (support.extended && !support.adaptive) return "legacy";
	return undefined;
}

export function isAnthropicThinkingDisabledRejected(modelId: string): boolean {
	return getAnthropicThinkingSupport(modelId)?.rejectsDisabled === true;
}

/**
 * 把能力表要求写进模型级 compat。
 * 只在“继承后的有效值”与要求不符时返回新的 compat；无需修改时返回 undefined。
 */
export function applyAnthropicThinkingCompat(
	api: string | undefined,
	reasoning: boolean,
	modelId: string,
	modelCompat: CompatSettings | undefined,
	providerCompat: CompatSettings | undefined = undefined,
): CompatSettings | undefined {
	if (api !== "anthropic-messages" || !reasoning) return undefined;
	const required = resolveAnthropicThinkingProtocol(modelId);
	if (!required) return undefined;
	const inherited = modelCompat?.forceAdaptiveThinking ?? providerCompat?.forceAdaptiveThinking;
	const wantsAdaptive = required === "adaptive";
	if (wantsAdaptive) {
		if (inherited === true) return undefined;
		return { ...(modelCompat ?? {}), forceAdaptiveThinking: true };
	}
	// extended-only 模型只在继承值是 adaptive 时需要显式写 false；未设置时 Pi 默认就是 legacy。
	if (inherited !== true) return undefined;
	return { ...(modelCompat ?? {}), forceAdaptiveThinking: false };
}

/** Pi 内置的 thinking 预算档位（与 Pi 默认值一致），用于把 budget_tokens 折回 effort。 */
const DEFAULT_THINKING_BUDGETS = { minimal: 1024, low: 2048, medium: 8192, high: 16384 } as const;

export function mapThinkingBudgetToEffort(budgetTokens: unknown): "low" | "medium" | "high" {
	const budget = typeof budgetTokens === "number" && Number.isFinite(budgetTokens)
		? budgetTokens
		: DEFAULT_THINKING_BUDGETS.minimal;
	if (budget <= DEFAULT_THINKING_BUDGETS.low) return "low";
	if (budget <= DEFAULT_THINKING_BUDGETS.medium) return "medium";
	return "high";
}

/**
 * 请求兜底：model 配置没走到注册层时，把必然 400 的 thinking 参数改写成能力表允许的形式。
 * 只处理“能力表明确禁止”的情况，双模与未知模型保持原样。
 */
export function normalizeAnthropicAdaptiveThinkingPayload(payload: unknown, modelId: string): unknown | undefined {
	if (!isObjectRecord(payload)) return undefined;
	if (typeof payload.model === "string" && payload.model !== modelId) return undefined;
	const support = getAnthropicThinkingSupport(modelId);
	if (!support?.adaptive) return undefined;
	const thinking = isObjectRecord(payload.thinking) ? payload.thinking : undefined;
	if (!thinking) return undefined;

	if (thinking.type === "disabled") {
		if (!support.rejectsDisabled) return undefined;
		const next: Record<string, unknown> = { ...payload };
		delete next.thinking;
		return next;
	}
	if (thinking.type !== "enabled" || support.extended) return undefined;

	const { budget_tokens: budgetTokens, ...rest } = thinking;
	const next: Record<string, unknown> = { ...payload, thinking: { ...rest, type: "adaptive" } };
	if (!isObjectRecord(next.thinking) || !("display" in next.thinking)) {
		(next.thinking as Record<string, unknown>).display = "summarized";
	}
	const outputConfig = isObjectRecord(payload.output_config) ? payload.output_config : undefined;
	if (outputConfig?.effort === undefined) {
		next.output_config = { ...(outputConfig ?? {}), effort: mapThinkingBudgetToEffort(budgetTokens) };
	}
	return next;
}
