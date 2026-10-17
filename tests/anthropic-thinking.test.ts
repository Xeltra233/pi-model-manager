// Anthropic thinking 协议能力表的回归测试：
// 覆盖官方 per-model 表（Claude 4.7+ 只接受 adaptive / 4.5- 只接受 extended）、
// 草稿与保存强制、启动迁移、models.json 同步、注册层下发和请求兜底。

import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import test, { after } from "node:test";
import {
	applyAnthropicThinkingCompat,
	getAnthropicThinkingSupport,
	normalizeAnthropicAdaptiveThinkingPayload,
	normalizeAnthropicModelId,
	resolveAnthropicThinkingProtocol,
} from "../anthropic-thinking.ts";
import type { StateDocument, StoredModel, StoredProvider } from "../types.ts";

const agentDir = await mkdtemp(resolve(".test-anthropic-thinking-"));
process.env.PI_CODING_AGENT_DIR = agentDir;
const {
	buildModelFromDraft,
	createModelDraftFromStoredModel,
	findAnthropicThinkingNormalizationTargets,
	normalizeAnthropicThinkingInDocument,
} = await import("../state-document.ts");
const { STATE_DIR, STATE_PATH } = await import("../state-metadata-store.ts");
const { MODELS_JSON_PATH } = await import("../models-json-manager.ts");
const { invalidateStateCache } = await import("../state-cache.ts");
const { createRequestPipeline } = await import("../request-pipeline.ts");
const { reconcileProvider } = await import("../provider-registrar.ts");
const { buildSynchronizedModelsDocument } = await import("../models-json-sync.ts");
after(async () => { await rm(agentDir, { recursive: true, force: true }); });

const ZERO_COST = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };

function model(id: string, patch: Partial<StoredModel> = {}): StoredModel {
	return { id, reasoning: true, input: ["text"], contextWindow: 200_000, maxTokens: 64_000, cost: { ...ZERO_COST }, ...patch };
}

function provider(id: string, models: StoredModel[], patch: Partial<StoredProvider> = {}): StoredProvider {
	return {
		name: id,
		api: "anthropic-messages",
		baseUrl: "https://relay.test",
		managed: true,
		clientHeaderProfile: "recommended",
		models,
		...patch,
	};
}

function document(providers: Record<string, StoredProvider>): StateDocument {
	return {
		version: 2,
		providers,
		managedProviderIds: Object.keys(providers),
		requestHeaderProfiles: {},
		clientHeaderCaptures: {},
	};
}

test("能力表按官方 per-model 表归类 thinking 协议", () => {
	assert.equal(resolveAnthropicThinkingProtocol("claude-opus-5-5"), "adaptive");
	assert.equal(resolveAnthropicThinkingProtocol("claude-opus-5.5"), "adaptive");
	assert.equal(resolveAnthropicThinkingProtocol("claude-opus-5-5-thinking"), "adaptive");
	assert.equal(resolveAnthropicThinkingProtocol("anthropic/claude-opus-5-5[1m]"), "adaptive");
	assert.equal(resolveAnthropicThinkingProtocol("claude-sonnet-5"), "adaptive");
	assert.equal(resolveAnthropicThinkingProtocol("claude-haiku-5-5"), "adaptive");
	assert.equal(resolveAnthropicThinkingProtocol("claude-fable-5-1"), "adaptive");
	assert.equal(resolveAnthropicThinkingProtocol("claude-mythos-5"), "adaptive");
	assert.equal(resolveAnthropicThinkingProtocol("kimi-for-coding"), "adaptive");

	// 双模模型保持用户选择；4.5 及更早只接受 extended。
	assert.equal(resolveAnthropicThinkingProtocol("claude-opus-4-6"), undefined);
	assert.equal(resolveAnthropicThinkingProtocol("claude-sonnet-4-6"), undefined);
	assert.equal(resolveAnthropicThinkingProtocol("claude-mythos-preview"), undefined);
	assert.equal(resolveAnthropicThinkingProtocol("claude-opus-4-5"), "legacy");
	assert.equal(resolveAnthropicThinkingProtocol("claude-sonnet-4-5-20250929"), "legacy");
	assert.equal(resolveAnthropicThinkingProtocol("claude-haiku-4-5"), "legacy");
	assert.equal(resolveAnthropicThinkingProtocol("claude-3-7-sonnet"), "legacy");
	assert.equal(resolveAnthropicThinkingProtocol("gpt-5.6-sol"), undefined);

	// 不能关闭思考的模型：Pi 用 thinkingLevelMap.off=null 省略 thinking。
	assert.equal(getAnthropicThinkingSupport("claude-opus-5-5")?.rejectsDisabled, true);
	assert.equal(getAnthropicThinkingSupport("claude-sonnet-5-5")?.rejectsDisabled, true);
	assert.equal(getAnthropicThinkingSupport("claude-opus-4-7")?.rejectsDisabled, false);
	assert.equal(getAnthropicThinkingSupport("claude-opus-4-5")?.rejectsDisabled, false);
	assert.equal(normalizeAnthropicModelId("  Claude-Opus-5.5  "), "claude-opus-5-5");
});

test("模型 compat 归一化只改写与能力表冲突的继承值", () => {
	assert.deepEqual(
		applyAnthropicThinkingCompat("anthropic-messages", true, "claude-opus-5-5", undefined),
		{ forceAdaptiveThinking: true },
	);
	assert.equal(
		applyAnthropicThinkingCompat("anthropic-messages", true, "claude-opus-5-5", undefined, { forceAdaptiveThinking: true }),
		undefined,
	);
	assert.deepEqual(
		applyAnthropicThinkingCompat("anthropic-messages", true, "claude-opus-4-5", { forceAdaptiveThinking: true }),
		{ forceAdaptiveThinking: false },
	);
	assert.equal(
		applyAnthropicThinkingCompat("anthropic-messages", true, "claude-opus-4-5", undefined),
		undefined,
	);
	assert.equal(
		applyAnthropicThinkingCompat("anthropic-messages", true, "claude-opus-4-6", undefined),
		undefined,
	);
	assert.equal(applyAnthropicThinkingCompat("openai-completions", true, "claude-opus-5-5", undefined), undefined);
});

test("模型草稿与保存都以能力表为准，并写入 off=null", () => {
	const relay = provider("relay", []);
	const adaptiveModel = model("claude-opus-5-5");
	const draft = createModelDraftFromStoredModel("relay", relay, adaptiveModel);
	assert.equal(draft.anthropicThinkingProtocol, "adaptive");

	draft.anthropicThinkingProtocol = "legacy";
	const saved = buildModelFromDraft(adaptiveModel, draft, relay.compat);
	assert.equal(saved.compat?.forceAdaptiveThinking, true);
	assert.equal(saved.thinkingLevelMap?.off, null);

	const legacyModel = model("claude-opus-4-5");
	const legacyDraft = createModelDraftFromStoredModel("relay", relay, legacyModel);
	assert.equal(legacyDraft.anthropicThinkingProtocol, "legacy");
	legacyDraft.anthropicThinkingProtocol = "adaptive";
	const legacySaved = buildModelFromDraft(legacyModel, legacyDraft, relay.compat);
	// 未设置时 Pi 默认就是 legacy，不必写显式 false。
	assert.equal(legacySaved.compat?.forceAdaptiveThinking, undefined);
	assert.notEqual(legacySaved.thinkingLevelMap?.off, null);
	// provider 级强开 adaptive 时，模型级必须显式覆盖回 legacy。
	const overriddenSaved = buildModelFromDraft(legacyModel, legacyDraft, { forceAdaptiveThinking: true });
	assert.equal(overriddenSaved.compat?.forceAdaptiveThinking, false);

	// 双模模型尊重用户选择。
	const bothModel = model("claude-opus-4-6");
	const bothDraft = createModelDraftFromStoredModel("relay", relay, bothModel);
	assert.equal(bothDraft.anthropicThinkingProtocol, "legacy");
	const bothSaved = buildModelFromDraft(bothModel, bothDraft, relay.compat);
	assert.equal(bothSaved.compat?.forceAdaptiveThinking, undefined);
});

test("启动迁移补齐旧配置并幂等，未受管理接入不动", () => {
	const relay = provider("legacy-relay", [model("claude-opus-5-5"), model("gpt-5.6-sol")]);
	const native = provider("native-relay", [model("claude-opus-5-5")], { managed: false });
	const before = document({ "legacy-relay": relay, "native-relay": native });

	// 迁移比对 models.json 的原始值：这里模拟用户现有的 shuai 配置（模型没写 compat）。
	const source = {
		providers: {
			"legacy-relay": { api: "anthropic-messages", models: [{ id: "claude-opus-5-5" }, { id: "gpt-5.6-sol" }] },
			"native-relay": { api: "anthropic-messages", models: [{ id: "claude-opus-5-5" }] },
		},
	};
	const targets = findAnthropicThinkingNormalizationTargets(before, source);
	assert.deepEqual(targets, [{ providerId: "legacy-relay", modelIds: ["claude-opus-5-5"] }]);

	const normalized = normalizeAnthropicThinkingInDocument(before, targets.map((target) => target.providerId));
	const models = normalized.providers["legacy-relay"]!.models;
	assert.equal(models.find((entry) => entry.id === "claude-opus-5-5")?.compat?.forceAdaptiveThinking, true);
	assert.equal(models.find((entry) => entry.id === "gpt-5.6-sol")?.compat?.forceAdaptiveThinking, undefined);
	assert.equal(normalized.providers["native-relay"]!.models[0]!.compat, undefined);

	// models.json 同步写回 compat 与 off=null，之后迁移不再命中（幂等）。
	const synced = buildSynchronizedModelsDocument(normalized, source, [], ["legacy-relay"]);
	assert.equal(synced.providers["legacy-relay"]?.models?.[0]?.compat?.forceAdaptiveThinking, true);
	assert.equal(synced.providers["legacy-relay"]?.models?.[0]?.thinkingLevelMap?.off, null);
	assert.deepEqual(findAnthropicThinkingNormalizationTargets(normalized, synced), []);
});

test("注册层把能力表结论写进 runtime 模型 compat", async () => {
	const registered: Array<{ getModels(): Array<Record<string, unknown>> }> = [];
	const pi = {
		registerProvider: (native: { getModels(): Array<Record<string, unknown>> }) => registered.push(native),
		unregisterProvider: () => {},
	} as never;
	await reconcileProvider(pi, "relay", provider("relay", [model("claude-opus-5-5"), model("claude-opus-4-5")]));
	const models = registered.at(-1)!.getModels();
	assert.equal((models.find((entry) => entry.id === "claude-opus-5-5")?.compat as Record<string, unknown>)?.forceAdaptiveThinking, true);
	// extended-only 模型未设置时就是 Pi 默认的 legacy，不必写显式 false。
	assert.equal(models.find((entry) => entry.id === "claude-opus-4-5")?.compat, undefined);

	// provider 级强开 adaptive 时，extended-only 模型必须显式覆盖回 legacy。
	await reconcileProvider(pi, "relay-forced", provider("relay-forced", [model("claude-opus-4-5")], {
		compat: { forceAdaptiveThinking: true },
	}));
	const forcedModel = registered.at(-1)!.getModels().find((entry) => entry.id === "claude-opus-4-5");
	assert.equal((forcedModel?.compat as Record<string, unknown>)?.forceAdaptiveThinking, false);
});

test("请求兜底把 legacy thinking 改写成 adaptive 并补 effort", () => {
	const payload = {
		model: "claude-opus-5-5",
		max_tokens: 64_000,
		thinking: { type: "enabled", budget_tokens: 8192, display: "summarized" },
	};
	const rewritten = normalizeAnthropicAdaptiveThinkingPayload(payload, "claude-opus-5-5") as Record<string, any>;
	assert.equal(rewritten.thinking.type, "adaptive");
	assert.equal(rewritten.thinking.display, "summarized");
	assert.equal("budget_tokens" in rewritten.thinking, false);
	assert.deepEqual(rewritten.output_config, { effort: "medium" });

	// 原文不可变。
	assert.equal(payload.thinking.type, "enabled");
	assert.equal(payload.thinking.budget_tokens, 8192);

	// 不能关闭思考的模型：省略 thinking，让上游按默认 adaptive 处理。
	const disabled = normalizeAnthropicAdaptiveThinkingPayload(
		{ model: "claude-opus-5-5", thinking: { type: "disabled" } },
		"claude-opus-5-5",
	) as Record<string, unknown>;
	assert.equal("thinking" in disabled, false);

	// 双模 / extended-only / 未知模型保持原样。
	assert.equal(normalizeAnthropicAdaptiveThinkingPayload(
		{ model: "claude-opus-4-6", thinking: { type: "enabled", budget_tokens: 4096 } },
		"claude-opus-4-6",
	), undefined);
	assert.equal(normalizeAnthropicAdaptiveThinkingPayload(
		{ model: "claude-opus-4-5", thinking: { type: "enabled", budget_tokens: 4096 } },
		"claude-opus-4-5",
	), undefined);
	assert.equal(normalizeAnthropicAdaptiveThinkingPayload(
		{ model: "claude-opus-5-5", thinking: { type: "enabled" } },
		"claude-sonnet-4-6",
	), undefined);
});

test("请求流水线只改写受管理 Anthropic 接入的 payload", async () => {
	await mkdir(STATE_DIR, { recursive: true });
	await writeFile(MODELS_JSON_PATH, JSON.stringify({
		providers: {
			relay: {
				baseUrl: "https://relay.test",
				api: "anthropic-messages",
				piModelManager: { managed: true },
				models: [{ id: "claude-opus-5-5" }, { id: "claude-opus-4-5" }, { id: "claude-opus-5-5-native" }],
			},
			native: {
				baseUrl: "https://native.test",
				api: "anthropic-messages",
				models: [{ id: "claude-opus-5-5" }],
			},
		},
	}));
	await writeFile(STATE_PATH, JSON.stringify({
		version: 5,
		managedProviderIds: ["relay"],
		providers: { relay: { clientHeaderProfile: "recommended" }, native: { clientHeaderProfile: "recommended" } },
		models: {},
		requestHeaderProfiles: {},
		clientHeaderCaptures: {},
	}));
	invalidateStateCache();

	const pipeline = createRequestPipeline();
	const ctx = (id: string, providerId = "relay") => ({ model: { provider: providerId, id, api: "anthropic-messages" }, hasUI: false }) as any;

	const rewritten = await pipeline.transform(
		{ model: "claude-opus-5-5", thinking: { type: "enabled", budget_tokens: 16_384 } },
		ctx("claude-opus-5-5"),
	) as Record<string, any>;
	assert.equal(rewritten.thinking.type, "adaptive");
	assert.deepEqual(rewritten.output_config, { effort: "high" });

	const untouched = await pipeline.transform(
		{ model: "claude-opus-4-5", thinking: { type: "enabled", budget_tokens: 4096 } },
		ctx("claude-opus-4-5"),
	) as Record<string, any> | undefined;
	// extended-only 模型保持 legacy；受管理接入可能追加 ClaudeCode metadata，但 thinking 不能动。
	assert.equal(untouched?.thinking?.type, "enabled");
	assert.equal(untouched?.thinking?.budget_tokens, 4096);

	// 未受管理接入不代改。
	assert.equal(await pipeline.transform(
		{ model: "claude-opus-5-5", thinking: { type: "enabled", budget_tokens: 4096 } },
		ctx("claude-opus-5-5", "native"),
	), undefined);
});
