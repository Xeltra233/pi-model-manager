import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import test, { after } from "node:test";
import { getClientHeadersForProfile, mergeModelRequestHeaders } from "../presets/client-headers.ts";
import type { StoredModel, StoredProvider } from "../types.ts";

const agentDir = await mkdtemp(resolve(".test-request-boundaries-"));
process.env.PI_CODING_AGENT_DIR = agentDir;
const { buildModelRequestHeaders } = await import("../provider-registrar.ts");
const { STATE_DIR, STATE_PATH } = await import("../state-metadata-store.ts");
const { MODELS_JSON_PATH } = await import("../models-json-manager.ts");
const { invalidateStateCache } = await import("../state-cache.ts");
const { createRequestPipeline } = await import("../request-pipeline.ts");
after(async () => { await rm(agentDir, { recursive: true, force: true }); });

const model: StoredModel = { id: "claude", reasoning: true, input: ["text"], contextWindow: 128000, maxTokens: 16000, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };

test("自定义 beta 保留原值，显式字段覆盖内置模板且同名头大小写不产生重复", () => {
	const custom = { "Anthropic-Beta": "interleaved-thinking-2025-05-14, custom-beta", "User-Agent": "explicit-client" };
	assert.deepEqual(getClientHeadersForProfile("custom", "anthropic-messages", custom, {}, { forceAdaptiveThinking: true }), custom);
	assert.equal(getClientHeadersForProfile("claude-code", "anthropic-messages", {}, {}, { forceAdaptiveThinking: true })?.["anthropic-beta"]?.includes("interleaved-thinking"), false);
	const provider: StoredProvider = {
		name: "Gateway", api: "anthropic-messages", baseUrl: "https://gw.test/v1", managed: true,
		clientHeaderProfile: "recommended", headers: { "ANTHROPIC-BETA": "provider-beta", "User-Agent": "provider-client" },
		models: [model], compat: { forceAdaptiveThinking: true },
	};
	const headers = buildModelRequestHeaders(provider, { ...model, headers: { "user-AGENT": "model-client" } }, {}, {})!;
	assert.equal(headers["user-agent"], "model-client");
	assert.equal(headers["anthropic-beta"], "provider-beta");
	provider.clientHeaderProfile = "custom";
	provider.customClientHeaders = custom;
	assert.equal(buildModelRequestHeaders(provider, model, {}, {})?.["anthropic-beta"], custom["Anthropic-Beta"]);
	assert.deepEqual(mergeModelRequestHeaders({ "X-Test": "default" }, { "x-test": "explicit" }), { "x-test": "explicit" });
});

async function seed(managed: boolean) {
	await mkdir(STATE_DIR, { recursive: true });
	await writeFile(MODELS_JSON_PATH, JSON.stringify({ providers: { gateway: {
		api: "openai-responses", baseUrl: "https://gw.test/v1", piModelManager: { managed },
		models: [{ id: "responses" }, { id: "claude", api: "anthropic-messages" }],
	} } }));
	await writeFile(STATE_PATH, JSON.stringify({ version: 5, managedProviderIds: ["gateway"], providers: { gateway: { clientHeaderProfile: "claude-code" } }, models: { "gateway/responses": { openAIServiceTier: "priority" } }, requestHeaderProfiles: {}, clientHeaderCaptures: {} }));
	invalidateStateCache();
}
const context = (id: string, api: string) => ({ model: { provider: "gateway", id, api }, hasUI: false }) as any;
const prompt = () => ({ model: "responses", input: [{ role: "developer", content: "system rules" }, { role: "user", content: "hello" }] });

test("共享改写入口对未管理模型保持原样，受管理模型保留 instructions 和 service_tier 约定", async () => {
	const pipeline = createRequestPipeline();
	await seed(false);
	assert.equal(await pipeline.transform(prompt(), context("responses", "openai-responses")), undefined);
	assert.equal(await pipeline.transform({ model: "claude" }, context("claude", "anthropic-messages")), undefined);
	await seed(true);
	const original = prompt();
	const transformed = await pipeline.transform(original, context("responses", "openai-responses")) as any;
	assert.equal(transformed.instructions, "system rules");
	assert.equal(transformed.service_tier, "priority");
	assert.deepEqual(transformed.input, [{ role: "user", content: "hello" }]);
	assert.equal(original.input.length, 2);
	for (const instructions of ["existing", "", null]) {
		const existing = { ...prompt(), instructions, service_tier: "auto" };
		assert.equal(await pipeline.transform(existing, context("responses", "openai-responses")), undefined);
	}
	assert.equal(await pipeline.transform({ model: "missing" }, context("missing", "anthropic-messages")), undefined);
	assert.equal(await pipeline.transform({ model: "responses" }, context("responses", "anthropic-messages")), undefined);
	const anthropic = await pipeline.transform({ model: "claude" }, context("claude", "anthropic-messages")) as any;
	assert.equal(typeof anthropic.metadata.user_id, "string");
});

test("无法安全提取的首条系统内容保留原 payload", async () => {
	await seed(true);
	const payload = { model: "responses", service_tier: "auto", input: [{ role: "developer", content: [{ type: "input_image", image_url: "local-fixture" }] }] };
	assert.equal(await createRequestPipeline().transform(payload, context("responses", "openai-responses")), undefined);
});

test("ClaudeCode 原生 1M 仅在开启且上下文窗口达到 1M 时注入 beta 请求头", () => {
	// 1. 未开启 1M 时，即使窗口为 1M 也不带 1M beta
	const headersWithoutOptIn = getClientHeadersForProfile("claude-code", "anthropic-messages", {}, {}, {}, 1_000_000);
	assert.equal(headersWithoutOptIn?.["anthropic-beta"]?.includes("context-1m-2025-08-07"), false);

	// 2. 开启 1M 但上下文小于 1M（如 200k）时，设置无效，不注入 1M beta
	const headersSmallContext = getClientHeadersForProfile("claude-code", "anthropic-messages", {}, {}, { claudeCode1mContext: true }, 200_000);
	assert.equal(headersSmallContext?.["anthropic-beta"]?.includes("context-1m-2025-08-07"), false);

	// 3. 开启 1M 且上下文达到 1M（1000000）时，正常注入 1M beta
	const headers1m = getClientHeadersForProfile("claude-code", "anthropic-messages", {}, {}, { claudeCode1mContext: true }, 1_000_000);
	assert.equal(headers1m?.["anthropic-beta"]?.includes("context-1m-2025-08-07"), true);

	// 4. 通过 buildModelRequestHeaders 端到端测试
	const ccProvider: StoredProvider = {
		name: "ClaudeGateway", api: "anthropic-messages", baseUrl: "https://gw.test/v1", managed: true,
		clientHeaderProfile: "recommended",
		models: [],
	};
	const model200k: StoredModel = { id: "sonnet-200k", reasoning: true, input: ["text"], contextWindow: 200_000, maxTokens: 16000, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, compat: { claudeCode1mContext: true } };
	const model1m: StoredModel = { id: "sonnet-1m", reasoning: true, input: ["text"], contextWindow: 1_000_000, maxTokens: 32000, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, compat: { claudeCode1mContext: true } };

	const headersResult200k = buildModelRequestHeaders(ccProvider, model200k, {}, {})!;
	assert.equal(headersResult200k["anthropic-beta"]?.includes("context-1m-2025-08-07"), false);

	const headersResult1m = buildModelRequestHeaders(ccProvider, model1m, {}, {})!;
	assert.equal(headersResult1m["anthropic-beta"]?.includes("context-1m-2025-08-07"), true);
});

test("ModelDraft 正确同步与持久化 claudeCode1mContext 兼容设置", async () => {
	const { buildModelFromDraft, createModelDraftFromStoredModel } = await import("../state-document.ts");
	const ccProvider: StoredProvider = {
		name: "ClaudeGateway", api: "anthropic-messages", baseUrl: "https://gw.test/v1", managed: true,
		clientHeaderProfile: "claude-code",
		models: [],
	};
	const stored1mModel: StoredModel = {
		id: "sonnet-test", reasoning: true, input: ["text"], contextWindow: 1_000_000, maxTokens: 32000,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, compat: { claudeCode1mContext: true },
	};
	const draft = createModelDraftFromStoredModel("ClaudeGateway", ccProvider, stored1mModel);
	assert.equal(draft.claudeCode1mContext, true);
	assert.equal(draft.contextWindow, 1_000_000);

	// 保存时保留该 compat
	const savedModel = buildModelFromDraft(stored1mModel, draft, ccProvider.compat);
	assert.equal(savedModel.compat?.claudeCode1mContext, true);

	// 关闭开关后保存
	draft.claudeCode1mContext = false;
	const savedDisabled = buildModelFromDraft(stored1mModel, draft, ccProvider.compat);
	assert.equal(savedDisabled.compat?.claudeCode1mContext, undefined);
});

test("provider 级 forceAdaptiveThinking 下放到无覆盖模型且保存不写入显式 false", async () => {
	const { buildModelFromDraft, createModelDraftFromStoredModel } = await import("../state-document.ts");
	const provider: StoredProvider = {
		name: "ClaudeGateway", api: "anthropic-messages", baseUrl: "https://gw.test/v1", managed: true,
		clientHeaderProfile: "claude-code", compat: { forceAdaptiveThinking: true }, models: [],
	};
	const plainModel: StoredModel = {
		id: "sonnet", reasoning: true, input: ["text"], contextWindow: 200_000, maxTokens: 16000,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	};

	// 模型无自身覆盖时应继承 provider 级设置，草稿为 adaptive
	const draft = createModelDraftFromStoredModel("ClaudeGateway", provider, plainModel);
	assert.equal(draft.anthropicThinkingProtocol, "adaptive");

	// 未改动开关直接保存时，不得把显式 false 写进模型 compat
	draft.modelName = "renamed";
	const saved = buildModelFromDraft(plainModel, draft, provider.compat);
	assert.equal(saved.compat?.forceAdaptiveThinking, undefined);

	// 模型级显式关闭仍优先于 provider 级开启
	const legacyModel: StoredModel = { ...plainModel, id: "sonnet-legacy", compat: { forceAdaptiveThinking: false } };
	const legacyDraft = createModelDraftFromStoredModel("ClaudeGateway", provider, legacyModel);
	assert.equal(legacyDraft.anthropicThinkingProtocol, "legacy");
});
