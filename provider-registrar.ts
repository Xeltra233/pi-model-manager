// 将配置交给 Pi 构造原生 Provider，插件仅包装其请求传输。

import { ModelRuntime, type ExtensionAPI, type ProviderConfig } from "@earendil-works/pi-coding-agent";
import type { Provider } from "@earendil-works/pi-ai";
import { applyAnthropicThinkingCompat } from "./anthropic-thinking.ts";
import { isBuiltinProviderId } from "./builtin-model-catalog.ts";
import { mergeCompatSettings } from "./compat-settings.ts";
import { formatUnknownError } from "./common.ts";
import { t } from "./i18n.ts";
import { removeProviderLocalProxyRoutes } from "./local-proxy-service.ts";
import { getClientHeadersForProfile, mergeModelRequestHeaders } from "./presets/client-headers.ts";
import { normalizeThinkingLevelMap } from "./presets/thinking.ts";
import { createProviderTransport } from "./provider-transport.ts";
import { resolveRuntimeBaseUrl } from "./runtime-base-url.ts";
import type { ApiKind, BuiltInClientHeaderProfileId, CompatSettings, StateDocument, StoredClientHeaderCapture, StoredModel, StoredProvider, StoredRequestHeaderProfile } from "./types.ts";

type ProviderModelConfig = NonNullable<ProviderConfig["models"]>[number];
const REGISTERED_PROVIDER_CONFIGS = new Map<string, Provider>();
const emptyCredentialStore = {
	async read() { return undefined; },
	async list() { return []; },
	async modify() { return undefined; },
	async delete() {},
};

function buildProviderConfig(
	provider: StoredProvider,
	requestHeaderProfiles: Record<string, StoredRequestHeaderProfile> = {},
	clientHeaderCaptures: Partial<Record<BuiltInClientHeaderProfileId, StoredClientHeaderCapture>> = {},
): ProviderConfig {
	const apiKey = provider.apiKey?.trim();
	return {
		name: provider.name,
		baseUrl: resolveRuntimeBaseUrl(provider.api, provider.baseUrl),
		...(apiKey ? { apiKey } : {}),
		api: provider.api,
		headers: mergeModelRequestHeaders(undefined, provider.headers),
		authHeader: provider.authHeader,
		models: provider.models.map((model) => buildModelConfig(provider, model, requestHeaderProfiles, clientHeaderCaptures)),
	};
}

async function buildNativeProvider(providerId: string, provider: StoredProvider, config: ProviderConfig): Promise<Provider> {
	// [喵喵喵]: 复用 Pi 的认证解析与多协议分派；临时 runtime 不读取用户认证或模型文件，也不刷新网络目录。
	const runtime = await ModelRuntime.create({ credentials: emptyCredentialStore, modelsPath: null, allowModelNetwork: false, refreshOnCreate: false });
	runtime.registerProvider(providerId, config);
	const native = runtime.getProvider(providerId);
	if (!native) throw new Error(t("无法构建接入：{providerId}", { providerId }));
	return createProviderTransport(runtime, native, provider);
}

function asManagedApi(value: string | undefined, fallback: ApiKind): ApiKind {
	if (value === "openai-completions" || value === "openai-responses"
		|| value === "anthropic-messages" || value === "google-generative-ai") return value;
	return fallback;
}

// [喵喵喵]: 只接受 adaptive / extended 其中一种的模型，注册时就把 compat 定死；
// 否则 Pi 会按 legacy 发 thinking.type=enabled，Claude 4.7+ 直接 400。
function resolveModelCompat(provider: StoredProvider, model: StoredModel): CompatSettings | undefined {
	const merged = mergeCompatSettings(provider.compat, model.compat);
	if (!provider.managed) return merged;
	return applyAnthropicThinkingCompat(
		asManagedApi(model.api, provider.api),
		model.reasoning,
		model.id,
		merged,
	) ?? merged;
}

function resolveModelRuntimeBaseUrl(provider: StoredProvider, model: StoredModel): string | undefined {
	if (!model.baseUrl) return undefined;
	return resolveRuntimeBaseUrl(asManagedApi(model.api, provider.api), model.baseUrl);
}

export function buildModelRequestHeaders(
	provider: StoredProvider,
	model: StoredModel,
	requestHeaderProfiles: Record<string, StoredRequestHeaderProfile>,
	clientHeaderCaptures: Partial<Record<BuiltInClientHeaderProfileId, StoredClientHeaderCapture>>,
): Record<string, string> | undefined {
	if (!provider.managed) return model.headers ? { ...model.headers } : undefined;
	const customHeaders = provider.clientHeaderProfile === "custom"
		? resolveProviderCustomHeaders(provider, requestHeaderProfiles)
		: {};
	const effectiveApi = asManagedApi(model.api, provider.api);
	const effectiveCompat = resolveModelCompat(provider, model);
	const profileHeaders = getClientHeadersForProfile(
		provider.clientHeaderProfile,
		effectiveApi,
		customHeaders,
		clientHeaderCaptures,
		effectiveCompat,
		model.contextWindow,
	);
	const explicitHeaders = mergeModelRequestHeaders(provider.headers, model.headers);
	return provider.clientHeaderProfile === "custom"
		? mergeModelRequestHeaders(explicitHeaders, profileHeaders)
		: mergeModelRequestHeaders(profileHeaders, explicitHeaders);
}

function buildModelConfig(
	provider: StoredProvider,
	model: StoredModel,
	requestHeaderProfiles: Record<string, StoredRequestHeaderProfile>,
	clientHeaderCaptures: Partial<Record<BuiltInClientHeaderProfileId, StoredClientHeaderCapture>>,
): ProviderModelConfig {
	return {
		id: model.id,
		name: model.name ?? model.id,
		api: model.api as ProviderModelConfig["api"],
		baseUrl: resolveModelRuntimeBaseUrl(provider, model),
		reasoning: model.reasoning,
		thinkingLevelMap: normalizeThinkingLevelMap(
			asManagedApi(model.api, provider.api),
			model.reasoning,
			model.thinkingLevelMap,
			model.id,
		),
		input: model.input,
		cost: model.cost,
		contextWindow: model.contextWindow,
		maxTokens: model.maxTokens,
		headers: buildModelRequestHeaders(provider, model, requestHeaderProfiles, clientHeaderCaptures),
		compat: resolveModelCompat(provider, model) as ProviderModelConfig["compat"],
	};
}

function resolveProviderCustomHeaders(
	provider: StoredProvider,
	requestHeaderProfiles: Record<string, StoredRequestHeaderProfile>,
): Record<string, string> {
	const profileId = provider.requestHeaderProfileId;
	if (profileId && requestHeaderProfiles[profileId]) return requestHeaderProfiles[profileId].headers;
	return provider.customClientHeaders ?? {};
}

/** 注销本扩展实际注册的动态 provider，并同步清理代理路由和回滚快照。 */
export function unregisterManagedProvider(pi: ExtensionAPI, providerId: string): void {
	removeProviderLocalProxyRoutes(providerId);
	const wasRegistered = REGISTERED_PROVIDER_CONFIGS.delete(providerId);
	if (wasRegistered) pi.unregisterProvider(providerId);
}

async function canRegisterManagedProvider(providerId: string, provider: StoredProvider): Promise<boolean> {
	return provider.managed && provider.models.length > 0 && !(await isBuiltinProviderId(providerId));
}

function replaceManagedProviderConfig(
	pi: ExtensionAPI,
	providerId: string,
	nextConfig: Provider,
): void {
	const previousConfig = REGISTERED_PROVIDER_CONFIGS.get(providerId);
	try {
		if (previousConfig) pi.unregisterProvider(providerId);
		pi.registerProvider(nextConfig);
		REGISTERED_PROVIDER_CONFIGS.set(providerId, nextConfig);
	} catch (error) {
		let rollbackError: unknown;
		try {
			if (previousConfig) {
				pi.unregisterProvider(providerId);
				pi.registerProvider(previousConfig);
				REGISTERED_PROVIDER_CONFIGS.set(providerId, previousConfig);
			} else {
				REGISTERED_PROVIDER_CONFIGS.delete(providerId);
			}
		} catch (restoreError) {
			rollbackError = restoreError;
		}
		const rollbackNote = rollbackError
			? t("；恢复上一版 runtime 也失败：{error}", { error: formatUnknownError(rollbackError) })
			: previousConfig
				? t("；已恢复上一版 runtime")
				: t("；新配置未注册到当前 runtime");
		throw new Error(`${formatUnknownError(error)}${rollbackNote}`);
	}
}

/** 同步一个 provider 到当前会话 transport。新配置会先完整构建；替换失败时恢复上一版动态配置。 */
export async function reconcileProvider(
	pi: ExtensionAPI,
	providerId: string,
	provider: StoredProvider,
	requestHeaderProfiles: Record<string, StoredRequestHeaderProfile> = {},
	clientHeaderCaptures: Partial<Record<BuiltInClientHeaderProfileId, StoredClientHeaderCapture>> = {},
): Promise<void> {
	// registerProvider 的 models 会整体替换目录；内置 provider 必须保留 Pi 已合成的完整目录。
	if (!(await canRegisterManagedProvider(providerId, provider))) {
		unregisterManagedProvider(pi, providerId);
		return;
	}

	const config = buildProviderConfig(provider, requestHeaderProfiles, clientHeaderCaptures);
	const native = await buildNativeProvider(providerId, provider, config);
	replaceManagedProviderConfig(pi, providerId, native);
}

async function reconcileAllFromState(
	pi: ExtensionAPI,
	document: StateDocument,
): Promise<string[]> {
	const warnings: string[] = [];
	const activeProviderIds = new Set(
		[...document.managedProviderIds].filter((providerId) => document.providers[providerId]?.managed),
	);
	for (const providerId of REGISTERED_PROVIDER_CONFIGS.keys()) {
		if (!activeProviderIds.has(providerId)) unregisterManagedProvider(pi, providerId);
	}
	for (const [providerId, provider] of Object.entries(document.providers)) {
		try {
			await reconcileProvider(pi, providerId, provider, document.requestHeaderProfiles, document.clientHeaderCaptures);
		} catch (error) {
			warnings.push(`${providerId}: ${formatUnknownError(error)}`);
		}
	}
	return warnings;
}

/** factory 阶段仅注册模型 catalog，不启动长生命周期本地代理。 */
export async function registerCatalogFromState(pi: ExtensionAPI, document: StateDocument): Promise<string[]> {
	return reconcileAllFromState(pi, document);
}

/** session_start 激活当前 state 的完整 provider transport。 */
export async function registerAllFromState(pi: ExtensionAPI, document: StateDocument): Promise<string[]> {
	return reconcileAllFromState(pi, document);
}
