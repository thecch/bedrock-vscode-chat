import * as vscode from "vscode";
import {
	CancellationToken,
	LanguageModelChatInformation,
	LanguageModelChatMessage,
	LanguageModelChatProvider,
	LanguageModelChatRequestHandleOptions,
	LanguageModelResponsePart,
	Progress,
} from "vscode";
import { ConverseStreamCommandInput } from "@aws-sdk/client-bedrock-runtime";
import { BedrockAPIClient } from "./bedrock-client";
import { StreamProcessor } from "./stream-processor";
import { convertMessages } from "./converters/messages";
import { convertTools } from "./converters/tools";
import { validateRequest } from "./validation";
import { logger } from "./logger";
import type { AuthConfig, AuthMethod, BedrockChatSettings, ThinkingBlock } from "./types";

const DEFAULT_MAX_OUTPUT_TOKENS = 4096;
const DEFAULT_CONTEXT_LENGTH = 200000;

function getSettings(): BedrockChatSettings {
	const config = vscode.workspace.getConfiguration("bedrockChat");
	const thinkingEnabled = config.get<boolean>("thinking.enabled", true);
	const thinkingBudgetTokens = Math.max(1024, config.get<number>("thinking.budgetTokens", 25600));
	const thinkingEffort = config.get<string>("thinking.effort", "max");
	const context1MEnabled = config.get<boolean>("context1M.enabled", true);
	const contextLimit = config.get<number>("contextLimit", 400000);
	const maxOutputTokens = config.get<number>("maxOutputTokens", 128000);
	return {
		thinking: { enabled: thinkingEnabled, budgetTokens: thinkingBudgetTokens, effort: thinkingEffort },
		context1M: { enabled: context1MEnabled },
		contextLimit,
		maxOutputTokens,
	};
}

export class BedrockChatModelProvider implements LanguageModelChatProvider {
	private client: BedrockAPIClient;
	private streamProcessor: StreamProcessor;
	private chatEndpoints: { model: string; modelMaxPromptTokens: number }[] = [];
	private lastThinkingBlock: ThinkingBlock | undefined = undefined;

	constructor(
		private readonly secrets: vscode.SecretStorage,
		private readonly globalState: vscode.Memento,
		private readonly userAgent: string
	) {
		const region = this.globalState.get<string>("bedrock.region") ?? "us-east-1";
		this.client = new BedrockAPIClient(region);
		this.streamProcessor = new StreamProcessor();
	}

	private estimateMessagesTokens(msgs: readonly vscode.LanguageModelChatMessage[]): number {
		let total = 0;
		for (const m of msgs) {
			for (const part of m.content) {
				if (part instanceof vscode.LanguageModelTextPart) {
					total += Math.ceil(part.value.length / 4);
				}
			}
		}
		return total;
	}

	private estimateToolTokens(
		toolConfig: { tools: Array<{ toolSpec: { name: string; description?: string; inputSchema: { json: object } } }> } | undefined
	): number {
		if (!toolConfig || toolConfig.tools.length === 0) {
			return 0;
		}
		try {
			const json = JSON.stringify(toolConfig);
			return Math.ceil(json.length / 4);
		} catch {
			return 0;
		}
	}

	async prepareLanguageModelChatInformation(
		options: { silent: boolean },
		_token: CancellationToken
	): Promise<LanguageModelChatInformation[]> {
		return this.provideLanguageModelChatInformation(options, _token);
	}

	async provideLanguageModelChatInformation(
		options: { silent: boolean },
		_token: CancellationToken
	): Promise<LanguageModelChatInformation[]> {
		const authConfig = await this.getAuthConfig(options.silent ?? false);
		if (!authConfig) {
			return [];
		}

		const region = this.globalState.get<string>("bedrock.region") ?? "us-east-1";
		this.client.setRegion(region);

		let models, availableProfileIds;
		try {
			[models, availableProfileIds] = await Promise.all([
				this.client.fetchModels(authConfig),
				this.client.fetchInferenceProfiles(authConfig),
			]);
		} catch (err) {
			const errorMsg = err instanceof Error ? err.message : String(err);
			logger.error("[Bedrock Model Provider] Failed to fetch models", err);
			if (!options.silent) {
				vscode.window.showErrorMessage(`Failed to fetch Bedrock models: ${errorMsg}`);
			}
			return [];
		}

		const infos: LanguageModelChatInformation[] = [];
		const regionPrefix = region.split("-")[0];
		const settings = getSettings();

		for (const m of models) {
			if (!m.responseStreamingSupported || !m.outputModalities.includes("TEXT")) {
				continue;
			}

			const isClaudeModel = m.modelId.toLowerCase().includes("anthropic") || m.modelId.toLowerCase().includes("claude");
			const contextLen = isClaudeModel && settings.context1M.enabled ? settings.contextLimit : DEFAULT_CONTEXT_LENGTH;
			const maxOutput = isClaudeModel && settings.thinking.enabled ? settings.maxOutputTokens : DEFAULT_MAX_OUTPUT_TOKENS;
			const maxInput = Math.max(1, contextLen - maxOutput);
			const vision = m.inputModalities.includes("IMAGE");

			const inferenceProfileId = `${regionPrefix}.${m.modelId}`;
			const hasInferenceProfile = availableProfileIds.has(inferenceProfileId);

			const contextLabel = isClaudeModel && settings.context1M.enabled ? `${Math.round(settings.contextLimit / 1000)}k ctx (1M model)` : '';
			const thinkingLabel = isClaudeModel && settings.thinking.enabled ? `thinking:${settings.thinking.effort}` : '';
			const extraDetail = [contextLabel, thinkingLabel].filter(Boolean).join(' • ');

			const modelInfo: LanguageModelChatInformation = {
				id: hasInferenceProfile ? inferenceProfileId : m.modelId,
				name: m.modelName,
				tooltip: `AWS Bedrock - ${m.providerName}${hasInferenceProfile ? ' (Cross-Region)' : ''}${extraDetail ? ' • ' + extraDetail : ''}`,
				detail: `${m.providerName} • ${hasInferenceProfile ? 'Multi-Region' : region}${extraDetail ? ' • ' + extraDetail : ''}`,
				family: "bedrock",
				version: "1.0.0",
				maxInputTokens: maxInput,
				maxOutputTokens: maxOutput,
				capabilities: {
					toolCalling: true,
					imageInput: vision,
				},
			};
			infos.push(modelInfo);
		}

		this.chatEndpoints = infos.map((info) => ({
			model: info.id,
			modelMaxPromptTokens: info.maxInputTokens + info.maxOutputTokens,
		}));

		return infos;
	}

	async provideLanguageModelChatResponse(
		model: LanguageModelChatInformation,
		messages: readonly LanguageModelChatMessage[],
		options: LanguageModelChatRequestHandleOptions,
		progress: Progress<LanguageModelResponsePart>,
		token: CancellationToken
	): Promise<void> {
		const trackingProgress: Progress<LanguageModelResponsePart> = {
			report: (part) => {
				try {
					progress.report(part);
				} catch (e) {
					logger.error("[Bedrock Model Provider] Progress.report failed", {
						modelId: model.id,
						error: e instanceof Error ? { name: e.name, message: e.message } : String(e),
					});
				}
			},
		};

		try {
			const authConfig = await this.getAuthConfig();
			if (!authConfig) {
				throw new Error("Bedrock authentication not configured");
			}

			logger.log("[Bedrock Model Provider] Converting messages, count:", messages.length);
			messages.forEach((msg, idx) => {
				const partTypes = msg.content.map(p => {
					if (p instanceof vscode.LanguageModelTextPart) return 'text';
					if (p instanceof vscode.LanguageModelToolCallPart) return 'toolCall';
					if (typeof p === 'object' && p !== null && 'mimeType' in p && (p as any).mimeType?.startsWith('image/')) return 'image';
					return 'toolResult';
				});
				logger.log(`[Bedrock Model Provider] Message ${idx} (${msg.role}):`, partTypes);
			});

			const converted = convertMessages(messages, model.id);
			validateRequest(messages);

			logger.log("[Bedrock Model Provider] Converted to Bedrock messages:", converted.messages.length);
			converted.messages.forEach((msg, idx) => {
				const contentTypes = msg.content.map(c => {
					if ('text' in c) return 'text';
					if ('image' in c) return 'image';
					if ('toolUse' in c) return 'toolUse';
					return 'toolResult';
				});
				logger.log(`[Bedrock Model Provider] Bedrock message ${idx} (${msg.role}):`, contentTypes);
			});

			const toolConfig = convertTools(options, model.id);

			if (options.tools && options.tools.length > 128) {
				throw new Error("Cannot have more than 128 tools per request.");
			}

			const inputTokenCount = this.estimateMessagesTokens(messages);
			const toolTokenCount = this.estimateToolTokens(toolConfig);
			const tokenLimit = Math.max(1, model.maxInputTokens);
			if (inputTokenCount + toolTokenCount > tokenLimit) {
				logger.error("[Bedrock Model Provider] Message exceeds token limit", {
					total: inputTokenCount + toolTokenCount,
					tokenLimit,
				});
				throw new Error("Message exceeds token limit.");
			}

			const isClaudeModel = model.id.toLowerCase().includes("anthropic") || model.id.toLowerCase().includes("claude");
			const settings = getSettings();
			let useThinking = isClaudeModel && settings.thinking.enabled;

			// Disable thinking on multi-turn tool-use continuations
			if (useThinking) {
				const assistantMsgCount = messages.filter(m => m.role === vscode.LanguageModelChatMessageRole.Assistant).length;
				if (assistantMsgCount > 1) {
					logger.log("[Bedrock Model Provider] Disabling thinking - multiple assistant messages (tool-use continuation)", { assistantMsgCount });
					useThinking = false;
					this.lastThinkingBlock = undefined;
				} else if (assistantMsgCount === 1 && !this.lastThinkingBlock?.signature) {
					logger.log("[Bedrock Model Provider] Disabling thinking - no stored thinking block for previous assistant message");
					useThinking = false;
				}
			}

			const effectiveMaxTokens = useThinking
				? settings.maxOutputTokens
				: Math.min(options.modelOptions?.max_tokens || DEFAULT_MAX_OUTPUT_TOKENS, model.maxOutputTokens);

			const requestInput: ConverseStreamCommandInput = {
				modelId: model.id,
				messages: converted.messages as any,
				inferenceConfig: {
					maxTokens: effectiveMaxTokens,
				},
			};

			// Only set temperature/topP for non-Claude models
			if (!isClaudeModel) {
				requestInput.inferenceConfig!.temperature = options.modelOptions?.temperature ?? 0.1;
				requestInput.inferenceConfig!.topP = options.modelOptions?.top_p ?? 1;
			}

			if (useThinking) {
				const betaHeaders: string[] = [];
				if (settings.context1M.enabled) {
					betaHeaders.push("context-1m-2025-08-07");
				}
				betaHeaders.push("effort-2025-11-24");

				(requestInput as any).additionalModelRequestFields = {
					thinking: {
						type: "enabled",
						budget_tokens: settings.thinking.budgetTokens,
					},
					output_config: {
						effort: settings.thinking.effort,
					},
					anthropic_beta: betaHeaders,
				};

				logger.log("[Bedrock Model Provider] Extended thinking enabled", {
					budgetTokens: settings.thinking.budgetTokens,
					maxTokens: effectiveMaxTokens,
					effort: settings.thinking.effort,
					context1M: settings.context1M.enabled,
					modelId: model.id,
				});
			}

			if (converted.system.length > 0) {
				requestInput.system = converted.system as any;
			}

			if (options.modelOptions) {
				const mo = options.modelOptions as Record<string, unknown>;
				if (!isClaudeModel && typeof mo.top_p === "number") {
					requestInput.inferenceConfig!.topP = mo.top_p as number;
				}
				if (typeof mo.stop === "string") {
					requestInput.inferenceConfig!.stopSequences = [mo.stop];
				} else if (Array.isArray(mo.stop)) {
					requestInput.inferenceConfig!.stopSequences = mo.stop;
				}
			}

			if (toolConfig) {
				requestInput.toolConfig = toolConfig as any;
			}

			logger.log("[Bedrock Model Provider] Starting streaming request");
			const stream = await this.client.startConversationStream(authConfig, requestInput);

			logger.log("[Bedrock Model Provider] Processing stream events");
			const result = await this.streamProcessor.processStream(stream, trackingProgress, token);

			if (useThinking && result?.thinkingBlock?.signature) {
				this.lastThinkingBlock = result.thinkingBlock;
				logger.log("[Bedrock Model Provider] Stored thinking block for next request", {
					signatureLength: result.thinkingBlock.signature.length,
					textLength: result.thinkingBlock.text.length,
				});
			} else if (useThinking) {
				this.lastThinkingBlock = undefined;
			}

			logger.log("[Bedrock Model Provider] Finished processing stream");
		} catch (err) {
			const errorMsg = err instanceof Error ? err.message : String(err);
			logger.error("[Bedrock Model Provider] Chat request failed", {
				modelId: model.id,
				messageCount: messages.length,
				error: err instanceof Error ? { name: err.name, message: err.message } : String(err),
			});
			vscode.window.showErrorMessage(`Bedrock chat request failed: ${errorMsg}`);
			throw err;
		}
	}

	async provideTokenCount(
		model: LanguageModelChatInformation,
		text: string | LanguageModelChatMessage,
		_token: CancellationToken
	): Promise<number> {
		if (typeof text === "string") {
			return Math.ceil(text.length / 4);
		} else {
			let totalTokens = 0;
			for (const part of text.content) {
				if (part instanceof vscode.LanguageModelTextPart) {
					totalTokens += Math.ceil(part.value.length / 4);
				}
			}
			return totalTokens;
		}
	}

	private async getAuthConfig(silent: boolean = false): Promise<AuthConfig | undefined> {
		const method = this.globalState.get<AuthMethod>("bedrock.authMethod") ?? "api-key";

		if (method === "api-key") {
			let apiKey = await this.secrets.get("bedrock.apiKey");
			if (!apiKey && !silent) {
				const entered = await vscode.window.showInputBox({
					title: "AWS Bedrock API Key",
					prompt: "Enter your AWS Bedrock API key",
					ignoreFocusOut: true,
					password: true,
				});
				if (entered && entered.trim()) {
					apiKey = entered.trim();
					await this.secrets.store("bedrock.apiKey", apiKey);
				}
			}
			if (!apiKey) {
				return undefined;
			}
			return { method: "api-key", apiKey };
		}

		if (method === "profile") {
			const profile = this.globalState.get<string>("bedrock.profile");
			if (!profile) {
				if (!silent) {
					vscode.window.showErrorMessage("AWS profile not configured. Please run 'Manage AWS Bedrock Provider'.");
				}
				return undefined;
			}
			return { method: "profile", profile };
		}

		if (method === "access-keys") {
			const accessKeyId = await this.secrets.get("bedrock.accessKeyId");
			const secretAccessKey = await this.secrets.get("bedrock.secretAccessKey");
			const sessionToken = await this.secrets.get("bedrock.sessionToken");

			if (!accessKeyId || !secretAccessKey) {
				if (!silent) {
					vscode.window.showErrorMessage("AWS access keys not configured. Please run 'Manage AWS Bedrock Provider'.");
				}
				return undefined;
			}

			return {
				method: "access-keys",
				accessKeyId,
				secretAccessKey,
				...(sessionToken && { sessionToken }),
			};
		}

		return undefined;
	}
}
