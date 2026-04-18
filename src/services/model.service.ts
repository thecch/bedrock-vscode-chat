import * as vscode from "vscode";
import type { LanguageModelChatInformation } from "vscode";
import type { BedrockModelSummary } from "../types";
import { BedrockClient } from "../clients/bedrock.client";
import { OpenRouterClient } from "./openrouter.client";
import { AuthenticationService } from "./authentication.service";
import { ConfigurationService } from "./configuration.service";
import { getModelProfile, buildConfigurationSchema } from "../profiles";
import { logger } from "../logger";

const DEFAULT_MAX_OUTPUT_TOKENS = 4096;
const DEFAULT_CONTEXT_LENGTH = 200000;

/**
 * Manages model information, capabilities, and metadata.
 * Coordinates between AWS Bedrock and OpenRouter data sources.
 */
export class ModelService {
	private bedrockClient: BedrockClient;
	private openRouterClient: OpenRouterClient;
	private chatEndpoints: { model: string; modelMaxPromptTokens: number }[] = [];

	constructor(
		private readonly authService: AuthenticationService,
		private readonly configService: ConfigurationService
	) {
		const region = this.configService.getRegion();
		this.bedrockClient = new BedrockClient(region);
		this.openRouterClient = new OpenRouterClient();
	}

	/**
	 * Handle configuration changes (e.g., region updates)
	 */
	handleConfigurationChange(): void {
		const region = this.configService.getRegion();
		this.bedrockClient.setRegion(region);
		logger.log("[Model Service] Configuration changed, region updated to:", region);
	}

	/**
	 * Fetch and prepare language model chat information
	 */
	async getLanguageModelChatInformation(
		silent: boolean = false
	): Promise<LanguageModelChatInformation[]> {
		const authConfig = await this.authService.getAuthConfig(silent);
		if (!authConfig) {
			return [];
		}

		const region = this.configService.getRegion();
		this.bedrockClient.setRegion(region);

		let models: BedrockModelSummary[];
		let availableProfileIds: Set<string>;

		try {
			const credentials = this.authService.getCredentials(authConfig);
			[models, availableProfileIds] = await Promise.all([
				this.bedrockClient.fetchModels(credentials),
				this.bedrockClient.fetchInferenceProfiles(credentials),
			]);
		} catch (err) {
			const errorMsg = err instanceof Error ? err.message : String(err);
			logger.error("[Model Service] Failed to fetch models", err);
			if (!silent) {
				vscode.window.showErrorMessage(`Failed to fetch Bedrock models: ${errorMsg}`);
			}
			return [];
		}

		const infos: LanguageModelChatInformation[] = [];
		const regionPrefix = region.split("-")[0];

		for (const m of models) {
			if (!m.responseStreamingSupported || !m.outputModalities.includes("TEXT")) {
				continue;
			}

			const inferenceProfileId = `${regionPrefix}.${m.modelId}`;
			const hasInferenceProfile = availableProfileIds.has(inferenceProfileId);
			const modelIdToUse = hasInferenceProfile ? inferenceProfileId : m.modelId;

			// Use profile-based token limits for known Claude models, fall back to OpenRouter
			const profile = getModelProfile(modelIdToUse);
			let maxInput: number;
			let maxOutput: number;

			if (profile.maxInputTokens > DEFAULT_CONTEXT_LENGTH || profile.maxOutputTokens > DEFAULT_MAX_OUTPUT_TOKENS) {
				// Known model with accurate profile data
				maxInput = profile.maxInputTokens;
				maxOutput = profile.maxOutputTokens;
			} else {
				// Unknown model — try OpenRouter, fall back to defaults
				const properties = await this.openRouterClient.getModelProperties(modelIdToUse);
				maxInput = properties?.contextLength ?? DEFAULT_CONTEXT_LENGTH;
				maxOutput = properties?.maxOutputTokens ?? DEFAULT_MAX_OUTPUT_TOKENS;
			}

			// Apply per-model token overrides from settings
			const modelOverride = this.configService.getModelOverride(modelIdToUse);
			if (modelOverride.maxInputTokens) maxInput = modelOverride.maxInputTokens;
			if (modelOverride.maxOutputTokens) maxOutput = modelOverride.maxOutputTokens;
			const vision = m.inputModalities.includes("IMAGE");

			const configSchema = buildConfigurationSchema(profile);
			const modelInfo = {
				id: modelIdToUse,
				name: m.modelName,
				tooltip: `AWS Bedrock - ${m.providerName}${hasInferenceProfile ? ' (Cross-Region)' : ''}`,
				detail: `${m.providerName} • ${hasInferenceProfile ? 'Multi-Region' : region}`,
				family: "bedrock",
				version: "1.0.0",
				maxInputTokens: maxInput,
				maxOutputTokens: maxOutput,
				...(configSchema ? { configurationSchema: configSchema } : {}),
				capabilities: {
					toolCalling: true,
					imageInput: vision,
				},
			} satisfies LanguageModelChatInformation;
			infos.push(modelInfo);
		}

		// Sort preferred model to front so VS Code selects it as default
		const defaultModel = this.configService.getDefaultModel();
		if (defaultModel) {
			const idx = infos.findIndex(i => i.id.includes(defaultModel));
			if (idx > 0) {
				const preferred = infos.splice(idx, 1);
				infos.unshift(...preferred);
				logger.log(`[Model Service] Moved preferred model to front: ${preferred[0].id}`);
			} else if (idx === 0) {
				logger.log(`[Model Service] Preferred model already first: ${infos[0].id}`);
			} else {
				logger.log(`[Model Service] Preferred model not found matching "${defaultModel}". Available:`, infos.map(i => i.id));
			}
		}

		this.chatEndpoints = infos.map((info) => ({
			model: info.id,
			modelMaxPromptTokens: info.maxInputTokens + info.maxOutputTokens,
		}));

		return infos;
	}

	/**
	 * Check if a model supports thinking/reasoning
	 */
	supportsThinking(modelId: string): boolean {
		const profile = getModelProfile(modelId);
		return profile.supportsThinking || profile.supportsAdaptiveThinking;
	}

	/**
	 * Check if a model supports adaptive thinking
	 */
	supportsAdaptiveThinking(modelId: string): boolean {
		const profile = getModelProfile(modelId);
		return profile.supportsAdaptiveThinking;
	}

	/**
	 * Check if a model supports thinking effort control
	 */
	supportsThinkingEffort(modelId: string): boolean {
		const profile = getModelProfile(modelId);
		return profile.supportsThinkingEffort;
	}

	/**
	 * Get cached chat endpoints
	 */
	getChatEndpoints(): { model: string; modelMaxPromptTokens: number }[] {
		return this.chatEndpoints;
	}
}
