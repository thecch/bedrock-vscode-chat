/**
 * Model profile system for handling provider-specific capabilities and token limits.
 * Provides accurate metadata for Claude models without external API calls.
 */

export interface ModelProfile {
	/** Whether the model supports the toolChoice parameter */
	supportsToolChoice: boolean;
	/** Format to use for tool result content ('text' or 'json') */
	toolResultFormat: 'text' | 'json';
	/** Whether the model supports thinking with type: "enabled" (requires budget_tokens) */
	supportsThinking: boolean;
	/** Whether the model supports thinking with type: "adaptive" (no budget needed) */
	supportsAdaptiveThinking: boolean;
	/** Whether the model supports the effort parameter (output_config.effort) */
	supportsThinkingEffort: boolean;
	/** Whether the model supports 1M context window */
	supports1MContext: boolean;
	/** Maximum input tokens for this model */
	maxInputTokens: number;
	/** Maximum output tokens for this model */
	maxOutputTokens: number;
	/** Effort levels supported by this model for the model picker dropdown */
	supportedEffortLevels?: readonly string[];
}

/**
 * Known Claude model profiles with accurate token limits and capability flags.
 * Patterns are matched in order (most specific first) against the model ID.
 */
const EFFORT_LEVELS_FULL = ['low', 'medium', 'high', 'max'] as const;

const CLAUDE_PROFILES: Array<{ pattern: string; profile: Omit<ModelProfile, 'supportsToolChoice' | 'toolResultFormat'> }> = [
	{
		pattern: 'claude-opus-4-7',
		profile: { supportsThinking: true, supportsAdaptiveThinking: true, supportsThinkingEffort: true, supports1MContext: true, maxInputTokens: 200_000, maxOutputTokens: 128_000, supportedEffortLevels: EFFORT_LEVELS_FULL },
	},
	{
		pattern: 'claude-opus-4-6',
		profile: { supportsThinking: true, supportsAdaptiveThinking: true, supportsThinkingEffort: true, supports1MContext: true, maxInputTokens: 200_000, maxOutputTokens: 128_000, supportedEffortLevels: EFFORT_LEVELS_FULL },
	},
	{
		pattern: 'claude-opus-4-5',
		profile: { supportsThinking: true, supportsAdaptiveThinking: true, supportsThinkingEffort: true, supports1MContext: false, maxInputTokens: 200_000, maxOutputTokens: 32_000, supportedEffortLevels: EFFORT_LEVELS_FULL },
	},
	{
		pattern: 'claude-opus-4-1',
		profile: { supportsThinking: true, supportsAdaptiveThinking: false, supportsThinkingEffort: false, supports1MContext: false, maxInputTokens: 200_000, maxOutputTokens: 32_000 },
	},
	{
		pattern: 'claude-sonnet-4-6',
		profile: { supportsThinking: true, supportsAdaptiveThinking: true, supportsThinkingEffort: true, supports1MContext: true, maxInputTokens: 200_000, maxOutputTokens: 64_000, supportedEffortLevels: EFFORT_LEVELS_FULL },
	},
	{
		pattern: 'claude-sonnet-4-5',
		profile: { supportsThinking: true, supportsAdaptiveThinking: true, supportsThinkingEffort: true, supports1MContext: true, maxInputTokens: 200_000, maxOutputTokens: 64_000, supportedEffortLevels: EFFORT_LEVELS_FULL },
	},
	{
		pattern: 'claude-sonnet-4-v',
		profile: { supportsThinking: true, supportsAdaptiveThinking: false, supportsThinkingEffort: false, supports1MContext: true, maxInputTokens: 200_000, maxOutputTokens: 64_000 },
	},
	{
		pattern: 'claude-haiku-4-5',
		profile: { supportsThinking: true, supportsAdaptiveThinking: false, supportsThinkingEffort: false, supports1MContext: false, maxInputTokens: 200_000, maxOutputTokens: 64_000 },
	},
	{
		pattern: 'claude-3-7-sonnet',
		profile: { supportsThinking: true, supportsAdaptiveThinking: false, supportsThinkingEffort: false, supports1MContext: false, maxInputTokens: 200_000, maxOutputTokens: 64_000 },
	},
	{
		pattern: 'claude-3-5-sonnet',
		profile: { supportsThinking: false, supportsAdaptiveThinking: false, supportsThinkingEffort: false, supports1MContext: false, maxInputTokens: 200_000, maxOutputTokens: 8_192 },
	},
	{
		pattern: 'claude-3-5-haiku',
		profile: { supportsThinking: false, supportsAdaptiveThinking: false, supportsThinkingEffort: false, supports1MContext: false, maxInputTokens: 200_000, maxOutputTokens: 8_192 },
	},
	{
		pattern: 'claude-3-sonnet',
		profile: { supportsThinking: false, supportsAdaptiveThinking: false, supportsThinkingEffort: false, supports1MContext: false, maxInputTokens: 200_000, maxOutputTokens: 4_096 },
	},
	{
		pattern: 'claude-3-haiku',
		profile: { supportsThinking: false, supportsAdaptiveThinking: false, supportsThinkingEffort: false, supports1MContext: false, maxInputTokens: 200_000, maxOutputTokens: 4_096 },
	},
];

const DEFAULT_PROFILE: ModelProfile = {
	supportsToolChoice: false,
	toolResultFormat: 'text',
	supportsThinking: false,
	supportsAdaptiveThinking: false,
	supportsThinkingEffort: false,
	supports1MContext: false,
	maxInputTokens: 200_000,
	maxOutputTokens: 4_096,
};

/**
 * Strip region/global prefix from a Bedrock model ID.
 * e.g., "us.anthropic.claude-opus-4-6-v1:0" → "anthropic.claude-opus-4-6-v1:0"
 */
function stripRegionPrefix(modelId: string): string {
	const parts = modelId.split('.');
	if (parts.length > 2) {
		const prefix = parts[0].toLowerCase();
		if (prefix.length <= 6) {
			return parts.slice(1).join('.');
		}
	}
	return modelId;
}

/**
 * Get the model profile for a given Bedrock model ID.
 * Returns accurate capabilities and token limits for Claude models,
 * with sensible defaults for other providers.
 *
 * @param modelId The full Bedrock model ID (e.g., "us.anthropic.claude-opus-4-6-v1:0")
 * @returns Model profile with capabilities and token limits
 */
export function getModelProfile(modelId: string): ModelProfile {
	const normalized = stripRegionPrefix(modelId);
	const parts = normalized.split('.');

	if (parts.length < 2) {
		return { ...DEFAULT_PROFILE };
	}

	const provider = parts[0];
	const modelPart = parts.slice(1).join('.');

	switch (provider) {
		case 'anthropic': {
			for (const entry of CLAUDE_PROFILES) {
				if (modelPart.includes(entry.pattern)) {
					return {
						supportsToolChoice: true,
						toolResultFormat: 'text',
						...entry.profile,
					};
				}
			}
			// Unknown Claude model — tool choice but no thinking
			return { ...DEFAULT_PROFILE, supportsToolChoice: true };
		}

		case 'mistral':
			return { ...DEFAULT_PROFILE, toolResultFormat: 'json' };

		case 'amazon':
			if (modelId.includes('nova')) {
				return { ...DEFAULT_PROFILE, supportsToolChoice: true };
			}
			return { ...DEFAULT_PROFILE };

		case 'cohere':
		case 'meta':
		case 'ai21':
		default:
			return { ...DEFAULT_PROFILE };
	}
}

/**
 * Build a configurationSchema for the model picker's "Thinking Effort" dropdown.
 * When a model supports thinking effort, this makes VS Code show the dropdown
 * in the model picker UI so the user can select the effort level directly.
 *
 * @param profile The model profile
 * @returns A configurationSchema object or undefined
 */
export function buildConfigurationSchema(profile: ModelProfile): Record<string, unknown> | undefined {
	if (!profile.supportedEffortLevels || profile.supportedEffortLevels.length === 0) {
		return undefined;
	}

	const levels = profile.supportedEffortLevels;

	const labelMap: Record<string, string> = {
		low: 'Low',
		medium: 'Medium',
		high: 'High',
		max: 'Max',
	};

	const descriptionMap: Record<string, string> = {
		low: 'Faster responses with less reasoning',
		medium: 'Balanced reasoning and speed',
		high: 'Greater reasoning depth but slower',
		max: 'Maximum reasoning depth',
	};

	return {
		properties: {
			reasoningEffort: {
				type: 'string',
				title: 'Thinking Effort',
				enum: [...levels],
				enumItemLabels: levels.map(l => labelMap[l] ?? l),
				enumDescriptions: levels.map(l => descriptionMap[l] ?? l),
				default: 'high',
				group: 'navigation',
			},
		},
	};
}
