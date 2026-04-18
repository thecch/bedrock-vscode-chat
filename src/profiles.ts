/**
 * Model profile system for handling provider-specific capabilities and token limits.
 * Provides accurate metadata for Claude models without external API calls.
 *
 * Claude model IDs encode family + version in a consistent naming convention, so
 * capabilities and token limits are derived algorithmically rather than maintained
 * as a per-model lookup table. New models following the convention are handled
 * automatically without code changes.
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

const EFFORT_LEVELS_FULL = ['low', 'medium', 'high', 'max'] as const;

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
 * Parse a Claude model ID part into family + version numbers.
 *
 * Handles two naming conventions Anthropic uses on Bedrock:
 *   New: claude-{family}-{major}-{minor}-v{n}:{n}  (e.g. claude-opus-4-6-v1:0)
 *   Old: claude-{major}-{minor}-{family}-...        (e.g. claude-3-7-sonnet-20250219-v1:0)
 */
function parseClaudeVersion(modelPart: string): { family: string; major: number; minor: number } | undefined {
	// New naming: claude-{family}-{major}-{minor}
	let m = modelPart.match(/claude-(opus|sonnet|haiku)-(\d+)-(\d+)/);
	if (m) return { family: m[1], major: parseInt(m[2]), minor: parseInt(m[3]) };

	// New naming with only major: claude-{family}-{major}-v
	m = modelPart.match(/claude-(opus|sonnet|haiku)-(\d+)-v/);
	if (m) return { family: m[1], major: parseInt(m[2]), minor: 0 };

	// Old naming: claude-{major}-{minor}-{family}
	m = modelPart.match(/claude-(\d+)-(\d+)-(sonnet|haiku|opus)/);
	if (m) return { family: m[3], major: parseInt(m[1]), minor: parseInt(m[2]) };

	// Old naming without minor: claude-{major}-{family}
	m = modelPart.match(/claude-(\d+)-(sonnet|haiku|opus)/);
	if (m) return { family: m[2], major: parseInt(m[1]), minor: 0 };

	return undefined;
}

/**
 * Derive Claude capability flags and token limits from parsed version info.
 *
 * Rules based on Anthropic's capability rollout by generation:
 *   - Claude 4 Opus:          adaptive thinking + effort, 128K output (4.5+), 32K (4.1)
 *   - Claude 4 Sonnet 4.5+:   adaptive thinking + effort, 64K output
 *   - Claude 4 Sonnet <4.5:   thinking only, 64K output
 *   - Claude 4 Haiku:         thinking only, 64K output
 *   - Claude 3.7:             thinking (enabled) only, 64K output
 *   - Claude 3.5:             no thinking, 8K output
 *   - Claude 3.0:             no thinking, 4K output
 *   - Claude 5+:              assume full capabilities (forward-compatible)
 */
function deriveClaudeCapabilities(family: string, major: number, minor: number): Omit<ModelProfile, 'supportsToolChoice' | 'toolResultFormat'> {
	if (major >= 5) {
		// Future generation — assume full capabilities
		return { supportsThinking: true, supportsAdaptiveThinking: true, supportsThinkingEffort: true, supports1MContext: true, maxInputTokens: 200_000, maxOutputTokens: 128_000, supportedEffortLevels: EFFORT_LEVELS_FULL };
	}

	if (major === 4) {
		if (family === 'opus') {
			const maxOutput = minor >= 5 ? 128_000 : 32_000;
			return { supportsThinking: true, supportsAdaptiveThinking: true, supportsThinkingEffort: true, supports1MContext: true, maxInputTokens: 200_000, maxOutputTokens: maxOutput, supportedEffortLevels: EFFORT_LEVELS_FULL };
		}
		if (family === 'sonnet') {
			if (minor >= 5) {
				return { supportsThinking: true, supportsAdaptiveThinking: true, supportsThinkingEffort: true, supports1MContext: true, maxInputTokens: 200_000, maxOutputTokens: 64_000, supportedEffortLevels: EFFORT_LEVELS_FULL };
			}
			// Sonnet 4.0 (original): thinking but no adaptive/effort
			return { supportsThinking: true, supportsAdaptiveThinking: false, supportsThinkingEffort: false, supports1MContext: true, maxInputTokens: 200_000, maxOutputTokens: 64_000 };
		}
		if (family === 'haiku') {
			return { supportsThinking: true, supportsAdaptiveThinking: false, supportsThinkingEffort: false, supports1MContext: false, maxInputTokens: 200_000, maxOutputTokens: 64_000 };
		}
		// Unknown Claude 4 family — assume adaptive + effort
		return { supportsThinking: true, supportsAdaptiveThinking: true, supportsThinkingEffort: true, supports1MContext: true, maxInputTokens: 200_000, maxOutputTokens: 64_000, supportedEffortLevels: EFFORT_LEVELS_FULL };
	}

	if (major === 3) {
		if (minor >= 7) {
			// Claude 3.7 Sonnet: extended thinking (enabled only), 64K output
			return { supportsThinking: true, supportsAdaptiveThinking: false, supportsThinkingEffort: false, supports1MContext: false, maxInputTokens: 200_000, maxOutputTokens: 64_000 };
		}
		if (minor >= 5) {
			return { supportsThinking: false, supportsAdaptiveThinking: false, supportsThinkingEffort: false, supports1MContext: false, maxInputTokens: 200_000, maxOutputTokens: 8_192 };
		}
		return { supportsThinking: false, supportsAdaptiveThinking: false, supportsThinkingEffort: false, supports1MContext: false, maxInputTokens: 200_000, maxOutputTokens: 4_096 };
	}

	// Claude 1/2 or unknown major
	return { supportsThinking: false, supportsAdaptiveThinking: false, supportsThinkingEffort: false, supports1MContext: false, maxInputTokens: 100_000, maxOutputTokens: 4_096 };
}

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
			const version = parseClaudeVersion(modelPart);
			if (version) {
				return {
					supportsToolChoice: true,
					toolResultFormat: 'text',
					...deriveClaudeCapabilities(version.family, version.major, version.minor),
				};
			}
			// Unrecognised Claude model — tool choice but no thinking
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
