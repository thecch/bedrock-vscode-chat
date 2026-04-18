import * as vscode from "vscode";
import type { AuthMethod } from "../types";

/**
 * Centralized configuration management for Bedrock extension.
 * All VS Code settings access should go through this service.
 */
export class ConfigurationService {
	private readonly configSection = 'languageModelChatProvider.bedrock';

	/**
	 * Get the AWS region from configuration
	 */
	getRegion(): string {
		const config = vscode.workspace.getConfiguration(this.configSection);
		return config.get<string>('region') ?? "us-east-1";
	}

	/**
	 * Get the authentication method from configuration
	 */
	getAuthMethod(): AuthMethod {
		const config = vscode.workspace.getConfiguration(this.configSection);
		return config.get<AuthMethod>('authMethod') ?? 'default';
	}

	/**
	 * Get API key from configuration (if using api-key auth)
	 */
	getApiKey(): string | undefined {
		const config = vscode.workspace.getConfiguration(this.configSection);
		return config.get<string>('apiKey');
	}

	/**
	 * Get AWS profile name from configuration (if using profile auth)
	 */
	getProfile(): string | undefined {
		const config = vscode.workspace.getConfiguration(this.configSection);
		return config.get<string>('profile');
	}

	/**
	 * Get AWS access key ID from configuration (if using access-keys auth)
	 */
	getAccessKeyId(): string | undefined {
		const config = vscode.workspace.getConfiguration(this.configSection);
		return config.get<string>('accessKeyId');
	}

	/**
	 * Get AWS secret access key from configuration (if using access-keys auth)
	 */
	getSecretAccessKey(): string | undefined {
		const config = vscode.workspace.getConfiguration(this.configSection);
		return config.get<string>('secretAccessKey');
	}

	/**
	 * Get AWS session token from configuration (if using access-keys auth with temp credentials)
	 */
	getSessionToken(): string | undefined {
		const config = vscode.workspace.getConfiguration(this.configSection);
		return config.get<string>('sessionToken');
	}

	/**
	 * Check if thinking (extended reasoning) is enabled
	 */
	isThinkingEnabled(): boolean {
		return this.getThinkingType() !== 'disabled';
	}

	/**
	 * Get thinking type: adaptive, enabled, or disabled
	 */
	getThinkingType(): 'adaptive' | 'enabled' | 'disabled' {
		const config = vscode.workspace.getConfiguration(this.configSection);
		const type = config.get<string>('thinkingType', 'adaptive');
		if (type === 'adaptive' || type === 'enabled' || type === 'disabled') {
			return type;
		}
		// Migration: treat old boolean thinkingEnabled as "enabled"
		const legacyEnabled = config.get<boolean>('thinkingEnabled', false);
		return legacyEnabled ? 'enabled' : 'adaptive';
	}

	/**
	 * Get thinking budget tokens (minimum 1024)
	 */
	getThinkingBudgetTokens(): number {
		const config = vscode.workspace.getConfiguration(this.configSection);
		const budget = config.get<number>('thinkingBudgetTokens', 10000);
		return Math.max(1024, budget);
	}

	/**
	 * Get thinking effort level
	 */
	getThinkingEffort(): 'max' | 'high' | 'medium' | 'low' {
		const config = vscode.workspace.getConfiguration(this.configSection);
		const effort = config.get<string>('thinkingEffort', 'max');
		if (effort === 'max' || effort === 'high' || effort === 'medium' || effort === 'low') {
			return effort;
		}
		return 'max';
	}

	/**
	 * Get the preferred default model substring
	 */
	getDefaultModel(): string {
		const config = vscode.workspace.getConfiguration(this.configSection);
		return config.get<string>('defaultModel', '').trim();
	}

	/**
	 * Get per-model overrides for a given model ID.
	 * Keys in modelOverrides are matched as substrings against the model ID.
	 */
	getModelOverride(modelId: string): { thinkingType?: 'adaptive' | 'enabled' | 'disabled'; maxInputTokens?: number; maxOutputTokens?: number } {
		const config = vscode.workspace.getConfiguration(this.configSection);
		const overrides = config.get<Record<string, unknown>>('modelOverrides', {});
		for (const [pattern, value] of Object.entries(overrides)) {
			if (modelId.includes(pattern) && value && typeof value === 'object') {
				return value as { thinkingType?: 'adaptive' | 'enabled' | 'disabled'; maxInputTokens?: number; maxOutputTokens?: number };
			}
		}
		return {};
	}

	/**
	 * Get thinking configuration for a specific model, applying per-model overrides.
	 * Returns undefined when disabled.
	 */
	getThinkingConfigForModel(modelId: string): { type: 'adaptive' | 'enabled'; budget_tokens?: number } | undefined {
		const override = this.getModelOverride(modelId);
		const thinkingType = override.thinkingType ?? this.getThinkingType();
		if (thinkingType === 'disabled') return undefined;
		if (thinkingType === 'adaptive') return { type: 'adaptive' };
		return { type: 'enabled', budget_tokens: this.getThinkingBudgetTokens() };
	}

	/**
	 * Get thinking configuration if thinking is not disabled.
	 * Returns undefined when disabled.
	 */
	getThinkingConfig(): { type: 'adaptive' | 'enabled'; budget_tokens?: number } | undefined {
		const thinkingType = this.getThinkingType();
		if (thinkingType === 'disabled') {
			return undefined;
		}

		if (thinkingType === 'adaptive') {
			return { type: 'adaptive' };
		}

		// type === 'enabled'
		return {
			type: 'enabled',
			budget_tokens: this.getThinkingBudgetTokens(),
		};
	}
}
