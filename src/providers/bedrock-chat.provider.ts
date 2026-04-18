import * as vscode from "vscode";
import type {
	CancellationToken,
	LanguageModelChatInformation,
	LanguageModelChatMessage,
	LanguageModelChatProvider,
	LanguageModelChatRequestHandleOptions,
	LanguageModelResponsePart,
	Progress,
} from "vscode";
import { ModelService } from "../services/model.service";
import { AuthenticationService } from "../services/authentication.service";
import { ConfigurationService } from "../services/configuration.service";
import { ChatRequestHandler } from "./chat-request.handler";
import { TokenEstimator } from "./token.estimator";

/**
 * Main Bedrock chat provider that coordinates all operations.
 * Delegates to specialized handlers for specific functionality.
 */
export class BedrockChatProvider implements LanguageModelChatProvider {
	private modelService: ModelService;
	private chatRequestHandler: ChatRequestHandler;
	private tokenEstimator: TokenEstimator;

	private readonly _onDidChangeLanguageModelInformation = new vscode.EventEmitter<void>();
	readonly onDidChangeLanguageModelChatInformation = this._onDidChangeLanguageModelInformation.event;

	constructor(
		private readonly configService: ConfigurationService,
		private readonly authService: AuthenticationService
	) {
		this.modelService = new ModelService(authService, configService);
		this.chatRequestHandler = new ChatRequestHandler(this.modelService, authService, configService);
		this.tokenEstimator = new TokenEstimator();
	}

	/**
	 * Signal VS Code to re-fetch the model list
	 */
	notifyModelInformationChanged(): void {
		this._onDidChangeLanguageModelInformation.fire();
	}

	/**
	 * Dispose resources
	 */
	dispose(): void {
		this._onDidChangeLanguageModelInformation.dispose();
	}

	/**
	 * Handle configuration changes
	 */
	handleConfigurationChange(): void {
		this.modelService.handleConfigurationChange();
		this.chatRequestHandler.handleConfigurationChange();
		this.notifyModelInformationChanged();
	}

	/**
	 * Prepare language model chat information (called by VS Code to get the model list)
	 */
	async prepareLanguageModelChatInformation(
		options: { silent: boolean },
		_token: CancellationToken
	): Promise<LanguageModelChatInformation[]> {
		return await this.modelService.getLanguageModelChatInformation(options.silent ?? false);
	}

	/**
	 * Provide language model chat information (required by @types/vscode interface)
	 */
	async provideLanguageModelChatInformation(
		options: { silent: boolean },
		_token: CancellationToken
	): Promise<LanguageModelChatInformation[]> {
		return await this.modelService.getLanguageModelChatInformation(options.silent ?? false);
	}

	/**
	 * Process a chat request and stream the response
	 */
	async provideLanguageModelChatResponse(
		model: LanguageModelChatInformation,
		messages: readonly LanguageModelChatMessage[],
		options: LanguageModelChatRequestHandleOptions,
		progress: Progress<LanguageModelResponsePart>,
		token: CancellationToken
	): Promise<void> {
		await this.chatRequestHandler.handleChatRequest(model, messages, options, progress, token);
	}

	/**
	 * Estimate token count for text or message
	 */
	async provideTokenCount(
		model: LanguageModelChatInformation,
		text: string | LanguageModelChatMessage,
		_token: CancellationToken
	): Promise<number> {
		return this.tokenEstimator.estimateTokens(model, text);
	}
}
