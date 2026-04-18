import * as vscode from "vscode";
import { ToolCallBufferManager } from "./tool-buffer";
import { logger } from "./logger";
import type { ThinkingBlock, StreamResult } from "./types";

interface StreamState {
	capturedThinkingBlock: ThinkingBlock | null;
	hasEmittedThinking: boolean;
	hasEmittedContent: boolean;
	stopReason: string | undefined;
}

export class StreamProcessor {
	private toolBuffer: ToolCallBufferManager;

	constructor() {
		this.toolBuffer = new ToolCallBufferManager();
	}

	async processStream(
		stream: AsyncIterable<any>,
		progress: vscode.Progress<vscode.LanguageModelResponsePart>,
		token: vscode.CancellationToken
	): Promise<StreamResult> {
		this.toolBuffer.reset();

		const state: StreamState = {
			capturedThinkingBlock: null,
			hasEmittedThinking: false,
			hasEmittedContent: false,
			stopReason: undefined,
		};

		try {
			for await (const event of stream) {
				if (token.isCancellationRequested) {
					break;
				}

				if (event.contentBlockStart) {
					const idx = event.contentBlockStart.contentBlockIndex ?? 0;
					const startData = event.contentBlockStart.start;
					const toolUse = startData?.toolUse;

					if (toolUse) {
						if (this.toolBuffer.shouldAddSpaceBeforeFirstTool()) {
							progress.report(new vscode.LanguageModelTextPart(' '));
						}
						this.toolBuffer.startToolCall(idx, toolUse.toolUseId || "", toolUse.name || "");
						this.toolBuffer.markFirstToolEmitted();
					}

					if (startData && "thinking" in startData) {
						const thinkingData = startData.thinking;
						const signature = typeof thinkingData === "object" && thinkingData && "signature" in thinkingData
							? String(thinkingData.signature) : undefined;
						state.capturedThinkingBlock = { signature, text: "" };
					}
				} else if (event.contentBlockDelta) {
					const idx = event.contentBlockDelta.contentBlockIndex ?? 0;
					const delta = event.contentBlockDelta.delta;

					if (delta?.text) {
						progress.report(new vscode.LanguageModelTextPart(delta.text));
						if (delta.text.length > 0) {
							this.toolBuffer.markHasText();
							state.hasEmittedContent = true;
						}
					}

					if (delta?.reasoningContent) {
						const reasoningText = typeof delta.reasoningContent.text === "string" ? delta.reasoningContent.text : undefined;
						const reasoningSignature = delta.reasoningContent.signature;

						if (reasoningText) {
							if (!state.capturedThinkingBlock) {
								state.capturedThinkingBlock = { text: "" };
							}
							state.capturedThinkingBlock.text += reasoningText;

							try {
								if (typeof (vscode as any).LanguageModelThinkingPart === "function") {
									progress.report(new (vscode as any).LanguageModelThinkingPart(reasoningText));
									state.hasEmittedThinking = true;
								}
							} catch (error) {
								const isTypeError = error instanceof TypeError || error instanceof ReferenceError
									|| String(error).includes("LanguageModelThinkingPart");
								if (!isTypeError) {
									throw error;
								}
							}
						}

						if (typeof reasoningSignature === "string") {
							if (!state.capturedThinkingBlock) {
								state.capturedThinkingBlock = { text: "" };
							}
							state.capturedThinkingBlock.signature = (state.capturedThinkingBlock.signature ?? "") + reasoningSignature;
						}
					}

					if (delta?.toolUse?.input) {
						this.toolBuffer.appendArgs(idx, delta.toolUse.input);
						await this.toolBuffer.tryEmit(idx, progress);
					}
				} else if (event.contentBlockStop) {
					const idx = event.contentBlockStop.contentBlockIndex ?? 0;
					await this.toolBuffer.tryEmit(idx, progress, true);
				} else if (event.messageStop) {
					state.stopReason = event.messageStop.stopReason;
					await this.toolBuffer.emitAll(progress);
				}
			}

			// Fallback: thinking captured but LanguageModelThinkingPart unavailable
			if (!state.hasEmittedContent && !state.hasEmittedThinking
				&& state.capturedThinkingBlock?.text && !token.isCancellationRequested) {
				progress.report(new vscode.LanguageModelTextPart(
					"*(The model produced only internal reasoning, but thinking display is not supported in this environment. Please try again or rephrase your request.)*"
				));
			}

			// Fallback: completely empty response
			if (!state.hasEmittedContent && !state.hasEmittedThinking
				&& !state.capturedThinkingBlock?.text && !token.isCancellationRequested
				&& state.stopReason === "end_turn") {
				progress.report(new vscode.LanguageModelTextPart(
					"*(The model returned an empty response. Please try again or rephrase your request.)*"
				));
			}
		} finally {
			this.toolBuffer.reset();
		}

		return { thinkingBlock: state.capturedThinkingBlock };
	}
}
