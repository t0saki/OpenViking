import { createTurnCapture, turnDedupKey } from "./turn-capture.mjs";
import { cleanZcodeText } from "./zcode-turns.mjs";

export const zcodeTurnDedupKey = turnDedupKey;

const capture = createTurnCapture({
  cleanText: cleanZcodeText,
  dedupKey: zcodeTurnDedupKey,
});

export const buildZcodeCapturePlan = capture.buildCapturePlan;
export const applyZcodeCaptureResult = capture.applyCaptureResult;
