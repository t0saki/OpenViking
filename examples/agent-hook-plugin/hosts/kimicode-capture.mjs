import { createTurnCapture, turnDedupKey } from "./turn-capture.mjs";
import { cleanKimicodeText } from "./kimicode-turns.mjs";

export const kimicodeTurnDedupKey = turnDedupKey;

const capture = createTurnCapture({
  cleanText: cleanKimicodeText,
  dedupKey: kimicodeTurnDedupKey,
});

export const buildKimicodeCapturePlan = capture.buildCapturePlan;
export const applyKimicodeCaptureResult = capture.applyCaptureResult;
