export function deriveWorkspacePeerId(cwd: unknown): string;
export function resolveEffectivePeerId(input?: {
  cfg?: { peerId?: string; workspacePeer?: boolean; peerSource?: unknown };
  cwd?: string;
  onWarn?: ((message: string) => void) | null;
}): {
  peerId: string;
  source: "explicit" | "workspace" | "none";
  origin: string;
  /** The pre-git id, when it differs from `peerId`; otherwise empty. */
  legacyPeerId: string;
};
