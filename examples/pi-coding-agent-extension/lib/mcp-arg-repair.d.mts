export type JsonSchema = Record<string, any>;

export function repairArguments(args: unknown, inputSchema?: JsonSchema | null): Record<string, any>;
export function makeArgRepair(inputSchema?: JsonSchema | null): (args: unknown) => Record<string, any>;
