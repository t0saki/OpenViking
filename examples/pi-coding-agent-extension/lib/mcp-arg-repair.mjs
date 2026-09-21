/**
 * Schema-driven repair of model-emitted MCP tool arguments.
 *
 * Governing principle: pi validates tool arguments locally, before it dispatches
 * the call, whereas every other harness hands the model's arguments straight to
 * the server. Repair exists only to erase that extra local strictness. It must
 * never be stricter than the server, and it must never be more permissive than
 * the server — a call this module lets through has to mean on the server exactly
 * what it would have meant coming from a harness with no local validation.
 *
 * Three rules, each recursive through `items` and `properties`:
 *
 *   R1  drop explicitly-null optional properties
 *   R2  wrap a non-array value where the schema declares an array
 *   R3  JSON.parse a string that the schema wants as an array or object
 *
 * Deliberately NOT done, because on every other harness the server rejects or
 * ignores these just the same and the model retries once it sees the error:
 * enum case or synonym fixing, numeric clamping, `maxItems` truncation,
 * camelCase key rewriting, unwrapping `{input:{...}}`. Unknown keys pass
 * through unchanged.
 *
 * Pure module: no imports, no I/O, no mutation of the inputs. Wired into pi
 * through `ToolDefinition.prepareArguments`, which runs before validation.
 */

function isPlainObject(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Only a single declared type drives a rule. OpenViking's schemas are made
 * portable server-side, so they carry no `$ref`, no `anyOf` and no type arrays;
 * if one ever shows up the value is left alone, since a union already accepts
 * more than one shape and repairing towards one of them could change meaning.
 */
function declaredType(schema) {
  return typeof schema.type === "string" ? schema.type : "";
}

function repairValue(value, schema) {
  if (value === null || value === undefined) return value;
  const type = declaredType(schema);
  let next = value;

  // R3: a string where the schema wants an array or an object. This is exactly
  // what FastMCP's pre_parse_json does for every other harness, so parsing here
  // only reproduces what the server would have done with the same string. A
  // parse failure leaves the value untouched — the server would keep it too.
  if ((type === "array" || type === "object") && typeof next === "string") {
    const trimmed = next.trim();
    if (trimmed.startsWith("[") || trimmed.startsWith("{")) {
      try {
        const parsed = JSON.parse(trimmed);
        if (parsed !== null && typeof parsed === "object") next = parsed;
      } catch {
        // Not JSON after all. Leave it; R2 may still wrap it, which is what the
        // server does with an unparseable string for a `str | list[str]` field.
      }
    }
  }

  // R2: the schema says array, the value is not one. The server's signatures
  // already accept `str | list[str]` (and a single object for remember's
  // messages), while pi rejects the scalar outright through 0.86.1.
  if (type === "array" && !Array.isArray(next)) next = [next];

  if (Array.isArray(next) && isPlainObject(schema.items)) {
    return next.map((item) => repairValue(item, schema.items));
  }
  if (type === "object" && isPlainObject(next)) return repairObject(next, schema);
  return next;
}

function repairObject(bag, schema) {
  const properties = isPlainObject(schema.properties) ? schema.properties : {};
  const required = new Set(Array.isArray(schema.required) ? schema.required : []);
  const out = {};

  for (const [key, value] of Object.entries(bag)) {
    const propSchema = properties[key];
    if (!isPlainObject(propSchema)) {
      // Unknown key: pass it through untouched, exactly as it would reach the
      // server from any other harness.
      out[key] = value;
      continue;
    }
    // R1: the server treats None as absent, but pi 0.80-0.84 silently coerced
    // an explicit null to 0 or "" (`read{limit:null}` arrived as `limit:0` and
    // read zero lines). Fixed upstream in pi-ai 0.85; the rule stays because it
    // is harmless. A required null is left in place so the model still gets a
    // real validation error instead of a call that quietly means something else.
    if (value === null && !required.has(key)) continue;
    out[key] = repairValue(value, propSchema);
  }

  return out;
}

/**
 * Repair one argument bag against a tool's `inputSchema`. Returns a new value;
 * `args` and `inputSchema` are never mutated. A malformed or missing schema
 * means "hand the arguments back unchanged", but a non-object argument bag
 * still becomes `{}` so the tool sees the shape pi promises it.
 */
export function repairArguments(args, inputSchema) {
  if (!isPlainObject(args)) return {};
  if (!isPlainObject(inputSchema)) return { ...args };
  return repairObject(args, inputSchema);
}

/** Curried form; `tools.ts` binds one per tool to `prepareArguments`. */
export function makeArgRepair(inputSchema) {
  return (args) => repairArguments(args, inputSchema);
}
