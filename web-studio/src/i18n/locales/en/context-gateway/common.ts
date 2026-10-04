const common = {
  title: 'Context Gateway',
  description:
    'Give any API-key model client OpenViking memory. Point the client at the gateway, and it adds relevant memory to each new message and saves conversations back to OpenViking.',
  tabs: {
    label: 'Context Gateway sections',
    overview: 'Overview',
    upstreams: 'Upstreams',
    profiles: 'Profiles',
    keys: 'Keys',
    requests: 'Requests',
    connect: 'Connect',
  },
  address: {
    label: 'Gateway address',
    copy: 'Copy gateway address',
  },
  actions: {
    cancel: 'Cancel',
    copy: 'Copy',
    delete: 'Delete',
    duplicate: 'Duplicate',
    edit: 'Edit',
    more: 'More actions',
    refresh: 'Refresh',
    remove: 'Remove',
    retry: 'Retry',
    revoke: 'Revoke',
    save: 'Save',
    viewAll: 'View all',
  },
  states: {
    loading: 'Loading…',
    saving: 'Saving…',
    updatedAt: 'Updated {{time}}',
    any: 'Any',
    on: 'On',
    off: 'Off',
  },
  copy: {
    done: 'Copied to clipboard',
    failed: "Couldn't copy. Select the text and copy it manually.",
  },
  access: {
    title: 'Account administrator access required',
    description:
      'Managing the Context Gateway needs an account administrator or root key for this account. Add one in Connection settings.',
    action: 'Open connection settings',
  },
  unavailable: {
    notEnabled: {
      title: 'The Context Gateway is turned off',
      description:
        'This OpenViking server has the Context Gateway turned off. Turn it on in ov.conf, restart OpenViking, then start the gateway.',
    },
    tokenMissing: {
      title: 'The management token is missing',
      description:
        'OpenViking needs the same management token as the gateway to manage it. Set this environment variable (at least 32 characters) for both OpenViking and the gateway, then restart them.',
    },
    unreachable: {
      title: "OpenViking can't reach the gateway",
      description:
        'The gateway is not running, or context_gateway.url in ov.conf points to the wrong address. Start the gateway and check that OpenViking can reach that address.',
    },
    failed: {
      title: "Couldn't load the Context Gateway",
    },
    fixLabel: 'What to change',
    terminal: 'Terminal',
    environment: 'Environment variable',
    docs: 'How to deploy',
  },
  errors: {
    reasons: {
      not_enabled: 'The Context Gateway is turned off on this server.',
      token_missing:
        "OpenViking can't manage the gateway because the management token isn't set.",
      unreachable: "OpenViking can't reach the Context Gateway.",
      conflict: 'This change conflicts with existing settings.',
      invalid:
        'The gateway rejected these settings. Check the values and try again.',
      forbidden: "You don't have permission to do this.",
      not_found: 'This item no longer exists. Refresh and try again.',
      unauthorized: 'Your key was rejected. Check Connection settings.',
      other: 'Something went wrong. Try again.',
    },
    inUse: 'Keys still use this. Revoke those keys first.',
    invalidSettings:
      'The gateway rejected these settings. Check the values and limits.',
    invalidKeyRequest:
      'The gateway rejected this key. Check the fields and try again.',
    unknownProfile: 'The selected context profile no longer exists.',
    unknownUpstream: 'One of the selected upstreams no longer exists.',
    keyNotFound:
      'The gateway key this conversation used has been revoked, so it can no longer be resynced.',
    sessionNotFound: 'The gateway no longer has this conversation.',
    upstreamKeyMissing:
      'There is no stored API key to test with. Clients of this upstream send their own key, or the key has not been set.',
    subscriptionKey:
      "Claude subscription logins aren't supported. Use a model API key.",
  },
  validation: {
    required: 'Required',
    number: 'Enter a number',
    integer: 'Enter a whole number',
    min: 'Must be at least {{min}}',
    max: 'Must be at most {{max}}',
    range: 'Must be between {{min}} and {{max}}',
    rangeExclusive: 'Must be more than {{min}} and at most {{max}}',
    greaterThan: 'Must be more than {{min}}',
    selectOneSource: 'Choose at least one source',
    unknownCategory: 'Unknown category “{{name}}”',
    quotasAllZero: 'Set at least one category above 0, or turn the limit off',
    selectOneTool: 'Choose at least one tool',
    baseUrl:
      'Enter an http:// or https:// URL without credentials, query or fragment',
    subscriptionKey:
      "Claude subscription tokens (sk-ant-oat…) aren't supported. Use a model API key.",
    apiKeyRequired:
      'Enter the API key, or let each client send its own key instead',
    headerInvalid: 'Header “{{name}}” has an invalid name or value',
    headerReserved: "“{{name}}” is set by the gateway and can't be changed",
    headerValue: 'Enter a value for “{{name}}”',
    aliasTarget: 'Enter the upstream model for “{{name}}”',
    contextWindow: 'The window for “{{name}}” must be at least {{min}} tokens',
  },
  units: {
    bytes: 'bytes',
    characters: 'characters',
    entries: 'entries',
    messages: 'messages',
    rounds: 'rounds',
    seconds: 'seconds',
    tokens: 'tokens',
    turns: 'turns',
  },
  field: {
    default: 'Default: {{value}}',
    notSet: 'Not set',
    advanced: 'Advanced settings',
    storedSecret: 'Stored — leave blank to keep',
  },
  keyValue: {
    add: 'Add',
    remove: 'Remove',
    duplicate: 'Listed twice; only the last one is kept',
  },
  tags: {
    remove: 'Remove {{value}}',
    suggestions: 'Suggestions',
  },
}

export default common
