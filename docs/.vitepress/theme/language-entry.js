/** Used unchanged by the blocking head script and the entry contract tests. */
export function docsLanguageEntry(href, base, preference) {
  const url = new URL(href);
  const root = '/' + base.split('/').filter(Boolean).join('/');
  const prefix = root === '/' ? '/' : root + '/';
  if (url.pathname !== root && !url.pathname.startsWith(prefix)) return null;
  const relative = url.pathname === root ? '' : url.pathname.slice(prefix.length);
  const match = /^(?:(en|zh)\/?|(?:en|zh)\/index\.html|index\.html)?$/.exec(relative);
  if (!match) return null;
  const pathLocale = /^(en|zh)(?:\/|$)/.exec(relative)?.[1];
  const locale = preference.resolve(pathLocale || preference.parse(url.searchParams.get('lang')) || null, true);
  url.pathname = prefix + locale + '/getting-started/01-introduction';
  return url.href;
}
