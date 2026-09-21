# Harness 插件统一更名为 OpenViking

插件统一命名为 **OpenViking**，与其连接的上下文数据库保持一致。Claude Code 和 Codex 使用 `openviking@openviking`；共享技能更名为 `openviking`，诊断技能更名为 `ov-plugin-doctor`。DSH 的 npm 包名改为 `@openviking/dsh-plugin`，OpenCode 的运行日志改为 `openviking.log`。

对应版本为 Claude Code 0.6.0、Codex 0.10.0、hook 宿主插件 0.4.0、DSH 0.5.0、OpenCode 0.4.0 和 Agent Plugins 0.2.0。pi 因共享运行时迁移而更新 patch 版本，其身份与工具名保持不变。

## 升级已有安装

为你使用的宿主重新运行[统一安装器](../agent-integrations/01-overview.md)。安装器先移除旧插件 ID，再安装新插件，迁移受管 hooks 和 MCP 配置，并移除 Cursor 的旧规则和技能。已有 OpenViking statusline 会自动更新；自定义 statusline 只有在你明确选择替换时才会被修改。

| 宿主 | 需要执行的操作 |
|---|---|
| 从仓库 marketplace 安装的 Claude Code | 更新 marketplace。Claude Code 2.1.193+ 会通过 marketplace 顶层的 `renames` 映射更新 `enabledPlugins` 和 `pluginConfigs`；远程 `git-subdir` 来源仍可能需要执行一次 `/plugin install openviking@openviking`，填充新的插件缓存。 |
| 通过安装器安装的 Claude Code | 重新运行安装器，更新本机生成的 marketplace manifest。 |
| 所有 Claude Code 安装 | 手动更新包含 `mcp__plugin_openviking-memory_openviking__*` 或 `plugin:openviking-memory:openviking` 的权限规则和 hook matcher；marketplace 改名不会重写这些配置。 |
| 低于 2.1.193 的 Claude Code，或使用 managed settings | 升级宿主或重装插件；管理员需要自行修改 managed settings。 |
| Codex / TraeCode CLI 2.0 | 重新运行安装器，重新信任 6 个 hooks，并在首次使用时审批新 server 名称下的 MCP 工具。 |
| Cursor / TRAE / ZCode | 重新运行安装器，替换旧的受管 hooks、规则、技能和配置项。 |
| DSH | 重新运行安装器；它会从所选 profile 中移除 `@openviking/dsh-memory-plugin`，再安装 `@openviking/dsh-plugin`。 |
| Hermes | 新 schema 暴露 6 个 `openviking_*` 工具：search、read、browse、remember、forget 和 add_resource。原来的 `viking_*` 别名仍可调用。 |

原生 Codex 安装器会先备份 `config.toml`，再移除旧插件启用表和 hook 信任表。`mcp_servers.openviking-memory.tools.*` 下的逐工具审批记录会保留供参考，但不能为新 server 身份授予权限。对于兼容 Codex 格式的客户端，安装器通过宿主 CLI 管理插件；弃用的 TRAE CLI 1.0 集成只会在新插件成功启用后移除。

旧共享运行时、marketplace 和 DSH tarball 目录，只有在已安装适配器、已知宿主注册信息或 DSH profile 不再引用它们后才会清理，因此可以分别升级各个宿主和 profile。安装器支持重复运行，不会重复添加受管 hooks。

升级后运行 `ov-plugin-doctor`，检查旧插件 ID、MCP 配置、权限规则，以及并存的 Cursor 规则和共享目录。保留下来的文件可能仍被其他宿主使用，请先按诊断提示完成升级，再决定是否删除。

## 兼容性与行为变化

Codex 的 MCP 工具前缀从 `mcp__openviking_memory__` 改为 `mcp__openviking__`。服务端已经识别后者，因此这些调用会开始计入注入用量统计和 Experience lineage。Claude Code 的派生前缀从 `mcp__plugin_openviking-memory_openviking__` 改为 `mcp__plugin_openviking_openviking__`。

捕获过滤器同时识别 `[openviking-memory]` 和 `[openviking]`。DSH 同时识别历史会话中的两种 `source.plugin` 值，Hermes 同时识别两代召回工具名，避免将已召回的上下文再次捕获入库。

`OPENVIKING_MEMORY_ENABLED`、`openviking_memory_*` 指标、`ov-experience-memory`、`ov-memory-troubleshoot` 和 `OpenViking memory digest:` 协议头保持不变。OpenClaw 与 pi 的身份和工具名、OpenWebUI 的 `ov_*` 工具、历史 changelog 和冻结的 pi 快照不在此次迁移范围内。

旧 GitHub 共享安装器地址会转发到 `examples/plugin-shared/install.sh`，计划在两个发布周期后删除转发脚本。TOS 在新旧 key 下发布相同产物；`plugins/memory-plugins.git` 长期保留，因为已有 Codex 安装会把它存为 marketplace 来源。指向单个已搬迁插件文件的旧 GitHub raw 链接不会重定向，需要更新为新目录。

## 发布配合

合并前需要首次发布 `@openviking/dsh-plugin`，并为新包配置 npm trusted publishing。合并后应及时发布 TOS 产物，或以 `update_latest=true` 手动触发 `release-tos.yml`；官网 agent 卡片会随 main 更新，需要新安装器 URL 已经可用。发布后将旧 DSH npm 包标记为 deprecated，不要 unpublish。

OpenClaw 尚未统一的工具和技能名、pi 的身份与工具名、OpenWebUI 工具前缀、digest 协议头，以及 workflow/job 展示名均留待后续处理。workflow 展示名需要与分支保护规则协调修改。共享目录搬迁还会命中现有 workflow 路径过滤，触发一次 OpenClaw 开发版发布。
