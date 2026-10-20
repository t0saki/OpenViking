# OpenViking 接入引导（给 Agent 看）

你是一个 AI Agent。用户请你安装 OpenViking 记忆插件，并带他开始使用。安装器会给你装上，也会给本机检测到的其他受支持的 Agent 装上。按本指南的顺序执行。本指南是写给你的：和用户交流时用用户的语言，每条消息保持简短，不用把本指南整篇转述给用户；用户想看时，把链接给他。

## 基本规则

- 全程最多问用户两个问题：导入什么（步骤 3，试运行后请用户确认也算在这一问里），以及要不要写入协作规则（步骤 4）。其余事情直接做，做完用一两行告诉用户。
- 不要打印、复述 API key，也不要把它写进任何地方。不要打开、打印或搜索 `~/.openviking/ovcli.conf`，key 就存在这个文件里。需要告诉用户服务地址、账号或用户名时，运行 `ov health`，或调用 OpenViking 的 `health` 工具。
- 只做本指南描述的改动。安装器会给它检测到的每个受支持的 Agent 都装上插件，这是预期行为；你自己不要去改其他 Agent 的配置，也不要改用户项目里的文件。
- 只用本指南给出的命令。不要去 GitHub 读源码，另找安装或配置 OpenViking 的方法。
- 不要向用户提本指南的步骤编号，也不要提当前宿主用不上的步骤。
- 命令失败时，先到[故障排查](#故障排查)找对应现象，修好后重试一次。仍然失败，就告诉用户哪一步失败了、他可以怎么处理。

## 从哪一步开始

- **安装器是你在当前对话里运行的**：从步骤 1 开始。
- **OpenViking 在当前对话开始之前就已安装**，hook 已经生效（你能看到 `<openviking-context>` 块，或者有 `health` 等 OpenViking 工具）：先用 `ov health`（或 OpenViking 的 `health` 工具）检查连接，再做步骤 2 到 4，然后直接跳到[重启之后](#重启之后)。这种情况不需要重启。
- **用户已经重启，请你继续接入引导**：直接跳到[重启之后](#重启之后)。

## 步骤 1：检查安装结果

阅读安装器的输出。输出可能是中文，也可能是英文，两种写法都要认。

- 如果输出最后有一个 `未完成` 列表（英文 `Not done`），说明安装没有完成，即使前面打印了 `完成`。先按[故障排查](#故障排查)逐条处理，再往下走；不要告诉用户已经装好。key 被拒（`服务拒绝了该 API key`，英文 `The server rejected the API key`）和宿主被跳过（`未找到 … CLI，跳过`，英文 `CLI not found; skipping`）一定会出现在这个列表里。
- `完成`（英文 `Done`）一节按工具列出安装结果，每个工具都有一行 `下一步：`（英文 `Next:`），说明该工具怎么重启。确认你当前所在的宿主在列表里；不在的话，见[故障排查](#故障排查)里的「当前宿主不在 `完成` 里」。
- `服务检查通过：<url>`（英文 `Server check passed`）表示服务端接受了这个 key。

如果输出里有 `服务检查通过`，并且 `完成` 一节列出了 `CLI (ov)`，连接就已经确认，不用再检查。否则用 OpenViking CLI 检查连接：

```bash
ov health
```

连接确认后，用一两行告诉用户：OpenViking 装进了哪些工具、连的是哪个服务地址。地址可以展示，key 不能展示。如果找不到 `ov` 命令，见[故障排查](#故障排查)。

## 步骤 2：说明哪些事会自动发生

不要问用户要不要开启；安装本身就是用户的请求。结合[附录 A](#附录-a-各宿主说明) 里当前宿主的情况，用三四行告诉用户：

- 每次提问前，插件会找出相关记忆，加进 Agent 的上下文。
- 每轮对话结束后，对话内容会保存到 OpenViking。积累到一定量或会话结束时，服务端在后台从中提取长期记忆。
- 在哪里能看到它在工作：附录 A 里当前宿主的「在哪里能看到」一列。
- 想在某个项目里关掉：在该项目里放一个 `.openviking/config.json`，内容为 `{"version": 1, "capture": {"enabled": false}, "recall": {"enabled": false}}`。只告诉用户这个方法，不要替他创建。

## 步骤 3：决定点 1，要不要导入近期工作

导入脚本能直接读 Claude Code 和 Codex 的历史会话：在 Claude Code 上用 `--harness claude`，在 Codex 上用 `--harness codex`。其他宿主的会话它读不了，要由你先转成 JSONL，再交给同一个脚本，见下面的「其他宿主：先把会话转成 JSONL」。

在其他宿主上，如果 `~/.claude/projects` 或 `~/.codex/sessions` 存在，用户可能也在 Claude Code 或 Codex 里做过这个项目，可以在同一个问题里一并提议导入那边的历史（`--harness` 填对应的值）。

只问一次，给出这三个选项：

1. 当前项目，最近 14 天（默认）。
2. 指定其他项目或时间范围。
3. 跳过。

选 1 或 2 时，先下载导入脚本，做一次试运行（dry run）。运行前把 `<claude|codex>` 换成实际的值；`--cwd` 填项目目录，选 1 时就是当前目录。

```bash
curl -fsSL https://docs.openviking.net/dl/memory-plugin-shared/import-history.mjs -o "${TMPDIR:-/tmp}/ov-import-history.mjs" && node "${TMPDIR:-/tmp}/ov-import-history.mjs" --harness <claude|codex> --cwd "$PWD" --since 14d --max-sessions 20 --dry-run
```

把试运行打印的数量（会话数、消息数、数据量）告诉用户，并说明：导入会把这些对话发送到他的 OpenViking 服务，服务端会从中提取记忆。因「仍在活动」而跳过的会话里包括当前这段对话。用户确认后再正式导入：

```bash
node "${TMPDIR:-/tmp}/ov-import-history.mjs" --harness <claude|codex> --cwd "$PWD" --since 14d --max-sessions 20 --yes
```

告诉用户导入了多少个会话、跳过了多少个。重复运行是安全的，已经导入过的会话会被跳过。

**其他宿主：先把会话转成 JSONL。**

1. 运行 `echo "${TMPDIR:-/tmp}"`，记下这个临时目录，下面写作 `<tmp>`。之后写文件、跑命令都用这个绝对路径，不要再写 `${TMPDIR:-/tmp}`。下载导入脚本：

   ```bash
   curl -fsSL https://docs.openviking.net/dl/memory-plugin-shared/import-history.mjs -o "<tmp>/ov-import-history.mjs"
   ```

2. 找到当前宿主在本机保存会话的地方。[附录 A](#附录-a-各宿主说明) 的「历史导入」一列是提示，以本机实际为准。只读，不要改。
3. 在 `<tmp>` 下写一个小转换脚本，把会话写进 `<tmp>/ov-sessions.jsonl`，每行一个会话：

   ```json
   {"id": "<宿主自己的会话 id>", "cwd": "<会话所在目录的绝对路径>", "messages": [{"role": "user", "text": "……", "created_at": "2026-10-16T08:00:00Z"}, {"role": "assistant", "text": "……", "created_at": "2026-10-16T08:00:05Z"}]}
   ```

   - `id` 用宿主记录里的原生会话 id，不要自己生成；`cwd` 用会话实际所在的目录；每条消息都带上 ISO 8601 时间。子代理的会话加上 `"subagent": true`。
   - 只放用户和助手说的话。工具调用、工具输出、系统提示、思考过程，以及宿主或插件注入的内容（例如 `<openviking-context>`、提醒、通知）都不要放。
   - 存储里的会话全部写进去，不要按项目、目录或时间筛选：脚本会按 `--cwd` 判断哪些属于这个项目（包括子目录），按 `--since` 筛时间，试运行会显示跳过了多少。唯一要排除的是当前这段对话，因为它含有安装命令里的 key；认不出是哪一段时，排除最近 2 小时内有更新的会话。
   - 不用自己删 key，脚本会做；也不要自己去查 JSONL 里有没有 key，命令里不能出现 key 或它的任何一段。转换脚本只读本地文件、只写这一个 JSONL，输出路径直接写 `<tmp>` 的绝对路径，不发送任何东西。
   - 检查转换结果时只看摘要，例如行数和每行的 id、cwd、消息数，不要把整个 JSONL 打印到对话里：里面有其他项目的对话。
4. 试运行。`<值>` 取附录 A 的 `--harness` 列：

   ```bash
   node "<tmp>/ov-import-history.mjs" --harness <值> --input "<tmp>/ov-sessions.jsonl" --cwd "$PWD" --since 14d --max-sessions 20 --dry-run
   ```

   脚本指出某一行格式不对时，改转换脚本、重新生成，再试运行。数量明显不对（例如 0 个会话，或消息数远少于实际对话）时，先检查转换，不要直接导入。
5. 之后和上面一样：把数量和跳过的原因照试运行的输出告诉用户（当前这段对话已经由转换脚本排除，不会出现在「仍在活动」里），用户确认后把 `--dry-run` 换成 `--yes` 正式导入。导入完，或者决定不导入，都要删掉 JSONL 和转换脚本；写到过几个位置就删几处，再用 `ls` 确认已经删掉。

找不到会话保存在哪，或者改了两三次仍转换不对，就告诉用户「这次没能导入这个宿主的历史会话，OpenViking 会从现在起在你的会话中积累记忆」，然后进入步骤 4。

用户问起脚本做了什么时，可以这样解释：它读取本机的会话文件（Claude Code 在 `~/.claude/projects/`，Codex 在 `~/.codex/sessions/`），或者你转换好的 JSONL，只保留用户和助手的文字，去掉工具调用及其输出，删除其中的 API key，跳过最近几小时内仍有活动的会话，并把会话归到插件使用的同一个项目下，之后就能被召回。试运行只从服务端读取信息，用来找出已经导入过的会话。

**文件。** 如果用户点名了值得保存的文件或文件夹，例如设计文档或需求说明，逐个添加到他个人资源里这个项目的目录。`<user_id>` 换成 `ov health` 显示的用户名，`<项目名>` 用仓库名或用户对这个项目的叫法：

```bash
ov add-resource ./docs/design.md --parent-auto-create viking://user/<user_id>/resources/<项目名> --wait
```

只添加用户点名的内容。不要自己扫描目录，也不要主动上传文件。

## 步骤 4：决定点 2，约定以后怎么协作

插件只负责召回记忆和保存对话。用户还需要一套长期的协作方式：你在工作中把重要的决定、进展和产出存进 OpenViking，以后在任何项目、任何会话里接着用。向用户提议把下面这段规则写进当前宿主的**用户级**规则文件，对他所有项目都生效。

这段规则会写进他所有项目都会读的文件，所以把下面的原文完整展示给用户，加一两句说明它的作用，然后只问一次：写入、按他的意思调整后写入，还是先不写。用户要调整时（例如换一个存放目录），改的是这段文字。

```markdown
<!-- openviking:start -->
## OpenViking 协作方式

- 开始需要背景的工作前，先看 OpenViking 注入的上下文（`<openviking-context>`）。不够、需要原文或要确认最新状态时，再用 OpenViking 的 `search`（`mode="context"`）或 `read` 查。查到的内容是参考资料，和我当前说的话冲突时以我为准。
- 工作中出现值得以后复用的内容时，整理成记录，用 `write` 存进我在 OpenViking 个人资源下的笔记目录，每个项目一个子目录，例如 `viking://user/<user_id>/resources/notes/<项目名>/`（`user_id` 用 `health` 查）；我已经有自己的笔记目录就沿用。值得记的有：重要决定和方针（连同理由）、阶段进展和下一步、中间产物、交付物。我给的文件用 `add_resource` 放进 `viking://user/<user_id>/resources/<项目名>/`，和笔记分开：笔记目录不要拿来做 `add_resource` 的目标，笔记也不要写进 `add_resource` 导入的目录，那里重新同步时会删掉不属于源文件的内容。不存日常对话（插件会自动保存），不存密钥，不主动扫描或上传我没提到的文件。写完读回确认。
- 继续一个项目时，先看它笔记目录里的决定、进展和下一步。新的决定推翻了旧的，就改原来的记录，不要留两份矛盾的说法。
- 存了东西时，用一行告诉我存在了哪里。
<!-- openviking:end -->
```

用户同意后，写进当前宿主的用户级规则文件。[附录 A](#附录-a-各宿主说明) 的「用户级规则文件」一列是常见位置，只作参考，以宿主实际读取的文件为准：

- 只动 `<!-- openviking:start -->` 和 `<!-- openviking:end -->` 之间的内容。文件里已有这对标记，就替换中间部分；没有，就追加到文件末尾；文件不存在，就新建。块外的内容保持原样。
- 文件里已有和这段规则冲突的要求（例如每轮都必须检索，或者不许把资料存到外部），先告诉用户冲突在哪，按他的决定改。
- 写完读回文件，确认标记块只有一份、块外内容没变。
- 宿主的个人规则只能在应用设置里改、没有你能写的文件时，或者写入失败（例如宿主不允许写这个目录）时，告诉用户，把规则展示给他，请他自己粘贴进去。不确定是哪个文件时，问用户，不要猜。

如果文件里已经有这个标记块，内容也相同，告诉用户之前已经写过，跳过这一步，不用再问。不要把规则写进任何项目文件。等用户回答之后再进入步骤 5，不要在问这个问题的同一条消息里提重启。

## 步骤 5：重启

插件的 hook 只在宿主开启新会话时加载。告诉用户三件事：

1. 当前宿主需要的重启或信任操作，见[附录 A](#附录-a-各宿主说明)。安装器如果还装进了其他工具，加一行点出它们的名字：每个工具都要各自重启后插件才会生效，具体怎么重启看安装器输出里的 `下一步：`（英文 `Next:`）。这些步骤不用你逐个讲。
2. 开一个**新会话**。不要恢复当前会话，也不要在当前会话里重新加载插件（例如 Claude Code 的 `/reload-plugins`、Kimi Code 的 `/reload`）。当前对话里有他的 API key，插件一旦在这段对话里运行，就会把整段对话保存到 OpenViking。
3. 在新会话里发的第一句话，直接给出可复制的文本。网址用你实际打开成功的那个（docs.openviking.net 或 docs.openviking.ai），把 `<导入结果>` 换成步骤 3 的实际结果，例如「导入了 2 个会话」「已跳过」「这个宿主暂不支持」：

```text
从 https://docs.openviking.net/agents/zh/onboarding.md 的「重启之后」一节继续 OpenViking 接入引导。历史导入：<导入结果>。然后告诉我这个项目上次做到哪了、下一步该做什么。
```

说完就停下，等用户操作。

## 重启之后

你现在处在插件安装之后新开的会话里。这一部分保持简短：这段对话也会被保存进记忆。

1. 确认插件已经生效：当前对话里有 `<openviking-context>` 块，或者 OpenViking 工具可用。调用 OpenViking 的 `health` 工具（名字可能带前缀，例如 `mcp__openviking__health` 或 `openviking_health`）。两者都没有时，见[故障排查](#故障排查)的「重启后没有召回」。
2. 结合召回的上下文回答用户的问题。不够时，用 OpenViking 的 `search` 工具并传 `mode="context"`。如果没有召回到相关内容（没有导入时这是正常的），用一行说明，然后根据项目本身回答：最近的提交、未提交的改动、README。刚导入的会话可能要几分钟才能处理完。
3. 在同一条回复里，答完问题后用最多三行收尾：
   - 接入了哪个工具，以及用户消息里写的历史导入结果（不要自己猜）；
   - 在这个宿主上哪里能看到 OpenViking 的作用（[附录 A](#附录-a-各宿主说明) 的「在哪里能看到」）；
   - 以后想导入别的项目，可以说「把 <项目> 的会话导入 OpenViking」，届时重新执行步骤 3。

## 附录 A：各宿主说明

| 宿主 | `--harness` | 重启或信任 | 在哪里能看到 | 历史导入 | 用户级规则文件（步骤 4，仅供参考） |
|---|---|---|---|---|---|
| Claude Code | `claude` | 退出后重新运行 `claude`；不要用 `/reload-plugins` 或 `claude --resume` | 用到 OpenViking 的回答下方有来源卡片（Claude Code 2.1.286 及以上）；`/openviking-memory:ov` 查看状态 | 支持 | `~/.claude/CLAUDE.md` |
| Codex | `codex` | 退出后重新运行 `codex`；出现「Hooks need review」时选 **Trust all and continue**（之后也可在 `/hooks` 中处理） | 回答后有一行 OV-Usage，例如 `OpenViking · 3 sources · 1 past event · 2 team docs · 1 read` | 支持 | `~/.codex/AGENTS.md`；同目录有 `AGENTS.override.md` 时 Codex 读的是它 |
| TraeCode CLI 2.0 | `trae-cli` | 退出后重新运行 `trae-cli`（或 `traecli`）；出现「Hooks need review」时选 **Trust all and continue**（之后也可在 `/hooks` 中处理） | 回答后有一行 OV-Usage，与 Codex 相同 | 转成 JSONL 导入；会话在 `~/.trae/cli/sessions/`，和 Codex 相同的 rollout 格式 | 未确认 |
| Cursor | `cursor` | 完全退出 Cursor 后重新打开，新建 Agent 会话（**+**）；不要在恢复出来的安装对话里继续 | Cursor 设置 → Hooks 里的 OpenViking hook；设置 → Tools & MCPs 里 `openviking` 显示已连接 | 转成 JSONL 导入；会话位置未确认 | Cursor 设置里的 User Rules，请用户粘贴 |
| TRAE / TRAE CN | `trae` / `trae-cn` | 完全退出应用后重新打开，新建会话；无需批准 hook | MCP 设置里 `openviking` 显示已连接 | 转成 JSONL 导入；会话位置未确认 | 应用设置里的个人规则，请用户粘贴 |
| ZCode | `zcode` | 重启 ZCode | 无文档说明 | 转成 JSONL 导入；会话位置未确认 | 应用设置里的个人规则，请用户粘贴 |
| Kimi Code | `kimicode` | 退出 Kimi Code 后新建会话；不要用 `/reload`，也不要恢复当前会话 | 无文档说明 | 转成 JSONL 导入；会话在 `~/.kimi-code/sessions/`（设置了 `$KIMI_CODE_HOME` 时在该目录下），每个会话的 `state.json` 带 cwd，消息在 `agents/main/wire.jsonl` | `~/.kimi-code/AGENTS.md`（设置了 `$KIMI_CODE_HOME` 时在该目录下） |
| OpenCode | `opencode` | 重启 OpenCode | 无文档说明 | 转成 JSONL 导入；会话在 SQLite 库 `~/.local/share/opencode/opencode.db`，没有 `sqlite3` 时可用 Node 22 的 `node:sqlite` 只读打开；`synthetic` 的文本是注入内容，`parent_id` 非空的是子代理会话 | `~/.config/opencode/AGENTS.md` |
| pi | `pi` | 重启 pi | 底栏显示 `OV ✓`；`/viking` 查看状态 | 转成 JSONL 导入；会话在 `~/.pi/agent/sessions/*/*.jsonl`，每个文件第一行 `{"type":"session"}` 带 id 和 cwd | `~/.pi/agent/AGENTS.md` |
| DeepSeek Harness | `dsh` | 用安装器输出中 `DeepSeek Harness profile：` 一行的 profile 新开会话：`dsh --profile <profile>` | 会话顶部显示「上下文注入 · openviking-memory」 | 转成 JSONL 导入；会话在 `~/.dsh/sessions/` | 未确认 |

- **Codex 的审批。** Codex 每次运行需要联网的命令前都会请求批准。安装器那一次选 **Yes, proceed**，不要选「以后以这个前缀开头的命令不再询问」，那条规则会把 key 存下来。OpenViking 的 `search` 工具第一次运行时，Codex 会请求允许，选 **Always allow** 即可。
- **无文档说明。** 在这些宿主上，用户可以直接问你收到了哪些 OpenViking 上下文，也可以运行[故障排查](#故障排查)里的诊断。命令行宿主可以在启动前设置 `OPENVIKING_DEBUG=1`，hook 日志会写到 `~/.openviking/logs/`；从程序坞或启动器打开的桌面应用读不到 shell 里设置的环境变量。

## 故障排查

| 现象 | 处理 |
|---|---|
| `服务拒绝了该 API key（HTTP 401/403）` 或 `该服务需要 API key`（英文 `The server rejected the API key`、`The server requires an API key`） | key 错误、已失效，或属于其他账号。请用户到 OpenViking 控制台（用户管理 → API Key）重新复制 key，用新的 `--api-key` 重跑安装命令。自部署服务要用 user key，root key 不能读写记忆。 |
| `无法连接 <url>`（英文 `Cannot reach`） | 运行 `curl -sS -o /dev/null -w '%{http_code}\n' <url>/health`。只要返回 HTTP 状态码（包括 401），就说明服务可达。超时或连接错误说明被代理、VPN 或防火墙挡住了，请用户处理。 |
| 当前宿主出现 `未找到 … CLI，跳过：<命令>`（英文 `CLI not found; skipping`，例如 `未找到 Codex 格式 CLI，跳过：codex`） | 安装器所在的 shell 找不到这个宿主的命令，常见原因是 `PATH` 只在交互式 shell 里设置。请用户在自己的终端里（能直接运行该命令的地方）运行他原始消息中的安装命令，然后从步骤 1 继续。不要把命令贴回给用户，命令里有 key。 |
| 没有 Node.js，或版本低于 18（pi 需要 22.19 及以上） | 请用户安装或升级 Node.js，然后重跑安装器。 |
| 沙箱拦截了联网或写入主目录 | 请用户批准在沙箱外运行安装器。不要绕过沙箱。 |
| 宿主拒绝执行 `curl … \| bash` | 请用户在自己的终端里运行他原始消息中的安装命令，然后从步骤 1 继续。不要把命令贴回给用户，命令里有 key。 |
| 当前宿主不在 `完成` 里，`未完成` 里也没提到它 | 安装器只会装进它检测到的工具，没检测到当前宿主。只为这个宿主重跑一次，`--harness` 的值取[附录 A](#附录-a-各宿主说明)的 `--harness` 列：`curl -fsSL https://openviking.ai/install \| bash -s -- --yes --harness <值>`。已保存的连接会保留，不用再带 key。然后从步骤 1 继续。 |
| Windows | 安装器只支持 macOS 和 Linux。告诉用户后停止。 |
| docs.openviking.net 打不开 | 换成 docs.openviking.ai，路径不变。安装器本身会自动尝试两个站点。 |
| 找不到 `ov` 命令 | 用 `--yes --harness cli` 重跑安装器，已保存的连接会保留。如果 `ov` 已安装但不在 `PATH` 里，把 `$(npm prefix -g)/bin` 加进 `PATH`；安装器回退到 `~/.local` 时则是 `~/.local/bin`。 |
| `ov` 提示需要先设置显示语言 | 运行 `ov language zh-CN`（或 `ov language en`）。 |
| DeepSeek Harness：`dsh plugin add` 失败 | 安装 pnpm（`npm i -g pnpm`），再用 `--harness dsh` 重跑安装器。 |
| `安装校验失败`（英文 `Installation validation failed`） | 重跑安装器，重跑是安全的。仍然失败，就把失败的检查项告诉用户。 |
| 导入脚本提示 `not supported yet` | 脚本不能直接读这个宿主的历史：按步骤 3 的「其他宿主：先把会话转成 JSONL」转换后用 `--input` 导入。 |
| 重启后没有召回 | Codex 和 TraeCode CLI：hook 没有被信任，在 `/hooks` 中信任，并在 `/plugins` 中确认 `openviking-memory` 已启用。Cursor 和 TRAE：应用没有完全退出。其他情况运行诊断：Claude Code 用 `/openviking-memory:ov` 或 `ov-memory-doctor` skill；Codex 用 `$ov-memory-doctor` skill；Cursor、TRAE、TRAE CN、ZCode 运行 `node ~/.openviking/agent-integrations/<cursor\|trae\|trae-cn\|zcode>/scripts/ov-memory-doctor.mjs`；pi 用 `/viking`；DeepSeek Harness 运行 `dsh --profile <profile> --dump-config`（`<profile>` 是安装器输出的那个），输出里应包含 `openviking-memory`。 |

各宿主的详细说明：https://docs.openviking.net/zh/agent-integrations/01-overview
