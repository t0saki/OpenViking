//! SQLite storage with the Python gateway's schema (kernel.sqlite3 + management.sqlite3).
//! Reads use a small connection pool; writes go through one writer connection, like the
//! Python store's single writer executor. Callers run these on tokio's blocking pool.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::Value;

/// kind -> anchor -> value, so lookups by borrowed strings need no allocation.
pub type Records = HashMap<String, HashMap<String, Value>>;

pub fn rec<'a>(records: &'a Records, kind: &str, anchor: &str) -> Option<&'a Value> {
    records.get(kind).and_then(|m| m.get(anchor))
}

pub fn rec_insert(records: &mut Records, kind: &str, anchor: &str, value: Value) {
    records
        .entry(kind.to_string())
        .or_default()
        .insert(anchor.to_string(), value);
}

#[derive(Clone, Debug, Default)]
pub struct Document {
    pub value: Value,
    pub version: i64,
}

impl Document {
    pub fn empty() -> Self {
        Document {
            value: Value::Object(Default::default()),
            version: 0,
        }
    }
}

pub fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

const INHERITED: &[&str] = &["injection", "hidden", "replacement", "reasoning"];

pub struct Store {
    readers: Mutex<Vec<Connection>>,
    writer: Mutex<Connection>,
    path: std::path::PathBuf,
    mgmt_writer: Mutex<Connection>,
    mgmt_readers: Mutex<Vec<Connection>>,
    mgmt_path: std::path::PathBuf,
}

fn open(path: &Path) -> Connection {
    let c = Connection::open(path).unwrap();
    c.busy_timeout(std::time::Duration::from_secs(30)).unwrap();
    c.execute_batch("PRAGMA foreign_keys=ON").unwrap();
    c
}

fn encode(v: &Value) -> Vec<u8> {
    crate::canon::to_vec(v)
}

fn decode(b: &[u8]) -> Value {
    serde_json::from_slice(b).unwrap_or(Value::Null)
}

impl Store {
    pub fn open(dir: &Path) -> Store {
        std::fs::create_dir_all(dir).unwrap();
        let path = dir.join("kernel.sqlite3");
        let c = open(&path);
        c.execute_batch(
            "PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS sessions (
                scope TEXT, session TEXT, touched REAL NOT NULL,
                PRIMARY KEY(scope,session));
            CREATE TABLE IF NOT EXISTS replay (
                scope TEXT, session TEXT, kind TEXT, anchor TEXT, value BLOB NOT NULL,
                PRIMARY KEY(scope,session,kind,anchor),
                FOREIGN KEY(scope,session) REFERENCES sessions ON DELETE CASCADE);
            CREATE INDEX IF NOT EXISTS replay_anchors ON replay(scope,session,anchor);
            CREATE INDEX IF NOT EXISTS replay_kinds ON replay(scope,kind,anchor);
            CREATE TABLE IF NOT EXISTS state (
                scope TEXT, key TEXT, value BLOB NOT NULL, version INTEGER NOT NULL,
                touched REAL NOT NULL,
                PRIMARY KEY(scope,key));
            CREATE TABLE IF NOT EXISTS capture (
                scope TEXT, session TEXT, value BLOB NOT NULL, version INTEGER NOT NULL,
                ready REAL, lease REAL NOT NULL DEFAULT 0, owner TEXT,
                PRIMARY KEY(scope,session),
                FOREIGN KEY(scope,session) REFERENCES sessions ON DELETE CASCADE);
            CREATE INDEX IF NOT EXISTS capture_ready ON capture(ready,lease);
            CREATE TABLE IF NOT EXISTS deleted_scopes (scope TEXT PRIMARY KEY);",
        )
        .unwrap();
        let mgmt_path = dir.join("management.sqlite3");
        let m = open(&mgmt_path);
        m.execute_batch(
            "PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS objects (
                account TEXT, kind TEXT, id TEXT, revision INTEGER, value BLOB,
                PRIMARY KEY(account,kind,id));
            CREATE TABLE IF NOT EXISTS request_logs (
                id INTEGER PRIMARY KEY, account TEXT, time REAL, value BLOB, user TEXT);
            CREATE INDEX IF NOT EXISTS log_account_user ON request_logs(account,user,time);",
        )
        .unwrap();
        Store {
            readers: Mutex::new(Vec::new()),
            writer: Mutex::new(c),
            path,
            mgmt_writer: Mutex::new(m),
            mgmt_readers: Mutex::new(Vec::new()),
            mgmt_path,
        }
    }

    fn read<T>(&self, f: impl FnOnce(&Connection) -> T) -> T {
        let c = self
            .readers
            .lock()
            .pop()
            .unwrap_or_else(|| open(&self.path));
        let r = f(&c);
        let mut pool = self.readers.lock();
        if pool.len() < 8 {
            pool.push(c);
        }
        r
    }

    fn mgmt_read<T>(&self, f: impl FnOnce(&Connection) -> T) -> T {
        let c = self
            .mgmt_readers
            .lock()
            .pop()
            .unwrap_or_else(|| open(&self.mgmt_path));
        let r = f(&c);
        let mut pool = self.mgmt_readers.lock();
        if pool.len() < 4 {
            pool.push(c);
        }
        r
    }

    pub fn save_object(&self, account: &str, kind: &str, id: &str, value: &Value) {
        self.mgmt_writer
            .lock()
            .execute(
                "INSERT INTO objects VALUES (?,?,?,1,?) ON CONFLICT(account,kind,id) \
                 DO UPDATE SET revision=revision+1,value=excluded.value",
                params![account, kind, id, encode(value)],
            )
            .unwrap();
    }

    /// `ManagementStore.get` (policies are read on every request, uncached, as in Python).
    pub fn get_object(&self, account: &str, kind: &str, id: &str) -> Option<Value> {
        self.mgmt_read(|c| {
            c.query_row(
                "SELECT value,revision FROM objects WHERE account=? AND kind=? AND id=?",
                params![account, kind, id],
                |r| Ok(decode(&r.get::<_, Vec<u8>>(0)?)),
            )
            .optional()
            .unwrap()
        })
    }

    pub fn log(&self, account: &str, value: &Value, user: &str) {
        self.mgmt_writer
            .lock()
            .execute(
                "INSERT INTO request_logs(account,time,value,user) VALUES (?,?,?,?)",
                params![account, now(), encode(value), user],
            )
            .unwrap();
    }

    /// `SQLiteReplayStore.replies` with the index the perf report asks for.
    pub fn replies(&self, scope: &str, anchors: &[String]) -> HashMap<String, HashSet<String>> {
        let anchors = serde_json::to_string(anchors).unwrap();
        self.read(|c| {
            let mut st = c
                .prepare_cached(
                    "SELECT anchor,session FROM replay INDEXED BY replay_kinds WHERE scope=? AND kind=? \
                     AND anchor IN (SELECT value FROM json_each(?))",
                )
                .unwrap();
            let mut owners: HashMap<String, HashSet<String>> = HashMap::new();
            let rows = st
                .query_map(params![scope, "reply", anchors], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
                .unwrap();
            for row in rows {
                let (a, s) = row.unwrap();
                owners.entry(a).or_default().insert(s);
            }
            owners
        })
    }

    /// `SQLiteKernelStore.load`: replay records, the session's observation and capture.
    pub fn load(
        &self,
        scope: &str,
        session: &str,
        anchors: &[&str],
        ancestors: &[String],
    ) -> (Records, Document, Document) {
        let mut sessions = vec![session.to_string()];
        sessions.extend(ancestors.iter().cloned());
        let sessions = serde_json::to_string(&sessions).unwrap();
        let anchors = serde_json::to_string(anchors).unwrap();
        self.read(|c| {
            let mut records = Records::new();
            let mut st = c
                .prepare_cached(
                    "SELECT session,kind,anchor,value FROM replay WHERE scope=? \
                     AND session IN (SELECT value FROM json_each(?)) \
                     AND anchor IN (SELECT value FROM json_each(?)) AND kind<>? \
                     ORDER BY session=?, session",
                )
                .unwrap();
            let rows = st
                .query_map(params![scope, sessions, anchors, "reply", session], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, Vec<u8>>(3)?,
                    ))
                })
                .unwrap();
            for row in rows {
                let (s, kind, anchor, value) = row.unwrap();
                if s == session || INHERITED.contains(&kind.as_str()) {
                    records
                        .entry(kind)
                        .or_default()
                        .insert(anchor, decode(&value));
                }
            }
            let observation = state_read_in(c, scope, &[session.to_string()])
                .remove(session)
                .unwrap_or_else(Document::empty);
            let capture = c
                .query_row(
                    "SELECT value,version FROM capture WHERE scope=? AND session=?",
                    params![scope, session],
                    |r| {
                        Ok(Document {
                            value: decode(&r.get::<_, Vec<u8>>(0)?),
                            version: r.get(1)?,
                        })
                    },
                )
                .optional()
                .unwrap()
                .unwrap_or_else(Document::empty);
            (records, observation, capture)
        })
    }

    pub fn state_read(&self, scope: &str, keys: &[String]) -> HashMap<String, Document> {
        self.read(|c| state_read_in(c, scope, keys))
    }

    fn touch(c: &Connection, scope: &str, session: &str) {
        if c.query_row(
            "SELECT 1 FROM deleted_scopes WHERE scope=?",
            params![scope],
            |_| Ok(()),
        )
        .optional()
        .unwrap()
        .is_some()
        {
            panic!("Gateway user data has been deleted");
        }
        let t = now();
        c.execute(
            "INSERT INTO sessions VALUES (?,?,?) ON CONFLICT(scope,session) \
             DO UPDATE SET touched=excluded.touched WHERE sessions.touched<?",
            params![scope, session, t, t - 60.0],
        )
        .unwrap();
    }

    /// `SQLiteReplayStore.put`: put-if-absent, returning the stored value.
    pub fn replay_put(
        &self,
        scope: &str,
        session: &str,
        kind: &str,
        anchor: &str,
        value: &Value,
    ) -> Value {
        let mut c = self.writer.lock();
        let tx = c
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .unwrap();
        Self::touch(&tx, scope, session);
        tx.execute(
            "INSERT OR IGNORE INTO replay VALUES (?,?,?,?,?)",
            params![scope, session, kind, anchor, encode(value)],
        )
        .unwrap();
        let stored: Vec<u8> = tx
            .query_row(
                "SELECT value FROM replay WHERE scope=? AND session=? AND kind=? AND anchor=?",
                params![scope, session, kind, anchor],
                |r| r.get(0),
            )
            .unwrap();
        tx.commit().unwrap();
        decode(&stored)
    }

    /// `SQLiteStateStore.swap`: single-key compare-and-swap on the version.
    pub fn state_swap(&self, scope: &str, key: &str, version: i64, value: &Value) -> bool {
        let mut c = self.writer.lock();
        let tx = c
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .unwrap();
        let changed = if version > 0 {
            tx.execute(
                "UPDATE state SET value=?,version=version+1,touched=? WHERE scope=? AND key=? AND version=?",
                params![encode(value), now(), scope, key, version],
            )
            .unwrap()
        } else {
            tx.execute(
                "INSERT OR IGNORE INTO state(scope,key,value,version,touched) VALUES (?,?,?,1,?)",
                params![scope, key, encode(value), now()],
            )
            .unwrap()
        };
        tx.commit().unwrap();
        changed > 0
    }

    pub fn capture_get(&self, scope: &str, session: &str) -> Document {
        self.read(|c| {
            c.query_row(
                "SELECT value,version FROM capture WHERE scope=? AND session=?",
                params![scope, session],
                |r| {
                    Ok(Document {
                        value: decode(&r.get::<_, Vec<u8>>(0)?),
                        version: r.get(1)?,
                    })
                },
            )
            .optional()
            .unwrap()
            .unwrap_or_else(Document::empty)
        })
    }

    /// `SQLiteCaptureQueue.swap` (without the worker lease check).
    pub fn capture_swap(
        &self,
        scope: &str,
        session: &str,
        version: i64,
        value: &Value,
        ready: Option<f64>,
    ) -> bool {
        let mut c = self.writer.lock();
        let tx = c
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .unwrap();
        Self::touch(&tx, scope, session);
        let changed = if version > 0 {
            tx.execute(
                "UPDATE capture SET value=?,version=version+1,ready=? WHERE scope=? AND session=? AND version=?",
                params![encode(value), ready, scope, session, version],
            )
            .unwrap()
        } else {
            tx.execute(
                "INSERT OR IGNORE INTO capture(scope,session,value,version,ready) VALUES (?,?,?,1,?)",
                params![scope, session, encode(value), ready],
            )
            .unwrap()
        };
        tx.commit().unwrap();
        changed > 0
    }

    /// `SQLiteCaptureQueue.claim`: lease the next due capture document.
    pub fn capture_claim(&self) -> Option<(String, String, Document)> {
        let mut c = self.writer.lock();
        let tx = c
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .unwrap();
        let t = now();
        let row = tx
            .query_row(
                "SELECT scope,session,value,version FROM capture WHERE ready<=? AND lease<? ORDER BY ready LIMIT 1",
                params![t, t],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        Document { value: decode(&r.get::<_, Vec<u8>>(2)?), version: r.get(3)? },
                    ))
                },
            )
            .optional()
            .unwrap();
        if let Some((scope, session, _)) = &row {
            tx.execute(
                "UPDATE capture SET lease=?,owner='rs' WHERE scope=? AND session=?",
                params![t + 120.0, scope, session],
            )
            .unwrap();
        }
        tx.commit().unwrap();
        row
    }

    pub fn capture_release(&self, scope: &str, session: &str) {
        self.writer
            .lock()
            .execute(
                "UPDATE capture SET lease=0,owner=NULL WHERE scope=? AND session=?",
                params![scope, session],
            )
            .unwrap();
    }
}

fn state_read_in(c: &Connection, scope: &str, keys: &[String]) -> HashMap<String, Document> {
    let keys = serde_json::to_string(keys).unwrap();
    let mut st = c
        .prepare_cached("SELECT key,value,version FROM state WHERE scope=? AND key IN (SELECT value FROM json_each(?))")
        .unwrap();
    let rows = st
        .query_map(params![scope, keys], |r| {
            Ok((
                r.get::<_, String>(0)?,
                Document {
                    value: decode(&r.get::<_, Vec<u8>>(1)?),
                    version: r.get(2)?,
                },
            ))
        })
        .unwrap();
    rows.map(|r| r.unwrap()).collect()
}
