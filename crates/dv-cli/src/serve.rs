//! `dv serve` — minimal drag-and-drop web front-end for `dv pipeline`.
//!
//! The browser is a thin client: it uploads a BAM/CRAM, the server
//! runs the native pipeline (so all of make-examples + the CRAM fix +
//! CoreML/MLProgram apply) and streams stage progress back as a
//! chunked response the page renders live. The final line is
//! `RESULT /result/<token>` pointing at the produced VCF.
//!
//! Slice 1: drag-drop + region + live stage log + VCF download.
//! Richer progress (%, ETA, variant tail, pileup thumbnails) layers
//! onto the same streaming channel later.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};

type Tokens = Arc<Mutex<std::collections::HashMap<String, PathBuf>>>;

const PAGE: &str = include_str!("serve_page.html");

pub fn serve_cmd(ref_fasta: &Path, checkpoint: &Path, port: u16) -> Result<()> {
    let server = tiny_http::Server::http(("0.0.0.0", port))
        .map_err(|e| anyhow::anyhow!("bind 0.0.0.0:{port}: {e}"))?;
    let ref_fasta = ref_fasta.to_path_buf();
    let checkpoint = checkpoint.to_path_buf();
    let tokens: Tokens = Arc::new(Mutex::new(Default::default()));
    let workroot = std::env::temp_dir().join("dv-serve");
    std::fs::create_dir_all(&workroot).ok();

    tracing::info!(%port, "dv serve listening — open http://localhost:{port}/");
    eprintln!("\n  dv serve → http://localhost:{port}/\n");

    for mut req in server.incoming_requests() {
        let url = req.url().to_string();
        let method = req.method().clone();
        let res = match (method, url.as_str()) {
            (tiny_http::Method::Get, "/") => {
                req.respond(html_response(PAGE)).map_err(Into::into)
            }
            (tiny_http::Method::Post, u) if u.starts_with("/run") => {
                let u = u.to_string();
                handle_run(req, &u, &ref_fasta, &checkpoint, &workroot, tokens.clone())
            }
            (tiny_http::Method::Get, u) if u.starts_with("/result/") => {
                handle_result(req, u, tokens.clone())
            }
            _ => req
                .respond(tiny_http::Response::from_string("not found").with_status_code(404))
                .map_err(Into::into),
        };
        if let Err(e) = res {
            tracing::warn!(error = %e, "request failed");
        }
    }
    Ok(())
}

fn html_response(body: &str) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    tiny_http::Response::from_string(body).with_header(
        tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"text/html; charset=utf-8"[..])
            .unwrap(),
    )
}

fn query_param(url: &str, key: &str) -> Option<String> {
    let q = url.split_once('?')?.1;
    for pair in q.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            if k == key {
                return Some(urldecode(v));
            }
        }
    }
    None
}

fn urldecode(s: &str) -> String {
    let b = s.replace('+', " ");
    let mut out = Vec::new();
    let mut it = b.bytes();
    while let Some(c) = it.next() {
        if c == b'%' {
            let h = (it.next(), it.next());
            if let (Some(a), Some(b)) = h {
                if let (Some(x), Some(y)) =
                    ((a as char).to_digit(16), (b as char).to_digit(16))
                {
                    out.push((x * 16 + y) as u8);
                    continue;
                }
            }
        } else {
            out.push(c);
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// A `Read` fed by a worker thread over an mpsc channel; EOF when the
/// sender drops. Lets tiny_http stream the pipeline's progress live.
struct ChanReader {
    rx: Receiver<Vec<u8>>,
    buf: Vec<u8>,
    pos: usize,
}
impl Read for ChanReader {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.buf.len() {
            match self.rx.recv() {
                Ok(chunk) => {
                    self.buf = chunk;
                    self.pos = 0;
                }
                Err(_) => return Ok(0), // sender dropped → EOF
            }
        }
        let n = (self.buf.len() - self.pos).min(out.len());
        out[..n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

fn handle_run(
    mut req: tiny_http::Request,
    url: &str,
    ref_fasta: &Path,
    checkpoint: &Path,
    workroot: &Path,
    tokens: Tokens,
) -> Result<()> {
    let name = query_param(url, "name").unwrap_or_else(|| "input.bam".into());
    let region = query_param(url, "region").unwrap_or_default();
    let ext = Path::new(&name)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("bam")
        .to_lowercase();
    if region.is_empty() {
        return req
            .respond(tiny_http::Response::from_string("missing ?region=").with_status_code(400))
            .map_err(Into::into);
    }

    let token = format!(
        "{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let workdir = workroot.join(&token);
    std::fs::create_dir_all(&workdir).context("mkdir workdir")?;
    let input = workdir.join(format!("input.{ext}"));

    // Slurp the uploaded file body to disk.
    {
        let mut f = std::fs::File::create(&input).context("create upload")?;
        std::io::copy(req.as_reader(), &mut f).context("write upload")?;
    }

    let (tx, rx) = std::sync::mpsc::channel::<Vec<u8>>();
    let exe = std::env::current_exe().context("current_exe")?;
    let ref_fasta = ref_fasta.to_path_buf();
    let checkpoint = checkpoint.to_path_buf();
    let workdir2 = workdir.clone();
    let token2 = token.clone();

    std::thread::spawn(move || {
        let cvo = workdir2.join("cvo.tfrecord.gz");
        let vcf = workdir2.join("out.vcf.gz");
        let contig = region.split(':').next().unwrap_or("chr1").to_string();

        let send = |tx: &Sender<Vec<u8>>, s: &str| {
            let _ = tx.send(s.as_bytes().to_vec());
        };
        send(&tx, &format!("LOG starting pipeline on {name} ({region})\n"));

        let ok = stream_cmd(
            &tx,
            Command::new(&exe)
                .arg("pipeline")
                .args(["--reads".as_ref(), input.as_os_str()])
                .args(["--ref-fasta".as_ref(), ref_fasta.as_os_str()])
                .args(["--region", &region])
                .args(["--checkpoint".as_ref(), checkpoint.as_os_str()])
                .args(["--output".as_ref(), cvo.as_os_str()])
                .args(["--batch-size", "128"])
                .env("RUST_LOG", "info")
                .env(
                    "ORT_DYLIB_PATH",
                    std::env::var("ORT_DYLIB_PATH").unwrap_or_default(),
                ),
        );
        if !ok {
            send(&tx, "ERROR pipeline failed (see server log)\n");
            return;
        }
        send(&tx, "LOG pipeline done — postprocess…\n");
        let ok = stream_cmd(
            &tx,
            Command::new(&exe)
                .arg("postprocess-variants")
                .args(["--cvo".as_ref(), cvo.as_os_str()])
                .args(["--output-vcf".as_ref(), vcf.as_os_str()])
                .args(["--contig", &format!("{contig}:300000000")])
                .args(["--sample-name", "SAMPLE"])
                .env(
                    "ORT_DYLIB_PATH",
                    std::env::var("ORT_DYLIB_PATH").unwrap_or_default(),
                ),
        );
        if !ok || !vcf.exists() {
            send(&tx, "ERROR postprocess failed\n");
            return;
        }
        tokens.lock().unwrap().insert(token2.clone(), vcf);
        send(&tx, &format!("RESULT /result/{token2}\n"));
    });

    let reader = ChanReader {
        rx,
        buf: Vec::new(),
        pos: 0,
    };
    let resp = tiny_http::Response::new(
        tiny_http::StatusCode(200),
        vec![tiny_http::Header::from_bytes(
            &b"Content-Type"[..],
            &b"text/plain; charset=utf-8"[..],
        )
        .unwrap()],
        reader,
        None,
        None,
    );
    req.respond(resp).map_err(Into::into)
}

/// Run a command, forwarding each stderr line to the channel as a
/// `STAGE`/`LOG` event the page parses. Returns success.
fn stream_cmd(tx: &Sender<Vec<u8>>, cmd: &mut Command) -> bool {
    // `dv` logs via tracing to STDOUT; panics/abort go to STDERR.
    let mut child = match cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
        Ok(c) => c,
        Err(e) => {
            let _ = tx.send(format!("ERROR spawn: {e}\n").into_bytes());
            return false;
        }
    };
    if let Some(err) = child.stderr.take() {
        let tx2 = tx.clone();
        std::thread::spawn(move || {
            for ln in BufReader::new(err).lines().map_while(Result::ok) {
                let ln = strip_ansi(&ln);
                if !ln.trim().is_empty() {
                    let _ = tx2.send(format!("LOG {ln}\n").into_bytes());
                }
            }
        });
    }
    if let Some(out) = child.stdout.take() {
        let rdr = BufReader::new(out);
        for raw in rdr.lines().map_while(Result::ok) {
            // tracing colourises output even when piped, so `stage="X"`
            // is really `stage<ESC>=<ESC>"X"` — strip ANSI first.
            let line = strip_ansi(&raw);
            if let Some(stage) = between(&line, "stage=\"", "\"") {
                let ms = after(&line, "ms=").unwrap_or_default();
                let _ = tx.send(format!("STAGE {stage} {ms}\n").into_bytes());
            } else if line.contains("candidate variants")
                || line.contains("emitted=")
                || line.contains("realigner candidate expansion")
            {
                if let Some(msg) = line.splitn(4, ' ').last() {
                    let _ = tx.send(format!("LOG {msg}\n").into_bytes());
                }
            }
        }
    }
    matches!(child.wait(), Ok(s) if s.success())
}

fn between(s: &str, a: &str, b: &str) -> Option<String> {
    let i = s.find(a)? + a.len();
    let j = s[i..].find(b)? + i;
    Some(s[i..j].to_string())
}
fn after(s: &str, a: &str) -> Option<String> {
    let i = s.find(a)? + a.len();
    Some(
        s[i..]
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect(),
    )
}
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            while let Some(&n) = chars.peek() {
                chars.next();
                if n.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn handle_result(req: tiny_http::Request, url: &str, tokens: Tokens) -> Result<()> {
    let tok = url.trim_start_matches("/result/");
    let path = tokens.lock().unwrap().get(tok).cloned();
    match path {
        Some(p) if p.exists() => {
            let f = std::fs::File::open(&p).context("open vcf")?;
            let resp = tiny_http::Response::from_file(f)
                .with_header(
                    tiny_http::Header::from_bytes(
                        &b"Content-Type"[..],
                        &b"application/gzip"[..],
                    )
                    .unwrap(),
                )
                .with_header(
                    tiny_http::Header::from_bytes(
                        &b"Content-Disposition"[..],
                        &b"attachment; filename=\"dv.vcf.gz\""[..],
                    )
                    .unwrap(),
                );
            req.respond(resp).map_err(Into::into)
        }
        _ => req
            .respond(tiny_http::Response::from_string("not ready").with_status_code(404))
            .map_err(Into::into),
    }
}

// Keep `Write`/`BufRead` imports used (silences unused on some toolchains).
#[allow(dead_code)]
fn _imports(_: &dyn Write, _: &dyn BufRead) {}
