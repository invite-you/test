#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

ANDROID = "{http://schemas.android.com/apk/res/android}"
TEXT_EXTENSIONS = {
    ".java", ".kt", ".kts", ".xml", ".json", ".txt", ".properties",
    ".yml", ".yaml", ".smali", ".js", ".dart", ".html", ".md", ".arb",
    ".gradle", ".toml", ".cfg", ".conf",
}

SDK_MARKERS: dict[str, tuple[str, ...]] = {
    "Flutter": ("io.flutter", "flutterembedding", "libflutter.so", "flutter_assets"),
    "React Native": ("com.facebook.react", "libreactnativejni.so", "index.android.bundle"),
    "Firebase Core": ("com.google.firebase", "firebaseapp", "google_app_id"),
    "Firebase Analytics / Google Measurement": ("com.google.android.gms.measurement", "firebase.analytics"),
    "Firebase Crashlytics": ("com.google.firebase.crashlytics", "crashlytics"),
    "Firebase Messaging": ("com.google.firebase.messaging", "firebase_messaging"),
    "Sentry": ("io.sentry", "sentry_dsn", "sentry.io"),
    "AppsFlyer": ("com.appsflyer", "appsflyer"),
    "PostHog": ("posthog",),
    "Amplitude": ("com.amplitude", "amplitude"),
    "Mixpanel": ("mixpanel",),
    "Microsoft Clarity": ("com.microsoft.clarity", "clarity"),
    "RevenueCat": ("com.revenuecat", "purchases_flutter", "revenuecat"),
    "Google Play Billing": ("com.android.billingclient", "billingclient"),
    "OneSignal": ("onesignal",),
    "Intercom": ("io.intercom", "intercom"),
    "Facebook SDK": ("com.facebook", "facebook_app_id"),
    "Google Sign-In": ("com.google.android.gms.auth.api.signin", "google_sign_in"),
    "Microsoft Identity / MSAL": ("com.microsoft.identity", "msal"),
    "OkHttp": ("okhttp3",),
    "Retrofit": ("retrofit2",),
    "Ktor": ("io.ktor",),
    "gRPC": ("io.grpc", "grpc"),
    "WebSocket": ("websocket",),
    "Socket.IO": ("socket.io", "socketio"),
    "WebRTC": ("org.webrtc", "webrtc"),
    "LiveKit": ("io.livekit", "livekit"),
    "Agora": ("io.agora", "agora"),
    "Twilio": ("com.twilio", "twilio"),
    "Room": ("androidx.room",),
    "DataStore": ("androidx.datastore",),
    "SQLCipher": ("sqlcipher",),
    "Realm": ("io.realm", "realm"),
    "WorkManager": ("androidx.work",),
    "ExoPlayer / Media3": ("androidx.media3", "exoplayer"),
    "Google ML Kit": ("com.google.mlkit", "mlkit"),
    "TensorFlow Lite": ("tensorflowlite", "tflite", ".tflite"),
    "ONNX Runtime": ("onnxruntime", ".onnx"),
    "OpenAI": ("api.openai.com", "openai"),
    "Anthropic": ("api.anthropic.com", "anthropic"),
    "DeepSeek": ("deepseek",),
    "Zhipu / GLM": ("bigmodel.cn", "zhipu", "glm-"),
    "Google Speech": ("speech.googleapis.com", "google.cloud.speech"),
    "Azure Speech": ("speech.microsoft.com", "cognitiveservices"),
    "AWS Transcribe": ("transcribe.amazonaws.com", "awstranscribe"),
    "Deepgram": ("deepgram",),
    "AssemblyAI": ("assemblyai",),
    "Speechmatics": ("speechmatics",),
}

EVIDENCE_GROUPS: dict[str, tuple[str, ...]] = {
    "audio_capture": (
        "android.media.audiorecord", "audiorecord", "android.media.mediarecorder",
        "mediarecorder", "record_audio", "startrecording", "audiocapture",
        "foregroundservicetype=\"microphone\"", "foregroundservicetype=\"mediaProjection\"",
    ),
    "streaming_and_upload": (
        "websocket", "socket.io", "grpc", "multipart", "uploadpart", "resumable",
        "tus", "content-range", "chunked", "presigned", "signedurl", "putobject",
        "backgroundupload", "uploadworker", "retry-after",
    ),
    "background_execution": (
        "androidx.work", "workmanager", "startforeground", "foregroundservice",
        "wakelock", "boot_completed", "jobservice", "expeditedworkrequest",
    ),
    "transcription_and_speaker": (
        "transcrib", "speech-to-text", "speech_to_text", "diariz", "speaker",
        "voice match", "voicematch", "voiceprint", "embedding", "vad", "whisper",
        "speechrecognizer", "recognitionservice",
    ),
    "meeting_memory_and_search": (
        "meeting history", "meeting_history", "semantic search", "vector", "embedding",
        "memory", "contact", "calendar", "transcript", "summary", "visual note",
    ),
    "security_and_integrity": (
        "certificatepinner", "network_security_config", "usescleartexttraffic", "trustmanager",
        "x509trustmanager", "hostnameverifier", "playintegrity", "integritymanager",
        "safetynet", "rootbeer", "magisk", "frida", "debuggable", "allowbackup",
    ),
    "authentication_and_billing": (
        "oauth", "openid", "pkce", "google_sign_in", "msal", "billingclient",
        "revenuecat", "purchases", "subscription", "otp", "passwordless",
    ),
}

URL_RE = re.compile(r"https?://[^\s\"'<>\\]+", re.I)
DOMAIN_RE = re.compile(
    r"(?<![\w.-])(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"(?:com|net|org|io|ai|app|dev|co|cloud|me|xyz|sg|id|tech|site|live|store|services)"
    r"(?::\d+)?",
    re.I,
)
SENSITIVE_QUERY_KEYS = re.compile(r"^(token|key|secret|sig|signature|auth|code|jwt|api[_-]?key)$", re.I)
SECRET_PATTERNS = {
    "Google API key": re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    "OpenAI-style key": re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    "Bearer token literal": re.compile(r"(?i)Bearer\s+[A-Za-z0-9._-]{16,}"),
    "Generic secret assignment": re.compile(
        r"(?i)(?:api[_-]?key|client[_-]?secret|access[_-]?token|private[_-]?key)\s*[:=]\s*[\"'][^\"']{8,}[\"']"
    ),
}


def run(args: list[str], *, timeout: int = 90) -> str:
    try:
        return subprocess.check_output(
            args, stderr=subprocess.STDOUT, text=True, errors="replace", timeout=timeout
        )
    except Exception as exc:
        return f"[command failed: {exc}]"


def read_text(path: Path, max_bytes: int = 8_000_000) -> str:
    try:
        data = path.read_bytes()[:max_bytes]
        return data.decode("utf-8", "replace")
    except Exception:
        return ""


def safe_rel(path: Path, roots: list[Path]) -> str:
    for root in roots:
        try:
            return str(path.relative_to(root))
        except ValueError:
            continue
    return str(path)


def sanitize_url(raw: str) -> str:
    raw = raw.rstrip(".,);]}>'\"")[:1000]
    try:
        parts = urlsplit(raw)
        query = []
        for key, value in parse_qsl(parts.query, keep_blank_values=True):
            query.append((key, "<redacted>" if SENSITIVE_QUERY_KEYS.match(key) else value[:160]))
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))[:500]
    except Exception:
        return re.sub(
            r"([?&](?:token|key|secret|sig|signature|auth|code)=)[^&\s]+",
            r"\1<redacted>",
            raw,
            flags=re.I,
        )[:500]


def iter_text_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            try:
                if not path.is_file() or path.stat().st_size > 5_000_000:
                    continue
            except OSError:
                continue
            if path.suffix.lower() in TEXT_EXTENSIONS or path.name in {
                "AssetManifest.json", "FontManifest.json", "GeneratedPluginRegistrant.java"
            }:
                files.append(path)
    return files


def make_snippet(text: str, needle: str, context: int = 5) -> str:
    lines = text.splitlines()
    low_needle = needle.lower()
    for idx, line in enumerate(lines):
        if low_needle in line.lower():
            start = max(0, idx - context)
            end = min(len(lines), idx + context + 1)
            body = []
            for pos in range(start, end):
                body.append(f"{pos + 1:6d}: {lines[pos][:400]}")
            return "\n".join(body)
    return ""


def mask_secret(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8", "ignore")).hexdigest()[:12]
    return f"<redacted sha256:{digest} length:{len(value)}>"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--apks-dir", required=True, type=Path)
    parser.add_argument("--apktool", required=True, type=Path)
    parser.add_argument("--jadx", required=True, type=Path)
    parser.add_argument("--extracted", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--blutter", type=Path)
    args = parser.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    selected_dir = out / "selected_code"
    selected_dir.mkdir(exist_ok=True)

    roots = [args.jadx / "sources", args.jadx / "resources", args.apktool, args.extracted]
    if args.blutter and args.blutter.exists():
        roots.append(args.blutter)

    apk_files = sorted(args.apks_dir.rglob("*.apk"))
    inventory_lines: list[str] = []
    all_archive_names: list[str] = []
    for apk in apk_files:
        digest = hashlib.sha256(apk.read_bytes()).hexdigest()
        inventory_lines.append(f"{apk.name}\t{apk.stat().st_size}\tsha256:{digest}")
        listing = run(["unzip", "-Z1", str(apk)], timeout=60)
        all_archive_names.extend(f"{apk.name}:{line}" for line in listing.splitlines())
    (out / "apk_inventory.tsv").write_text("\n".join(inventory_lines), encoding="utf-8")
    (out / "archive_entries.txt").write_text("\n".join(all_archive_names), encoding="utf-8")
    archive_blob = "\n".join(all_archive_names).lower()

    badging = read_text(out / "aapt_badging.txt")
    first = badging.splitlines()[0] if badging else ""
    def badging_attr(name: str) -> str:
        match = re.search(rf"\b{name}='([^']*)'", first)
        return match.group(1) if match else "unknown"
    package_name = badging_attr("name")
    version_code = badging_attr("versionCode")
    version_name = badging_attr("versionName")
    min_match = re.search(r"sdkVersion:'([^']+)'", badging)
    target_match = re.search(r"targetSdkVersion:'([^']+)'", badging)
    min_sdk = min_match.group(1) if min_match else "unknown"
    target_sdk = target_match.group(1) if target_match else "unknown"

    manifest_path = args.apktool / "AndroidManifest.xml"
    manifest_text = read_text(manifest_path)
    if manifest_text:
        (out / "AndroidManifest.xml").write_text(manifest_text, encoding="utf-8")
    permissions: list[str] = []
    components: list[tuple[str, str, str, str, str]] = []
    app_attrs: dict[str, str] = {}
    intent_schemes: set[str] = set()
    try:
        tree = ET.parse(manifest_path)
        root = tree.getroot()
        for tag in ("uses-permission", "uses-permission-sdk-23"):
            for node in root.findall(tag):
                value = node.get(ANDROID + "name")
                if value:
                    permissions.append(value)
        app = root.find("application")
        if app is not None:
            for key in (
                "debuggable", "allowBackup", "fullBackupContent", "dataExtractionRules",
                "usesCleartextTraffic", "networkSecurityConfig", "requestLegacyExternalStorage",
                "extractNativeLibs", "largeHeap",
            ):
                value = app.get(ANDROID + key)
                if value is not None:
                    app_attrs[key] = value
            for tag in ("activity", "activity-alias", "service", "receiver", "provider"):
                for node in app.findall(tag):
                    name = node.get(ANDROID + "name", "")
                    exported = node.get(ANDROID + "exported", "implicit")
                    authority = node.get(ANDROID + "authorities", "")
                    foreground_type = node.get(ANDROID + "foregroundServiceType", "")
                    has_filter = "yes" if node.find("intent-filter") is not None else "no"
                    components.append((tag, name, exported, has_filter, authority or foreground_type))
                    for data in node.findall("./intent-filter/data"):
                        scheme = data.get(ANDROID + "scheme")
                        host = data.get(ANDROID + "host")
                        if scheme:
                            intent_schemes.add(f"{scheme}://{host or ''}")
    except Exception as exc:
        (out / "manifest_parse_error.txt").write_text(str(exc), encoding="utf-8")

    (out / "permissions.txt").write_text("\n".join(sorted(set(permissions))), encoding="utf-8")
    (out / "components.tsv").write_text(
        "type\tname\texported\tintent_filter\tauthority_or_foreground_type\n"
        + "\n".join("\t".join(row) for row in components), encoding="utf-8"
    )

    text_files = iter_text_files(roots)
    combined_parts: list[str] = []
    file_texts: dict[Path, str] = {}
    max_combined = 120_000_000
    combined_size = 0
    for path in text_files:
        text = read_text(path, max_bytes=5_000_000)
        if not text:
            continue
        file_texts[path] = text
        if combined_size < max_combined:
            chunk = f"\n\n###FILE:{safe_rel(path, roots)}\n{text}\n"
            combined_parts.append(chunk)
            combined_size += len(chunk)
    combined = "".join(combined_parts)
    combined_lower = combined.lower()
    marker_blob = archive_blob + "\n" + combined_lower

    sdk_hits: dict[str, list[str]] = {}
    for sdk, markers in SDK_MARKERS.items():
        found = sorted({marker for marker in markers if marker.lower() in marker_blob})
        if found:
            sdk_hits[sdk] = found

    framework: list[str] = []
    if "libflutter.so" in archive_blob or "flutter_assets" in archive_blob or "Flutter" in sdk_hits:
        framework.append("Flutter / Dart AOT")
    if "index.android.bundle" in archive_blob or "libreactnativejni.so" in archive_blob or "React Native" in sdk_hits:
        framework.append("React Native")
    if "assets/www/" in archive_blob and ("cordova" in marker_blob or "capacitor" in marker_blob):
        framework.append("Cordova / Capacitor")
    if not framework:
        framework.append("Native Android or an unrecognized framework")

    package_counts: Counter[str] = Counter()
    source_paths = [p for p in text_files if (args.jadx / "sources") in p.parents]
    for path in source_paths:
        try:
            rel = path.relative_to(args.jadx / "sources")
        except ValueError:
            continue
        parts = rel.parts[:-1]
        package = ".".join(parts[:3]) if parts else "<root>"
        package_counts[package] += 1
    (out / "package_counts.tsv").write_text(
        "\n".join(f"{count}\t{name}" for name, count in package_counts.most_common()), encoding="utf-8"
    )

    urls: set[str] = set()
    domains: set[str] = set()
    secret_hits: defaultdict[str, set[str]] = defaultdict(set)
    for path, text in file_texts.items():
        rel = safe_rel(path, roots)
        for match in URL_RE.findall(text):
            value = sanitize_url(match)
            urls.add(value)
            try:
                host = urlsplit(value).hostname
                if host:
                    domains.add(host.lower())
            except Exception:
                pass
        for match in DOMAIN_RE.findall(text):
            domains.add(match.lower().rstrip("."))
        for label, pattern in SECRET_PATTERNS.items():
            for match in pattern.findall(text):
                raw = match if isinstance(match, str) else "".join(match)
                secret_hits[label].add(f"{rel}: {mask_secret(raw)}")

    native_strings_path = out / "native_strings_all.txt"
    native_text = read_text(native_strings_path, max_bytes=40_000_000)
    for match in URL_RE.findall(native_text):
        value = sanitize_url(match)
        urls.add(value)
        try:
            host = urlsplit(value).hostname
            if host:
                domains.add(host.lower())
        except Exception:
            pass
    for match in DOMAIN_RE.findall(native_text):
        domains.add(match.lower().rstrip("."))

    noisy_suffixes = ("example.com", "schema.org", "w3.org", "apache.org", "gnu.org")
    domains = {d for d in domains if not d.endswith(noisy_suffixes)}
    (out / "urls.txt").write_text("\n".join(sorted(urls)), encoding="utf-8")
    (out / "domains.txt").write_text("\n".join(sorted(domains)), encoding="utf-8")
    secret_lines = []
    for label, hits in sorted(secret_hits.items()):
        secret_lines.append(f"[{label}] count={len(hits)}")
        secret_lines.extend(sorted(hits)[:50])
    (out / "hardcoded_secret_indicators_redacted.txt").write_text("\n".join(secret_lines), encoding="utf-8")

    model_assets: list[str] = []
    interesting_assets: list[str] = []
    for entry in all_archive_names:
        low = entry.lower()
        if any(token in low for token in (".tflite", ".onnx", ".gguf", ".bin", "whisper", "model", "vocab", "tokenizer", "sentencepiece")):
            model_assets.append(entry)
        if any(token in low for token in ("flutter_assets", "assetmanifest", ".json", ".arb", ".env", "config", "network_security", "firebase")):
            interesting_assets.append(entry)
    (out / "model_assets.txt").write_text("\n".join(sorted(set(model_assets))), encoding="utf-8")
    (out / "interesting_assets.txt").write_text("\n".join(sorted(set(interesting_assets))[:10000]), encoding="utf-8")

    evidence_output: list[str] = []
    evidence_counts: dict[str, int] = {}
    for group, keywords in EVIDENCE_GROUPS.items():
        evidence_output.append(f"\n\n## {group}\n")
        hits = 0
        seen: set[tuple[str, str]] = set()
        for path, text in file_texts.items():
            low = text.lower()
            for keyword in keywords:
                if keyword.lower() not in low:
                    continue
                rel = safe_rel(path, roots)
                key = (rel, keyword.lower())
                if key in seen:
                    continue
                seen.add(key)
                snippet = make_snippet(text, keyword)
                if snippet:
                    evidence_output.append(f"\n### {rel} — `{keyword}`\n```text\n{snippet}\n```\n")
                    hits += 1
                if hits >= 12:
                    break
            if hits >= 12:
                break
        if hits < 12 and native_text:
            for keyword in keywords:
                if keyword.lower() in native_text.lower():
                    snippet = make_snippet(native_text, keyword, context=2)
                    if snippet:
                        evidence_output.append(f"\n### native_strings_all.txt — `{keyword}`\n```text\n{snippet}\n```\n")
                        hits += 1
                if hits >= 12:
                    break
        evidence_counts[group] = hits
        if hits == 0:
            evidence_output.append("No direct text evidence found in decoded sources/resources/native strings.\n")
    (out / "code_evidence.md").write_text("".join(evidence_output), encoding="utf-8")

    candidates: list[tuple[int, Path]] = []
    for path, text in file_texts.items():
        rel = safe_rel(path, roots).lower()
        if not path.suffix.lower() in {".java", ".kt", ".smali", ".js", ".dart", ".xml"}:
            continue
        score = 0
        if "generatedpluginregistrant" in rel:
            score += 100
        if "ai/meeting/app" in rel or "ai.meeting.app" in text.lower():
            score += 60
        for group, words in EVIDENCE_GROUPS.items():
            score += min(20, sum(2 for word in words if word.lower() in text.lower()))
        if any(noise in rel for noise in ("androidx/", "kotlin/", "java/", "org/apache/", "com/google/protobuf/")):
            score -= 25
        if score > 5:
            candidates.append((score, path))
    candidates.sort(key=lambda item: (-item[0], str(item[1])))
    selected_index: list[str] = []
    used_names: set[str] = set()
    for rank, (score, path) in enumerate(candidates[:40], start=1):
        rel = safe_rel(path, roots)
        dest_name = re.sub(r"[^A-Za-z0-9._-]+", "_", rel)[-180:]
        if dest_name in used_names:
            dest_name = f"{rank}_{dest_name}"
        used_names.add(dest_name)
        text = file_texts[path]
        if len(text) > 250_000:
            excerpts = []
            for group, words in EVIDENCE_GROUPS.items():
                for word in words:
                    snippet = make_snippet(text, word, context=10)
                    if snippet:
                        excerpts.append(f"// {group}: {word}\n{snippet}")
                    if len(excerpts) >= 12:
                        break
                if len(excerpts) >= 12:
                    break
            text = "\n\n".join(excerpts)
        (selected_dir / dest_name).write_text(text, encoding="utf-8")
        selected_index.append(f"score={score}\t{rel}\t{dest_name}")
    (out / "selected_code_index.tsv").write_text("\n".join(selected_index), encoding="utf-8")

    blutter_summary: list[str] = []
    if args.blutter and args.blutter.exists():
        for name in ("objs.txt", "pp.txt", "blutter_frida.js"):
            path = args.blutter / name
            if path.exists():
                text = read_text(path, max_bytes=25_000_000)
                matches = []
                for line in text.splitlines():
                    low = line.lower()
                    if any(k in low for k in (
                        "meeting", "record", "audio", "transcrib", "speaker", "upload", "websocket",
                        "summary", "visual", "calendar", "contact", "subscription", "billing",
                        "api", "repository", "service", "bloc", "provider", "controller",
                    )):
                        matches.append(line[:1000])
                blutter_summary.append(f"## {name}\n" + "\n".join(matches[:5000]))
        asm_dir = args.blutter / "asm"
        if asm_dir.exists():
            asm_names = [str(p.relative_to(args.blutter)) for p in asm_dir.rglob("*") if p.is_file()]
            blutter_summary.append("## asm file inventory\n" + "\n".join(asm_names[:20000]))
    (out / "blutter_filtered.txt").write_text("\n\n".join(blutter_summary), encoding="utf-8")

    class_names = [p.stem for p in source_paths]
    short_names = sum(1 for name in class_names if len(name) <= 2)
    obfuscation_ratio = (short_names / len(class_names)) if class_names else 0.0

    signer_text = read_text(out / "signing.txt")
    signer_lines = [
        line.strip() for line in signer_text.splitlines()
        if "Signer #1 certificate" in line or "Verified using" in line or "Number of signers" in line
    ]
    exported = [row for row in components if row[2] == "true" or (row[2] == "implicit" and row[3] == "yes")]
    sensitive_permissions = [
        p for p in permissions if any(x in p for x in (
            "RECORD_AUDIO", "CAMERA", "READ_CONTACTS", "WRITE_CONTACTS", "READ_CALENDAR",
            "WRITE_CALENDAR", "READ_MEDIA", "READ_EXTERNAL_STORAGE", "WRITE_EXTERNAL_STORAGE",
            "POST_NOTIFICATIONS", "FOREGROUND_SERVICE", "WAKE_LOCK", "BLUETOOTH_CONNECT",
            "READ_PHONE_STATE", "ACCESS_FINE_LOCATION", "ACCESS_COARSE_LOCATION",
        ))
    ]

    report = []
    report.append("# Meeting.ai Android static decompilation report\n")
    report.append("## Package identity")
    report.append(f"- Package: `{package_name}`")
    report.append(f"- Version: `{version_name}` (versionCode `{version_code}`)")
    report.append(f"- minSdk / targetSdk: `{min_sdk}` / `{target_sdk}`")
    report.append(f"- APK files analyzed: `{len(apk_files)}`")
    report.append(f"- Framework detection: **{', '.join(framework)}**")
    report.append(f"- Decompiled source files: `{len(source_paths)}`")
    report.append(f"- Rough short-class-name ratio: `{obfuscation_ratio:.1%}` (only a heuristic for R8/obfuscation)")
    if signer_lines:
        report.append("- Signing evidence: " + " | ".join(f"`{line}`" for line in signer_lines[:10]))

    report.append("\n## Application manifest security posture")
    report.append(f"- Application attributes: `{app_attrs or 'not explicitly set'}`")
    report.append(f"- Exported or implicitly exported components: `{len(exported)}` of `{len(components)}`")
    report.append(f"- Deep-link schemes/hosts: `{', '.join(sorted(intent_schemes)) or 'none found'}`")
    report.append("- Sensitive/relevant permissions: " + (", ".join(f"`{p}`" for p in sorted(sensitive_permissions)) or "none found"))

    report.append("\n## Detected libraries and SDKs")
    if sdk_hits:
        for sdk, markers in sorted(sdk_hits.items()):
            report.append(f"- **{sdk}** — evidence: `{', '.join(markers[:8])}`")
    else:
        report.append("- No known SDK marker matched the decoded text or archive inventory.")

    report.append("\n## Code evidence coverage")
    for group, count in evidence_counts.items():
        report.append(f"- `{group}`: {count} excerpt(s) in `code_evidence.md`")

    report.append("\n## Network and model footprint")
    report.append(f"- Unique URLs recovered: `{len(urls)}`; see `urls.txt` (sensitive query values redacted)")
    report.append(f"- Unique domains recovered: `{len(domains)}`; see `domains.txt`")
    report.append(f"- Possible on-device model/assets markers: `{len(set(model_assets))}`; see `model_assets.txt`")
    report.append(f"- Potential hard-coded secret indicators: `{sum(len(v) for v in secret_hits.values())}`; values are not retained, only redacted hashes and source paths")

    report.append("\n## Files for manual code review")
    report.append(f"- Representative decoded/decompiled files selected: `{len(selected_index)}`")
    report.append("- Index: `selected_code_index.tsv`; content: `selected_code/`")
    report.append("- Decompiled evidence snippets: `code_evidence.md`")
    report.append("- Flutter AOT symbol/object evidence, when Blutter succeeds: `blutter_filtered.txt`")

    report.append("\n## Method limitations")
    report.append("- Static decompilation shows shipped client behavior, SDK integration, endpoints, and local orchestration. It cannot reveal server-side prompts, models, retrieval indexes, or business logic that never ships in the APK.")
    report.append("- Flutter release builds compile Dart to ARM64 native code. JADX recovers Android host/plugin code; Blutter/native-string analysis is required for partial Dart symbol/object recovery and does not reconstruct original Dart source exactly.")
    report.append("- URLs and names can include unused dependency strings. A finding is treated as strong only when supported by manifest, code path, and/or multiple independent markers.")

    (out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print((out / "REPORT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
