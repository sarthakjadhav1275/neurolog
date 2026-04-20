import argparse
import random
from datetime import datetime, timedelta


def _rand_ts(start_dt: datetime, i: int, delta_seconds: int) -> datetime:
    # Deterministic-ish progression: each line advances by delta_seconds (+ jitter).
    jitter = random.randint(-2, 2)
    return start_dt + timedelta(seconds=i * delta_seconds + jitter)


def _pick_service(services: list[str]) -> str:
    return random.choice(services)


def generate_log_lines(
    *,
    start_iso: str,
    lines: int,
    delta_seconds: int,
    seed: int,
    services: list[str],
    failure_every: int,
) -> list[str]:
    random.seed(seed)
    start_dt = datetime.fromisoformat(start_iso)

    normal_templates = [
        "service={svc} request processed latency=<{lat}ms>",
        "service={svc} healthcheck OK uptime=<%s>" % random.randint(1000, 9999),
        "service={svc} cache hit key=<user:{id}>",
        "service={svc} background job completed job=<cleanup> duration=<{dur}ms>",
        "service={svc} metrics emitted cpu=<{cpu}%> mem=<{mem}MB>",
    ]

    # Each pair is (level_a, message_a, level_b, message_b)
    # Messages intentionally do NOT include log levels; the app expects levels only after the timestamp.
    error_pair_templates = [
        # Pair 1 (auth -> db): includes rule keywords
        (
            "ERROR",
            "service={svc1} failed login attempt user=<u{u}> reason=timeout exception",
            "CRITICAL",
            "service={svc2} crashed handler while updating session error=exception",
        ),
        # Pair 2 (api -> worker)
        (
            "WARN",
            "service={svc1} blocked request ip=<10.0.0.{ip}> denied access timeout",
            "ERROR",
            "service={svc2} service failed to process job=<{job}> exception=runtime",
        ),
        # Pair 3 (gateway -> auth)
        (
            "ERROR",
            "service={svc1} timeout from upstream blocked retries=3 failed",
            "CRITICAL",
            "service={svc2} exception while authenticating token denied",
        ),
    ]

    levels = ["INFO", "WARN", "ERROR", "CRITICAL"]
    out: list[str] = []

    for i in range(lines):
        ts = _rand_ts(start_dt, i, delta_seconds)

        # Inject failure "pairs" periodically to create correlated failure patterns.
        if failure_every > 0 and i > 0 and i % failure_every == 0:
            svc1 = _pick_service(services)
            svc2 = _pick_service(services)
            u = random.randint(1, 9999)
            ip = random.randint(1, 254)
            job = random.choice(["sync", "index", "queue", "billing", "worker"])
            tmpl = random.choice(error_pair_templates)
            level_a, msg_a_t, level_b, msg_b_t = tmpl
            msg_a = msg_a_t.format(svc1=svc1, svc2=svc2, u=u, ip=ip, job=job)
            msg_b = msg_b_t.format(svc1=svc1, svc2=svc2, u=u, ip=ip, job=job)

            # First error at ts
            out.append(f"{ts:%Y-%m-%d %H:%M:%S} {level_a} {msg_a}")

            # Second error within 2 minutes (for correlation window)
            ts2 = ts + timedelta(seconds=random.randint(30, 120))
            out.append(f"{ts2:%Y-%m-%d %H:%M:%S} {level_b} {msg_b}")

            continue

        svc = _pick_service(services)
        # Mostly INFO, sometimes WARN
        r = random.random()
        if r < 0.82:
            level = "INFO"
        elif r < 0.95:
            level = "WARN"
        else:
            # small amount of stray errors
            level = random.choice(["ERROR", "CRITICAL"])

        if level == "INFO":
            lat = random.randint(5, 250)
            dur = random.randint(10, 900)
            cpu = random.randint(1, 92)
            mem = random.randint(200, 20480)
            template = random.choice(normal_templates)
            # normalize templates
            msg = template.format(svc=svc, lat=lat, dur=dur, cpu=cpu, mem=mem, id=random.randint(1, 9999))
        else:
            # Include keywords so rule-based anomalies also catch these.
            keyword = random.choice(["failed", "timeout", "crashed", "denied", "blocked", "error", "exception"])
            template_error = random.choice(
                [
                    "service={svc} {kw} while handling request ip=<10.0.0.{ip}>",
                    "service={svc} error={kw} exception=<RuntimeError> service_state=<degraded>",
                    "service={svc} {kw} job=<{job}> retries=<{rt}>",
                ]
            )
            msg = template_error.format(
                svc=svc,
                kw=keyword,
                ip=random.randint(1, 254),
                job=random.choice(["sync", "index", "queue", "billing", "worker"]),
                rt=random.randint(1, 8),
            )

        out.append(f"{ts:%Y-%m-%d %H:%M:%S} {level} {msg}")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a large synthetic log file for NeuroLog testing.")
    parser.add_argument("--output", default="massive_sample.log", help="Output .log file name (in current folder).")
    parser.add_argument("--lines", type=int, default=100000, help="How many log lines to generate.")
    parser.add_argument("--delta-seconds", type=int, default=1, help="Base seconds between log lines.")
    parser.add_argument("--start", default="2026-03-01T00:00:00", help="Start datetime in ISO format.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--failure-every",
        type=int,
        default=2500,
        help="Inject a correlated failure pair every N lines (0 disables).",
    )
    args = parser.parse_args()

    services = ["auth-service", "api-gateway", "db", "worker", "billing", "cache", "scheduler"]

    lines = generate_log_lines(
        start_iso=args.start,
        lines=args.lines,
        delta_seconds=args.delta_seconds,
        seed=args.seed,
        services=services,
        failure_every=args.failure_every,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"Generated {len(lines)} lines -> {args.output}")


if __name__ == "__main__":
    main()

