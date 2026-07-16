import re
import math
import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def parse_ping_file(file_path: str):
    """
    Parse Wifi.txt atau Ethernet.txt dengan format log seperti file Anda.
    Menghasilkan dict berisi:
    - ping_no
    - latency_ms (None jika lost)
    - status ('SUCCESS' / 'LOST')
    - summary
    """
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    ping_pattern = re.compile(
        r"Ping\s+#(\d+):\s+(SUCCESS|LOST/TIMEOUT),\s*time=(\d+)\s*ms",
        re.IGNORECASE
    )

    summary_patterns = {
        "host": re.compile(r"Host\s*:\s*(.+)", re.IGNORECASE),
        "ip": re.compile(r"IP\s*:\s*(.+)", re.IGNORECASE),
        "sent": re.compile(r"Packets Sent\s*:\s*(\d+)", re.IGNORECASE),
        "received": re.compile(r"Packets Received\s*:\s*(\d+)", re.IGNORECASE),
        "lost": re.compile(r"Packets Lost\s*:\s*(\d+)", re.IGNORECASE),
        "loss_pct": re.compile(r"Packet Loss\s*:\s*([\d.]+)\s*%", re.IGNORECASE),
        "min": re.compile(r"Min Time\s*:\s*([\d.]+)\s*ms", re.IGNORECASE),
        "max": re.compile(r"Max Time\s*:\s*([\d.]+)\s*ms", re.IGNORECASE),
        "avg": re.compile(r"Average Time\s*:\s*([\d.]+)\s*ms", re.IGNORECASE),
    }

    ping_no = []
    latency_ms = []
    status = []

    summary = {
        "host": None,
        "ip": None,
        "sent": None,
        "received": None,
        "lost": None,
        "loss_pct": None,
        "min": None,
        "max": None,
        "avg": None,
    }

    for line in lines:
        m = ping_pattern.search(line)
        if m:
            idx = int(m.group(1))
            st = m.group(2).upper()
            t = int(m.group(3))

            ping_no.append(idx)
            if "LOST" in st:
                latency_ms.append(None)
                status.append("LOST")
            else:
                latency_ms.append(t)
                status.append("SUCCESS")
            continue

        for key, pattern in summary_patterns.items():
            sm = pattern.search(line)
            if sm:
                value = sm.group(1).strip()
                if key in {"sent", "received", "lost"}:
                    summary[key] = int(value)
                elif key in {"loss_pct", "min", "max", "avg"}:
                    summary[key] = float(value)
                else:
                    summary[key] = value

    # Hitung fallback kalau summary tidak lengkap
    success_values = [x for x in latency_ms if x is not None]
    lost_count = sum(1 for x in latency_ms if x is None)

    if summary["sent"] is None:
        summary["sent"] = len(ping_no)
    if summary["received"] is None:
        summary["received"] = len(success_values)
    if summary["lost"] is None:
        summary["lost"] = lost_count
    if summary["loss_pct"] is None:
        summary["loss_pct"] = (lost_count / len(ping_no) * 100) if ping_no else 0.0
    if summary["min"] is None:
        summary["min"] = min(success_values) if success_values else None
    if summary["max"] is None:
        summary["max"] = max(success_values) if success_values else None
    if summary["avg"] is None:
        summary["avg"] = (sum(success_values) / len(success_values)) if success_values else None

    return {
        "ping_no": ping_no,
        "latency_ms": latency_ms,
        "status": status,
        "summary": summary,
    }


def find_lost_ranges(ping_no, status):
    """Cari rentang lost berurutan, contoh: [(9,11), (593,595)]"""
    ranges = []
    start = None
    prev = None

    for n, st in zip(ping_no, status):
        if st == "LOST":
            if start is None:
                start = n
            prev = n
        else:
            if start is not None:
                ranges.append((start, prev))
                start = None
                prev = None

    if start is not None:
        ranges.append((start, prev))

    return ranges


def longest_lost_streak(lost_ranges):
    if not lost_ranges:
        return 0
    return max(end - start + 1 for start, end in lost_ranges)


def build_info_box(summary, lost_ranges, timeout_y):
    streak = longest_lost_streak(lost_ranges)
    lines = [
        f"Sent      : {summary['sent']}",
        f"Received  : {summary['received']}",
        f"Lost      : {summary['lost']}",
        f"Loss      : {summary['loss_pct']:.2f}%",
        f"Min       : {summary['min']:.2f} ms" if summary['min'] is not None else "Min       : -",
        f"Max       : {summary['max']:.2f} ms" if summary['max'] is not None else "Max       : -",
        f"Average   : {summary['avg']:.2f} ms" if summary['avg'] is not None else "Average   : -",
        f"Longest lost streak : {streak}",
        f"Timeout shown at    : {timeout_y:.0f} ms",
    ]

    # Tambahan singkat kalau lost range sedikit
    if lost_ranges and len(lost_ranges) <= 5:
        pretty = []
        for a, b in lost_ranges:
            if a == b:
                pretty.append(str(a))
            else:
                pretty.append(f"{a}-{b}")
        lines.append("Lost range : " + ", ".join(pretty))

    return "\n".join(lines)


def plot_ping(file_path: str, output: str = None, title: str = None):
    data = parse_ping_file(file_path)

    ping_no = data["ping_no"]
    latency_ms = data["latency_ms"]
    status = data["status"]
    summary = data["summary"]

    success_x = [x for x, y in zip(ping_no, latency_ms) if y is not None]
    success_y = [y for y in latency_ms if y is not None]
    lost_x = [x for x, y in zip(ping_no, latency_ms) if y is None]

    if success_y:
        data_max = max(success_y)
        timeout_y = max(1000, data_max * 1.1)
    else:
        timeout_y = 1000

    lost_y = [170] * len(lost_x)
    lost_ranges = find_lost_ranges(ping_no, status)

    file_name = Path(file_path).stem
    host = summary.get("host") or "-"
    ip = summary.get("ip") or "-"
    chart_title = title or f"Ping Latency - {file_name}"

    plt.figure(figsize=(15, 7))

    # Plot success
    plt.plot(success_x, success_y, marker="o", markersize=3, linewidth=1.2, label="Success latency")

    # Plot lost marker merah
    if lost_x:
        plt.scatter(
            lost_x,
            lost_y,
            marker="x",
            s=80,
            color="red",
            linewidths=2,
            label="Packet Lost"
        )

    # Arsiran untuk lost berurutan
    for start, end in lost_ranges:
        plt.axvspan(start - 0.5, end + 0.5, alpha=0.12)

    # Garis average
    if summary["avg"] is not None:
        plt.axhline(summary["avg"], linestyle="--", linewidth=1, label=f"Average = {summary['avg']:.2f} ms")

    plt.xlabel("Ping number")
    plt.ylabel("Latency (ms)")
    plt.title(f"{chart_title}\nHost: {host} | IP: {ip}")

    # Biar area timeout tetap kelihatan tapi data normal tetap kebaca
    upper_limit = timeout_y * 1.05
    plt.ylim(-180, 180)

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="lower left")

    # Info box kanan bawah
    info_text = build_info_box(summary, lost_ranges, timeout_y)
    plt.gcf().text(
        0.985, 0.02, info_text,
        ha="right", va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    if output:
        plt.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Grafik disimpan ke: {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot grafik ping dari Wifi.txt atau Ethernet.txt")
    parser.add_argument(
        "file",
        help="Path file input, misal: Wifi.txt atau Ethernet.txt"
    )
    parser.add_argument(
        "-o", "--output",
        help="Nama file output gambar, misal: wifi_plot.png",
        default=None
    )
    parser.add_argument(
        "-t", "--title",
        help="Judul custom grafik",
        default=None
    )
    args = parser.parse_args()

    plot_ping(args.file, args.output, args.title)


if __name__ == "__main__":
    main()