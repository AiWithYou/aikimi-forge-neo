"""既存の一回分の推論fixtureを、常駐workerのジョブ関数にする。"""

import textwrap


def worker_script(source):
    lines = []
    for line in source.splitlines():
        if line.startswith("payload = json.loads("):
            continue
        if line.startswith("request_path = "):
            line = 'request_path = pathlib.Path(payload["input_images"][0]).parents[1] / "request.json"'
        lines.append(line)
    return "def resident_run(payload):\n" + textwrap.indent("\n".join(lines), "    ") + "\n"
