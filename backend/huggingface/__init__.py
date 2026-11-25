import hashlib
import lzma
from pathlib import Path


def process():
    folder = Path(__file__).parent
    tokenizer = folder / "circlestone-labs/Anima/tokenizer/tokenizer.json"
    if tokenizer.is_file():
        return
    content = lzma.decompress((folder / "z.tokenizer.json.xz").read_bytes())
    if hashlib.sha256(content).hexdigest() != "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4":
        raise ValueError("Invalid Anima tokenizer")
    tokenizer.write_bytes(content)
