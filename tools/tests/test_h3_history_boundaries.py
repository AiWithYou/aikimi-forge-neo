from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest import TestCase, main, mock

from modules_forge.minimax_h3_bridge import (
    H3BridgeError,
    HistoryItem,
    _resolve_history_selection,
    cache_history_video,
    list_history,
)


class H3HistoryBoundaryTests(TestCase):
    def test_unknown_selections_are_rejected_without_resolving_the_supplied_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "MiniMax_H3_valid.mp4"
            video.write_bytes(b"generated")
            item = HistoryItem(video, 1.0, "Forge Neo")
            for value in ("../../private.mp4", r"\\untrusted.invalid\share\private.mp4", "unknown-id"):
                queried = []
                supplied = Path(value)
                original = Path.resolve

                def resolve(path, *args, supplied=supplied, queried=queried, original=original, **kwargs):
                    if path == supplied:
                        queried.append(path)
                        return root / "unlisted.mp4"
                    return original(path, *args, **kwargs)

                with self.subTest(value=value), mock.patch.object(Path, "resolve", resolve):
                    with self.assertRaises(H3BridgeError):
                        _resolve_history_selection(value, [item])
                self.assertEqual(queried, [])

    def test_registered_file_paths_are_not_accepted_as_browser_history_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "MiniMax_H3_valid.mp4"
            video.write_bytes(b"generated")
            item = HistoryItem(video, 1.0, "Forge Neo")
            with self.assertRaises(H3BridgeError):
                _resolve_history_selection(str(video), [item])

    def test_history_does_not_adopt_links_outside_each_output_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "forge-output"
            runtime = root / "runtime"
            runtime_output = runtime / "output/video"
            output.mkdir()
            runtime_output.mkdir(parents=True)
            outside = root / "private.mp4"
            outside.write_bytes(b"private")
            valid = output / "MiniMax_H3_valid.mp4"
            valid.write_bytes(b"generated")
            for folder in (output, runtime_output):
                link = folder / "MiniMax_H3_external.mp4"
                try:
                    link.symlink_to(outside)
                except OSError as exc:
                    self.skipTest(f"File symlinks are unavailable: {exc}")
            (output / "MiniMax_H3_directory.mp4").mkdir()

            items = list_history(runtime, output)

            self.assertEqual([item.path for item in items], [valid.resolve()])

    def test_history_cache_replaces_a_link_without_reading_its_external_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "runtime/video.mp4"
            source.parent.mkdir()
            source.write_bytes(b"generated")
            output = root / "forge-output"
            item = HistoryItem(source, source.stat().st_mtime, "ComfyUI")
            cached = Path(cache_history_video(item.public_id, [item], output))
            cached.unlink()
            outside = root / "private.mp4"
            outside.write_bytes(b"sensitive")
            # Same size and a newer timestamp used to skip the cache copy.
            os.utime(outside, (source.stat().st_mtime + 5, source.stat().st_mtime + 5))
            try:
                cached.symlink_to(outside)
            except OSError as exc:
                self.skipTest(f"File symlinks are unavailable: {exc}")

            result = Path(cache_history_video(item.public_id, [item], output))

            self.assertFalse(result.is_symlink())
            self.assertEqual(result.read_bytes(), b"generated")
            self.assertEqual(outside.read_bytes(), b"sensitive")


if __name__ == "__main__":
    main()
