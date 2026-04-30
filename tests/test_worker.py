from queue import Empty

import pytest

from screenlens_detection.worker import _LatestFrameQueue


def test_latest_frame_queue_drops_stale_frame() -> None:
    frame_queue = _LatestFrameQueue()

    frame_queue.put("first")
    frame_queue.put("second")

    assert frame_queue.get(timeout=0.01) == "second"
    assert frame_queue.dropped_frames == 1

    with pytest.raises(Empty):
        frame_queue.get(timeout=0.01)
