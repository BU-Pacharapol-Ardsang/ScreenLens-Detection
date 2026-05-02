from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from .models import DetectionBox, FrameAnalysis, MonitorSpec
from .overlay_tracks import OverlayBox, OverlayTrackManager
from .windows_capture_exclusion import set_window_capture_exclusion


@dataclass(slots=True, frozen=True)
class _LocalTrack:
    box: OverlayBox
    offset_x: int
    offset_y: int
    confidence: float


@dataclass(slots=True)
class _VisualAnchor:
    key: str
    template: np.ndarray
    core_template: np.ndarray
    core_raw_template: np.ndarray
    box_offset_x: int
    box_offset_y: int
    core_offset_x: int
    core_offset_y: int
    last_box: OverlayBox
    missing_frames: int = 0
    confidence: float = 1.0


def overlay_text_for_box(box: DetectionBox) -> str:
    translated = " ".join(box.translated_text.split())
    if translated:
        return translated
    return " ".join(box.text.split())


def scale_overlay_rect(
    box: DetectionBox,
    *,
    overlay_width: int,
    overlay_height: int,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    if frame_width <= 0 or frame_height <= 0:
        return box.x, box.y, box.w, box.h

    scale_x = overlay_width / frame_width
    scale_y = overlay_height / frame_height
    return (
        max(int(box.x * scale_x), 0),
        max(int(box.y * scale_y), 0),
        max(int(box.w * scale_x), 1),
        max(int(box.h * scale_y), 1),
    )


def overlay_font_pixel_size(rect_height: int) -> int:
    if rect_height <= 0:
        return 1
    return max(min(int(rect_height * 0.62), 28), 1)


class TranslationOverlay(QWidget):
    def __init__(self) -> None:
        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        if hasattr(Qt.WindowType, "WindowTransparentForInput"):
            flags |= Qt.WindowType.WindowTransparentForInput

        super().__init__(None, flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self._monitor: MonitorSpec | None = None
        self._overlay_boxes: list[OverlayBox] = []
        self._capture_exclusion_applied = False
        self._tracking_enabled = False
        self._tracking_mode = "legacy"
        self._realtime_tracking_active = False
        self._tracking_lost_frames = 0
        self._tracking_gray_frame: np.ndarray | None = None
        self._tracking_frame_scale = 1.0
        self._visual_anchors: list[_VisualAnchor] = []
        self._max_tracked_missing_frames = 10
        self._max_realtime_lost_frames = 6
        self._max_local_tracked_boxes = 12
        self._max_anchor_tracked_boxes = 16
        self._max_visible_overlay_boxes = 24
        self._max_visible_predicted_frames = 3
        self._local_tracking_min_confidence = 0.42
        self._anchor_tracking_min_confidence = 0.62
        self._anchor_core_min_confidence = 0.62
        self._scene_change_ratio_threshold = 0.42
        self._track_manager = OverlayTrackManager(
            max_visible_tracks=self._max_visible_overlay_boxes,
            max_predicted_frames=self._max_visible_predicted_frames,
        )

    def show_for_monitor(self, monitor: MonitorSpec) -> None:
        self._monitor = monitor
        self._apply_monitor_geometry(monitor)
        self.show()
        self._ensure_capture_exclusion()
        self.raise_()

    def clear_analysis(self) -> None:
        self._track_manager.clear()
        self._overlay_boxes = []
        self._visual_anchors = []
        self.update()

    def set_tracking_mode(self, mode: str | None) -> None:
        normalized = mode if mode in {"legacy", "anchor"} else "legacy"
        if normalized == self._tracking_mode:
            return
        self._tracking_mode = normalized
        self._visual_anchors = []

    def set_tracking_enabled(self, enabled: bool) -> None:
        self._tracking_enabled = enabled
        if not enabled:
            self._realtime_tracking_active = False
            self.clear_analysis()

    def set_realtime_tracking_active(self, active: bool) -> None:
        self._realtime_tracking_active = active
        self._tracking_gray_frame = None
        if active:
            self._tracking_lost_frames = 0

    def apply_tracking_frame(self, tracking_frame: object) -> None:
        gray_frame = getattr(tracking_frame, "gray_frame", None)
        if gray_frame is None:
            return

        gray = np.asarray(gray_frame)
        if gray.ndim != 2:
            return

        previous_gray = self._tracking_gray_frame
        frame_scale = max(float(getattr(tracking_frame, "frame_scale", 1.0)), 1e-6)
        self._tracking_gray_frame = gray.copy()
        self._tracking_frame_scale = frame_scale
        if not self._tracking_enabled or not self._realtime_tracking_active or not self._overlay_boxes:
            return

        global_confidence = float(getattr(tracking_frame, "global_confidence", 0.0))
        if self._tracking_mode == "anchor":
            probable_scene_change = (
                previous_gray is not None
                and previous_gray.shape == gray.shape
                and self._is_probable_scene_change(
                    previous_gray,
                    gray,
                    global_confidence=global_confidence,
                )
            )
            self._apply_visual_anchor_tracking(
                gray,
                frame_scale=frame_scale,
                global_offset_x=float(getattr(tracking_frame, "global_offset_x", 0.0)),
                global_offset_y=float(getattr(tracking_frame, "global_offset_y", 0.0)),
                global_confidence=global_confidence,
                probable_scene_change=probable_scene_change,
            )
            return

        if previous_gray is None or previous_gray.shape != gray.shape:
            return

        probable_scene_change = self._is_probable_scene_change(
            previous_gray,
            gray,
            global_confidence=global_confidence,
        )
        tracked_boxes = self._track_boxes_between_frames(
            previous_gray,
            gray,
            frame_scale=frame_scale,
            global_offset_x=float(getattr(tracking_frame, "global_offset_x", 0.0)),
            global_offset_y=float(getattr(tracking_frame, "global_offset_y", 0.0)),
            global_confidence=global_confidence,
        )
        if tracked_boxes:
            has_fresh_track = any(box.missing_frames == 0 for box in tracked_boxes)
            if probable_scene_change and not has_fresh_track:
                self.clear_analysis()
                return

            if has_fresh_track:
                self._tracking_lost_frames = 0
            else:
                self._tracking_lost_frames += 1
                if self._tracking_lost_frames >= self._max_realtime_lost_frames:
                    self.clear_analysis()
                    return

            self._overlay_boxes = self._track_manager.update_from_visual_tracking(
                self._limit_overlay_boxes(tracked_boxes)
            )
            self.update()
            return

        if probable_scene_change:
            self.clear_analysis()
            return

        self._tracking_lost_frames += 1
        self._overlay_boxes = self._track_manager.mark_all_occluded()
        self.update()
        if self._tracking_lost_frames >= self._max_realtime_lost_frames:
            self._overlay_boxes = self._track_manager.mark_all_occluded()
            self.update()

    def apply_tracking_offset(self, offset_x: float, offset_y: float, confidence: float) -> None:
        if not self._tracking_enabled or not self._realtime_tracking_active or not self._overlay_boxes:
            return

        if self._tracking_mode == "anchor":
            return

        if confidence < 0.10:
            self._tracking_lost_frames += 1
            if self._tracking_lost_frames >= self._max_realtime_lost_frames:
                self._overlay_boxes = self._track_manager.mark_all_occluded()
                self.update()
            return

        self._tracking_lost_frames = 0
        self._overlay_boxes = self._track_manager.update_from_visual_tracking(
            self._limit_overlay_boxes(
                self._offset_overlay_boxes(
                    self._overlay_boxes,
                    int(round(offset_x)),
                    int(round(offset_y)),
                    increment_missing=False,
                )
            )
        )
        self.update()

    def _apply_visual_anchor_tracking(
        self,
        current_gray: np.ndarray,
        *,
        frame_scale: float,
        global_offset_x: float,
        global_offset_y: float,
        global_confidence: float,
        probable_scene_change: bool,
    ) -> None:
        if not self._visual_anchors:
            return

        tracked_boxes = self._track_boxes_with_visual_anchors(
            current_gray,
            frame_scale=frame_scale,
            global_offset_x=global_offset_x,
            global_offset_y=global_offset_y,
            global_confidence=global_confidence,
        )
        if tracked_boxes:
            self._tracking_lost_frames = 0
            self._overlay_boxes = self._track_manager.update_from_visual_tracking(
                self._limit_overlay_boxes(tracked_boxes)
            )
            self.update()
            return

        self._tracking_lost_frames += 1
        self._overlay_boxes = self._track_manager.mark_all_occluded()
        self.update()
        if probable_scene_change or self._tracking_lost_frames >= 2:
            self._visual_anchors = []

    def _track_boxes_with_visual_anchors(
        self,
        current_gray: np.ndarray,
        *,
        frame_scale: float,
        global_offset_x: float,
        global_offset_y: float,
        global_confidence: float,
    ) -> list[OverlayBox]:
        tracked_boxes: list[OverlayBox] = []
        used_anchor_indices: set[int] = set()
        candidate_boxes = sorted(
            self._overlay_boxes,
            key=self._overlay_box_priority,
            reverse=True,
        )[: self._max_anchor_tracked_boxes]

        for box in candidate_boxes:
            anchor_index = self._find_visual_anchor_index(box, used_anchor_indices)
            if anchor_index is None:
                continue

            anchor = self._visual_anchors[anchor_index]
            match = self._match_visual_anchor(
                anchor,
                box,
                current_gray,
                frame_scale=frame_scale,
                global_offset_x=global_offset_x,
                global_offset_y=global_offset_y,
                global_confidence=global_confidence,
            )
            if match is None:
                continue

            tracked_box, confidence = match
            anchor.last_box = tracked_box
            anchor.missing_frames = 0
            anchor.confidence = confidence
            self._refresh_visual_anchor_template(anchor, tracked_box, current_gray, frame_scale, confidence)
            used_anchor_indices.add(anchor_index)
            tracked_boxes.append(tracked_box)

        self._visual_anchors = [
            anchor
            for index, anchor in enumerate(self._visual_anchors)
            if index in used_anchor_indices
        ]
        tracked_boxes.sort(key=lambda item: (item.y, item.x))
        return tracked_boxes

    def _match_visual_anchor(
        self,
        anchor: _VisualAnchor,
        box: OverlayBox,
        current_gray: np.ndarray,
        *,
        frame_scale: float,
        global_offset_x: float,
        global_offset_y: float,
        global_confidence: float,
    ) -> tuple[OverlayBox, float] | None:
        search_rect = self._anchor_search_rect(
            anchor,
            box,
            current_gray.shape,
            frame_scale=frame_scale,
            global_offset_x=global_offset_x,
            global_offset_y=global_offset_y,
            global_confidence=global_confidence,
        )
        if search_rect is None:
            return None

        match = self._match_template_image(anchor.template, current_gray, search_rect)
        if match is None:
            return None

        match_left, match_top, confidence, confidence_margin = match
        required_confidence = self._anchor_tracking_min_confidence
        if global_confidence < 0.28:
            required_confidence = max(required_confidence, 0.68)
        if anchor.missing_frames:
            required_confidence = max(required_confidence, 0.74)
        if confidence < required_confidence:
            return None
        if confidence_margin < 0.06 and confidence < 0.88:
            return None

        tracked_x = int(round((match_left + anchor.box_offset_x) / frame_scale))
        tracked_y = int(round((match_top + anchor.box_offset_y) / frame_scale))
        tracked = self._place_single_box(box, tracked_x, tracked_y, missing_frames=0)
        if tracked is None:
            return None
        if self._visual_anchor_core_match_score(anchor, tracked, current_gray, frame_scale) < self._anchor_core_min_confidence:
            return None
        return tracked, confidence

    def _anchor_search_rect(
        self,
        anchor: _VisualAnchor,
        box: OverlayBox,
        frame_shape: tuple[int, ...],
        *,
        frame_scale: float,
        global_offset_x: float,
        global_offset_y: float,
        global_confidence: float,
    ) -> tuple[int, int, int, int] | None:
        height, width = frame_shape[:2]
        template_h, template_w = anchor.template.shape[:2]
        expected_left = int(round(box.x * frame_scale)) - anchor.box_offset_x
        expected_top = int(round(box.y * frame_scale)) - anchor.box_offset_y
        if global_confidence >= 0.16:
            expected_left += int(round(global_offset_x * frame_scale))
            expected_top += int(round(global_offset_y * frame_scale))

        box_w = max(int(round(box.w * frame_scale)), 1)
        box_h = max(int(round(box.h * frame_scale)), 1)
        radius_x = min(
            max(box_w * 2, int(64 * frame_scale)),
            max(int(width * 0.18), int(64 * frame_scale)),
        )
        radius_y = min(
            max(box_h * 8, int(80 * frame_scale)),
            max(int(height * 0.35), int(80 * frame_scale)),
        )
        if global_confidence >= 0.28:
            radius_x = min(
                max(box_w, int(40 * frame_scale)),
                max(int(width * 0.10), int(40 * frame_scale)),
            )
            radius_y = min(
                max(box_h * 4, int(60 * frame_scale)),
                max(int(height * 0.22), int(60 * frame_scale)),
            )

        search_left = max(expected_left - radius_x, 0)
        search_top = max(expected_top - radius_y, 0)
        search_right = min(expected_left + template_w + radius_x, width)
        search_bottom = min(expected_top + template_h + radius_y, height)
        if search_right - search_left < template_w or search_bottom - search_top < template_h:
            return None
        return search_left, search_top, search_right, search_bottom

    def _track_boxes_between_frames(
        self,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
        *,
        frame_scale: float,
        global_offset_x: float,
        global_offset_y: float,
        global_confidence: float,
    ) -> list[OverlayBox]:
        tracked_boxes: list[OverlayBox] = []
        local_candidate_indices = self._local_tracking_candidate_indices(self._overlay_boxes)
        local_tracks: dict[int, _LocalTrack] = {}

        for index in local_candidate_indices:
            if index >= len(self._overlay_boxes):
                continue

            local_track = self._track_box_locally(
                self._overlay_boxes[index],
                previous_gray,
                current_gray,
                frame_scale,
            )
            if local_track is not None:
                local_tracks[index] = local_track

        consensus_offset = self._consensus_motion_offset(
            list(local_tracks.values()),
            global_offset_x=global_offset_x,
            global_offset_y=global_offset_y,
            global_confidence=global_confidence,
        )

        for index, box in enumerate(self._overlay_boxes):
            local_track = local_tracks.get(index)
            if local_track is not None:
                tracked_boxes.append(local_track.box)
                continue

            fallback = self._track_box_with_global_fallback(
                box,
                previous_gray,
                current_gray,
                frame_scale=frame_scale,
                global_offset_x=global_offset_x,
                global_offset_y=global_offset_y,
                global_confidence=global_confidence,
                consensus_offset=consensus_offset,
                allow_global_motion=index in local_candidate_indices,
            )
            if fallback is not None:
                tracked_boxes.append(fallback)

        tracked_boxes.sort(key=lambda item: (item.y, item.x))
        return tracked_boxes

    def _track_box_locally(
        self,
        box: OverlayBox,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
        frame_scale: float,
    ) -> _LocalTrack | None:
        template_rect = self._template_rect_for_box(box, previous_gray.shape, frame_scale)
        if template_rect is None:
            return None

        search_rect = self._search_rect_for_template(template_rect, box, current_gray.shape, frame_scale)
        if search_rect is None:
            return None

        match = self._match_template(previous_gray, current_gray, template_rect, search_rect)
        if match is None:
            return None

        offset_x, offset_y, confidence = match
        if confidence < self._local_tracking_min_confidence:
            return None

        tracked_offset_x = int(round(offset_x / frame_scale))
        tracked_offset_y = int(round(offset_y / frame_scale))
        tracked_box = self._offset_single_box(
            box,
            tracked_offset_x,
            tracked_offset_y,
            missing_frames=0,
        )
        if tracked_box is None:
            return None

        return _LocalTrack(
            box=tracked_box,
            offset_x=tracked_offset_x,
            offset_y=tracked_offset_y,
            confidence=confidence,
        )

    def _track_box_with_global_fallback(
        self,
        box: OverlayBox,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
        *,
        frame_scale: float,
        global_offset_x: float,
        global_offset_y: float,
        global_confidence: float,
        consensus_offset: tuple[int, int] | None,
        allow_global_motion: bool,
    ) -> OverlayBox | None:
        missing_frames = box.missing_frames + 1
        if missing_frames > self._max_realtime_lost_frames:
            return None

        same_position_score = self._same_position_match_score(box, previous_gray, current_gray, frame_scale)
        if same_position_score >= self._local_tracking_min_confidence:
            return self._offset_single_box(box, 0, 0, missing_frames=0)

        if consensus_offset is not None:
            consensus_x, consensus_y = consensus_offset
            return self._offset_single_box(
                box,
                consensus_x,
                consensus_y,
                missing_frames=missing_frames,
            )

        if allow_global_motion and global_confidence >= 0.28:
            tracked = self._offset_single_box(
                box,
                int(round(global_offset_x)),
                int(round(global_offset_y)),
                missing_frames=missing_frames,
            )
            if tracked is not None:
                return tracked

        return None

    def _consensus_motion_offset(
        self,
        local_tracks: list[_LocalTrack],
        *,
        global_offset_x: float,
        global_offset_y: float,
        global_confidence: float,
    ) -> tuple[int, int] | None:
        moving_tracks = [
            track
            for track in local_tracks
            if self._motion_magnitude(track.offset_x, track.offset_y) >= 3
        ]
        if not moving_tracks:
            global_x = int(round(global_offset_x))
            global_y = int(round(global_offset_y))
            if global_confidence >= 0.28 and self._motion_magnitude(global_x, global_y) >= 3:
                return global_x, global_y
            return None

        clusters: list[dict[str, float]] = []
        tolerance = 18.0
        for track in moving_tracks:
            weight = self._motion_vote_weight(track)
            best_cluster: dict[str, float] | None = None
            best_distance = float("inf")
            for cluster in clusters:
                mean_x = cluster["sum_x"] / max(cluster["weight"], 1e-6)
                mean_y = cluster["sum_y"] / max(cluster["weight"], 1e-6)
                distance = max(abs(track.offset_x - mean_x), abs(track.offset_y - mean_y))
                if distance <= tolerance and distance < best_distance:
                    best_cluster = cluster
                    best_distance = distance

            if best_cluster is None:
                clusters.append(
                    {
                        "sum_x": track.offset_x * weight,
                        "sum_y": track.offset_y * weight,
                        "weight": weight,
                        "count": 1.0,
                        "confidence": track.confidence * weight,
                    }
                )
                continue

            best_cluster["sum_x"] += track.offset_x * weight
            best_cluster["sum_y"] += track.offset_y * weight
            best_cluster["weight"] += weight
            best_cluster["count"] += 1.0
            best_cluster["confidence"] += track.confidence * weight

        if not clusters:
            return None

        global_motion_available = global_confidence >= 0.28 and self._motion_magnitude(
            int(round(global_offset_x)),
            int(round(global_offset_y)),
        ) >= 3

        def cluster_score(cluster: dict[str, float]) -> float:
            mean_x = cluster["sum_x"] / max(cluster["weight"], 1e-6)
            mean_y = cluster["sum_y"] / max(cluster["weight"], 1e-6)
            score = cluster["weight"] + (cluster["count"] * 0.65)
            if global_motion_available:
                distance = max(abs(mean_x - global_offset_x), abs(mean_y - global_offset_y))
                score += max(24.0 - distance, -24.0) * 0.08
            return score

        best = max(clusters, key=cluster_score)
        mean_confidence = best["confidence"] / max(best["weight"], 1e-6)
        if best["count"] < 2.0 and mean_confidence < 0.62 and not global_motion_available:
            return None

        consensus_x = int(round(best["sum_x"] / max(best["weight"], 1e-6)))
        consensus_y = int(round(best["sum_y"] / max(best["weight"], 1e-6)))
        if self._motion_magnitude(consensus_x, consensus_y) < 3:
            return None
        return consensus_x, consensus_y

    @staticmethod
    def _motion_magnitude(offset_x: float, offset_y: float) -> float:
        return max(abs(offset_x), abs(offset_y))

    @staticmethod
    def _motion_vote_weight(track: _LocalTrack) -> float:
        area = max(track.box.w * track.box.h, 1)
        area_weight = 1.0 + min(area / 9000.0, 4.0)
        confidence_weight = max(track.confidence, 0.05)
        translated_bonus = 1.3 if track.box.translated else 1.0
        return area_weight * confidence_weight * translated_bonus

    def _template_rect_for_box(
        self,
        box: OverlayBox,
        frame_shape: tuple[int, ...],
        frame_scale: float = 1.0,
    ) -> tuple[int, int, int, int] | None:
        height, width = frame_shape[:2]
        scaled_x = int(round(box.x * frame_scale))
        scaled_y = int(round(box.y * frame_scale))
        scaled_w = max(int(round(box.w * frame_scale)), 1)
        scaled_h = max(int(round(box.h * frame_scale)), 1)
        pad_x = max(min(scaled_w // 12, int(24 * frame_scale)), max(int(4 * frame_scale), 2))
        pad_y = max(min(scaled_h // 2, int(18 * frame_scale)), max(int(3 * frame_scale), 2))
        left = max(scaled_x - pad_x, 0)
        top = max(scaled_y - pad_y, 0)
        right = min(scaled_x + scaled_w + pad_x, width)
        bottom = min(scaled_y + scaled_h + pad_y, height)
        if right - left < 8 or bottom - top < 8:
            return None
        return left, top, right, bottom

    def _anchor_template_rect_for_box(
        self,
        box: OverlayBox,
        frame_shape: tuple[int, ...],
        frame_scale: float = 1.0,
    ) -> tuple[int, int, int, int] | None:
        height, width = frame_shape[:2]
        scaled_x = int(round(box.x * frame_scale))
        scaled_y = int(round(box.y * frame_scale))
        scaled_w = max(int(round(box.w * frame_scale)), 1)
        scaled_h = max(int(round(box.h * frame_scale)), 1)
        pad_x = max(int(scaled_w * 0.50), int(28 * frame_scale), 8)
        pad_y = max(int(scaled_h * 1.25), int(22 * frame_scale), 8)
        left = max(scaled_x - pad_x, 0)
        top = max(scaled_y - pad_y, 0)
        right = min(scaled_x + scaled_w + pad_x, width)
        bottom = min(scaled_y + scaled_h + pad_y, height)
        if right - left < 10 or bottom - top < 10:
            return None
        return left, top, right, bottom

    def _anchor_core_rect_for_box(
        self,
        box: OverlayBox,
        frame_shape: tuple[int, ...],
        frame_scale: float = 1.0,
    ) -> tuple[int, int, int, int] | None:
        height, width = frame_shape[:2]
        scaled_x = int(round(box.x * frame_scale))
        scaled_y = int(round(box.y * frame_scale))
        scaled_w = max(int(round(box.w * frame_scale)), 1)
        scaled_h = max(int(round(box.h * frame_scale)), 1)
        pad_x = max(min(scaled_w // 10, int(12 * frame_scale)), max(int(3 * frame_scale), 2))
        pad_y = max(min(scaled_h // 4, int(8 * frame_scale)), max(int(3 * frame_scale), 2))
        left = max(scaled_x - pad_x, 0)
        top = max(scaled_y - pad_y, 0)
        right = min(scaled_x + scaled_w + pad_x, width)
        bottom = min(scaled_y + scaled_h + pad_y, height)
        if right - left < 8 or bottom - top < 8:
            return None
        return left, top, right, bottom

    def _search_rect_for_template(
        self,
        template_rect: tuple[int, int, int, int],
        box: OverlayBox,
        frame_shape: tuple[int, ...],
        frame_scale: float = 1.0,
    ) -> tuple[int, int, int, int] | None:
        height, width = frame_shape[:2]
        left, top, right, bottom = template_rect
        scaled_w = max(int(round(box.w * frame_scale)), 1)
        scaled_h = max(int(round(box.h * frame_scale)), 1)
        radius_x = max(min(max(scaled_w // 3, scaled_h * 5), int(220 * frame_scale)), max(int(32 * frame_scale), 16))
        radius_y = max(min(max(scaled_h * 9, scaled_w // 5), int(260 * frame_scale)), max(int(48 * frame_scale), 20))
        search_left = max(left - radius_x, 0)
        search_top = max(top - radius_y, 0)
        search_right = min(right + radius_x, width)
        search_bottom = min(bottom + radius_y, height)
        if search_right - search_left < right - left or search_bottom - search_top < bottom - top:
            return None
        return search_left, search_top, search_right, search_bottom

    @staticmethod
    def _match_template(
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
        template_rect: tuple[int, int, int, int],
        search_rect: tuple[int, int, int, int],
    ) -> tuple[float, float, float] | None:
        left, top, right, bottom = template_rect
        search_left, search_top, search_right, search_bottom = search_rect
        template = previous_gray[top:bottom, left:right]
        search = current_gray[search_top:search_bottom, search_left:search_right]
        if template.size == 0 or search.size == 0:
            return None
        if template.shape[0] > search.shape[0] or template.shape[1] > search.shape[1]:
            return None
        if float(np.std(template)) < 3.0:
            return None

        scale = 1.0
        template_area = template.shape[0] * template.shape[1]
        search_area = search.shape[0] * search.shape[1]
        if template_area > 80_000 or search_area > 300_000:
            scale = 0.5
            template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            search = cv2.resize(search, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if template.shape[0] < 4 or template.shape[1] < 4:
                return None

        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _min_value, max_value, _min_location, max_location = cv2.minMaxLoc(result)
        match_left = search_left + (max_location[0] / scale)
        match_top = search_top + (max_location[1] / scale)
        return match_left - left, match_top - top, float(max_value)

    def _same_position_match_score(
        self,
        box: OverlayBox,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
        frame_scale: float = 1.0,
    ) -> float:
        template_rect = self._template_rect_for_box(box, previous_gray.shape, frame_scale)
        if template_rect is None:
            return 0.0

        left, top, right, bottom = template_rect
        if right > current_gray.shape[1] or bottom > current_gray.shape[0]:
            return 0.0

        match = self._match_template(previous_gray, current_gray, template_rect, template_rect)
        if match is None:
            return 0.0
        _offset_x, _offset_y, confidence = match
        return confidence

    def _capture_visual_anchors(self, observations: list[OverlayBox]) -> None:
        if self._tracking_gray_frame is None or not observations:
            return

        used_anchor_indices: set[int] = set()
        for box in sorted(observations, key=self._overlay_box_priority, reverse=True)[: self._max_anchor_tracked_boxes]:
            anchor = self._new_visual_anchor(box, self._tracking_gray_frame, self._tracking_frame_scale)
            if anchor is None:
                continue

            match_index = self._find_visual_anchor_index(box, used_anchor_indices)
            if match_index is None:
                self._visual_anchors.append(anchor)
                used_anchor_indices.add(len(self._visual_anchors) - 1)
                continue

            self._visual_anchors[match_index] = anchor
            used_anchor_indices.add(match_index)

        self._prune_visual_anchors()

    def _new_visual_anchor(
        self,
        box: OverlayBox,
        gray_frame: np.ndarray,
        frame_scale: float,
    ) -> _VisualAnchor | None:
        template_rect = self._anchor_template_rect_for_box(box, gray_frame.shape, frame_scale)
        core_rect = self._anchor_core_rect_for_box(box, gray_frame.shape, frame_scale)
        if template_rect is None or core_rect is None:
            return None

        left, top, right, bottom = template_rect
        template = self._anchor_feature_image(gray_frame[top:bottom, left:right])
        if template.size == 0 or float(np.std(template)) < 3.0:
            return None

        core_left, core_top, core_right, core_bottom = core_rect
        core_raw_template = gray_frame[core_top:core_bottom, core_left:core_right]
        core_template = self._anchor_feature_image(core_raw_template)
        if core_template.size == 0 or float(np.std(core_template)) < 3.0:
            return None

        scaled_x = int(round(box.x * frame_scale))
        scaled_y = int(round(box.y * frame_scale))
        return _VisualAnchor(
            key=self._anchor_key(box),
            template=template.copy(),
            core_template=core_template.copy(),
            core_raw_template=core_raw_template.copy(),
            box_offset_x=scaled_x - left,
            box_offset_y=scaled_y - top,
            core_offset_x=core_left - scaled_x,
            core_offset_y=core_top - scaled_y,
            last_box=box,
        )

    def _refresh_visual_anchor_template(
        self,
        anchor: _VisualAnchor,
        box: OverlayBox,
        gray_frame: np.ndarray,
        frame_scale: float,
        confidence: float,
    ) -> None:
        if confidence < 0.72:
            return

        refreshed = self._new_visual_anchor(box, gray_frame, frame_scale)
        if refreshed is None:
            return

        anchor.key = refreshed.key
        anchor.template = refreshed.template
        anchor.core_template = refreshed.core_template
        anchor.core_raw_template = refreshed.core_raw_template
        anchor.box_offset_x = refreshed.box_offset_x
        anchor.box_offset_y = refreshed.box_offset_y
        anchor.core_offset_x = refreshed.core_offset_x
        anchor.core_offset_y = refreshed.core_offset_y

    def _find_visual_anchor_index(
        self,
        box: OverlayBox,
        used_indices: set[int],
    ) -> int | None:
        key = self._anchor_key(box)
        best_index: int | None = None
        best_score = 0.0
        for index, anchor in enumerate(self._visual_anchors):
            if index in used_indices:
                continue
            if anchor.key != key:
                continue

            iou = self._box_iou(box, anchor.last_box)
            proximity = self._box_center_proximity(box, anchor.last_box)
            score = (iou * 0.70) + (proximity * 0.30) - (anchor.missing_frames * 0.20)
            if score > best_score:
                best_index = index
                best_score = score

        return best_index

    def _prune_visual_anchors(self) -> None:
        self._visual_anchors.sort(
            key=lambda anchor: self._overlay_box_priority(anchor.last_box) - (anchor.missing_frames * 80.0),
            reverse=True,
        )
        self._visual_anchors = self._visual_anchors[: self._max_visible_overlay_boxes]

    @staticmethod
    def _match_template_image(
        template: np.ndarray,
        current_gray: np.ndarray,
        search_rect: tuple[int, int, int, int],
    ) -> tuple[float, float, float, float] | None:
        search_left, search_top, search_right, search_bottom = search_rect
        search = TranslationOverlay._anchor_feature_image(
            current_gray[search_top:search_bottom, search_left:search_right]
        )
        if template.size == 0 or search.size == 0:
            return None
        if template.shape[0] > search.shape[0] or template.shape[1] > search.shape[1]:
            return None
        if float(np.std(template)) < 3.0:
            return None

        scale = 1.0
        template_area = template.shape[0] * template.shape[1]
        search_area = search.shape[0] * search.shape[1]
        if template_area > 70_000 or search_area > 360_000:
            scale = 0.5
            template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            search = cv2.resize(search, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if template.shape[0] < 5 or template.shape[1] < 5:
                return None

        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _min_value, max_value, _min_location, max_location = cv2.minMaxLoc(result)
        suppressed = result.copy()
        suppression_left = max(max_location[0] - max(template.shape[1] // 3, 4), 0)
        suppression_top = max(max_location[1] - max(template.shape[0] // 3, 4), 0)
        suppression_right = min(max_location[0] + max(template.shape[1] // 3, 4) + 1, suppressed.shape[1])
        suppression_bottom = min(max_location[1] + max(template.shape[0] // 3, 4) + 1, suppressed.shape[0])
        suppressed[suppression_top:suppression_bottom, suppression_left:suppression_right] = -1.0
        second_value = -1.0
        if np.any(suppressed > -1.0):
            _second_min, second_value, _second_min_location, _second_location = cv2.minMaxLoc(suppressed)

        match_left = search_left + (max_location[0] / scale)
        match_top = search_top + (max_location[1] / scale)
        return match_left, match_top, float(max_value), float(max_value - second_value)

    @staticmethod
    def _anchor_feature_image(gray: np.ndarray) -> np.ndarray:
        if gray.size == 0:
            return gray
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        gradient = cv2.morphologyEx(
            blurred,
            cv2.MORPH_GRADIENT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )
        return cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    @staticmethod
    def _visual_anchor_core_match_score(
        anchor: _VisualAnchor,
        box: OverlayBox,
        current_gray: np.ndarray,
        frame_scale: float,
    ) -> float:
        scaled_x = int(round(box.x * frame_scale))
        scaled_y = int(round(box.y * frame_scale))
        left = scaled_x + anchor.core_offset_x
        top = scaled_y + anchor.core_offset_y
        height, width = current_gray.shape[:2]
        core_h, core_w = anchor.core_template.shape[:2]
        right = left + core_w
        bottom = top + core_h
        if left < 0 or top < 0 or right > width or bottom > height:
            return 0.0

        current_core_raw = current_gray[top:bottom, left:right]
        current_core = TranslationOverlay._anchor_feature_image(current_core_raw)
        if current_core.shape != anchor.core_template.shape:
            return 0.0
        if float(np.std(current_core)) < 3.0 or float(np.std(anchor.core_template)) < 3.0:
            return 0.0

        feature_result = cv2.matchTemplate(current_core, anchor.core_template, cv2.TM_CCOEFF_NORMED)
        _feature_min, feature_value, _feature_min_location, _feature_location = cv2.minMaxLoc(feature_result)

        if current_core_raw.shape != anchor.core_raw_template.shape:
            return 0.0
        if float(np.std(current_core_raw)) < 3.0 or float(np.std(anchor.core_raw_template)) < 3.0:
            return 0.0

        raw_result = cv2.matchTemplate(current_core_raw, anchor.core_raw_template, cv2.TM_CCOEFF_NORMED)
        _raw_min, raw_value, _raw_min_location, _raw_location = cv2.minMaxLoc(raw_result)
        return min(float(feature_value), float(raw_value))

    @staticmethod
    def _anchor_key(box: OverlayBox) -> str:
        return " ".join(box.text.casefold().split())

    @staticmethod
    def _box_iou(first: OverlayBox, second: OverlayBox) -> float:
        left = max(first.x, second.x)
        top = max(first.y, second.y)
        right = min(first.x + first.w, second.x + second.w)
        bottom = min(first.y + first.h, second.y + second.h)
        intersection = max(right - left, 0) * max(bottom - top, 0)
        if intersection <= 0:
            return 0.0
        first_area = max(first.w * first.h, 1)
        second_area = max(second.w * second.h, 1)
        return intersection / max(first_area + second_area - intersection, 1)

    @staticmethod
    def _box_center_proximity(first: OverlayBox, second: OverlayBox) -> float:
        first_center_x = first.x + (first.w / 2.0)
        first_center_y = first.y + (first.h / 2.0)
        second_center_x = second.x + (second.w / 2.0)
        second_center_y = second.y + (second.h / 2.0)
        distance = max(abs(first_center_x - second_center_x), abs(first_center_y - second_center_y))
        tolerance = max(first.w, first.h, second.w, second.h, 1) * 10.0
        return max(1.0 - (distance / tolerance), 0.0)

    def _local_tracking_candidate_indices(self, boxes: list[OverlayBox]) -> set[int]:
        ranked: list[tuple[float, int]] = []
        for index, box in enumerate(boxes):
            area = box.w * box.h
            if area <= 0:
                continue
            if area > 240_000:
                continue
            if box.h > 260:
                continue
            priority = (box.missing_frames * -1000.0) + min(area / 5000.0, 60.0) + min(box.w / 30.0, 40.0)
            ranked.append((priority, index))

        ranked.sort(reverse=True)
        return {index for _priority, index in ranked[: self._max_local_tracked_boxes]}

    def _limit_overlay_boxes(self, boxes: list[OverlayBox]) -> list[OverlayBox]:
        ranked = sorted(
            boxes,
            key=self._overlay_box_priority,
            reverse=True,
        )
        kept = ranked[: self._max_visible_overlay_boxes]
        kept.sort(key=lambda item: (item.y, item.x))
        return kept

    @staticmethod
    def _overlay_box_priority(box: OverlayBox) -> float:
        text = " ".join(box.text.split())
        area = box.w * box.h
        score = min(len(text) * 2.0, 80.0) + min(area / 4500.0, 45.0) + min(box.w / 35.0, 35.0)
        if box.translated:
            score += 80.0
        if box.missing_frames:
            score -= box.missing_frames * 35.0
        if len(text) <= 2:
            score -= 45.0
        return score

    def _is_probable_scene_change(
        self,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
        *,
        global_confidence: float,
    ) -> bool:
        if global_confidence >= 0.16:
            return False

        delta = cv2.absdiff(previous_gray, current_gray)
        if delta.size == 0:
            return False
        changed_ratio = cv2.countNonZero(cv2.threshold(delta, 36, 255, cv2.THRESH_BINARY)[1]) / max(delta.size, 1)
        return changed_ratio >= self._scene_change_ratio_threshold

    def _offset_single_box(
        self,
        box: OverlayBox,
        offset_x: int,
        offset_y: int,
        *,
        missing_frames: int,
    ) -> OverlayBox | None:
        width = self._monitor.width if self._monitor is not None else self.width()
        height = self._monitor.height if self._monitor is not None else self.height()
        tracked = OverlayBox(
            x=box.x + offset_x,
            y=box.y + offset_y,
            w=box.w,
            h=box.h,
            text=box.text,
            missing_frames=missing_frames,
            translated=box.translated,
        )
        if tracked.x + tracked.w <= 0 or tracked.y + tracked.h <= 0:
            return None
        if tracked.x >= width or tracked.y >= height:
            return None
        return tracked

    def _place_single_box(
        self,
        box: OverlayBox,
        x: int,
        y: int,
        *,
        missing_frames: int,
    ) -> OverlayBox | None:
        width = self._monitor.width if self._monitor is not None else self.width()
        height = self._monitor.height if self._monitor is not None else self.height()
        tracked = OverlayBox(
            x=x,
            y=y,
            w=box.w,
            h=box.h,
            text=box.text,
            missing_frames=missing_frames,
            translated=box.translated,
        )
        if tracked.x + tracked.w <= 0 or tracked.y + tracked.h <= 0:
            return None
        if tracked.x >= width or tracked.y >= height:
            return None
        return tracked

    def update_analysis(self, analysis: FrameAnalysis) -> None:
        current_boxes = [
            OverlayBox(
                box.x,
                box.y,
                box.w,
                box.h,
                text,
                translated=bool(" ".join(box.translated_text.split())),
            )
            for box in analysis.boxes
            if (text := overlay_text_for_box(box))
        ]

        if self._tracking_enabled:
            self._track_manager.max_visible_tracks = self._max_visible_overlay_boxes
            self._track_manager.max_predicted_frames = self._max_visible_predicted_frames
            self._overlay_boxes = self._track_manager.update_from_pipeline(
                current_boxes,
                self._predict_tracked_boxes(analysis),
            )
            if self._tracking_mode == "anchor" and self._overlay_boxes:
                self._capture_visual_anchors(self._overlay_boxes)
        elif current_boxes:
            self._track_manager.max_visible_tracks = self._max_visible_overlay_boxes
            self._overlay_boxes = self._track_manager.replace_with_observations(current_boxes)
        else:
            self._track_manager.clear()
            self._overlay_boxes = []
        self.update()

    def _predict_tracked_boxes(self, analysis: FrameAnalysis) -> list[OverlayBox]:
        if not self._overlay_boxes:
            return []
        if self._realtime_tracking_active:
            return self._offset_overlay_boxes(self._overlay_boxes, 0, 0, increment_missing=True)
        if analysis.content_motion_confidence < 0.08:
            return []

        offset_x = int(round(analysis.content_offset_x))
        offset_y = int(round(analysis.content_offset_y))
        if self._motion_magnitude(offset_x, offset_y) < 3:
            return []
        return self._offset_overlay_boxes(self._overlay_boxes, offset_x, offset_y, increment_missing=True)

    def _offset_overlay_boxes(
        self,
        boxes: list[OverlayBox],
        offset_x: int,
        offset_y: int,
        *,
        increment_missing: bool,
    ) -> list[OverlayBox]:
        width = self._monitor.width if self._monitor is not None else self.width()
        height = self._monitor.height if self._monitor is not None else self.height()
        next_boxes: list[OverlayBox] = []
        for box in boxes:
            missing_frames = box.missing_frames + 1 if increment_missing else box.missing_frames
            if missing_frames > self._max_tracked_missing_frames:
                continue

            tracked = OverlayBox(
                x=box.x + offset_x,
                y=box.y + offset_y,
                w=box.w,
                h=box.h,
                text=box.text,
                missing_frames=missing_frames,
                translated=box.translated,
            )
            if tracked.x + tracked.w <= 0 or tracked.y + tracked.h <= 0:
                continue
            if tracked.x >= width or tracked.y >= height:
                continue
            next_boxes.append(tracked)
        return next_boxes

    def paintEvent(self, _event: object) -> None:
        if not self._overlay_boxes or self._monitor is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        frame_width = max(self._monitor.width, 1)
        frame_height = max(self._monitor.height, 1)

        for box in self._overlay_boxes:
            scaled = scale_overlay_rect(
                DetectionBox(x=box.x, y=box.y, w=box.w, h=box.h, text=box.text),
                overlay_width=max(self.width(), 1),
                overlay_height=max(self.height(), 1),
                frame_width=frame_width,
                frame_height=frame_height,
            )
            anchor_rect = QRect(*scaled)
            bubble_rect = self._expanded_bubble_rect(
                anchor_rect,
                box.text,
                bounds_width=max(self.width(), 1),
                bounds_height=max(self.height(), 1),
            )
            self._paint_box(painter, bubble_rect, box.text, anchor_height=anchor_rect.height())

    def _paint_box(self, painter: QPainter, rect: QRect, text: str, *, anchor_height: int | None = None) -> None:
        accent = QColor(48, 231, 149, 220)
        background = QColor(15, 23, 42, 212)
        text_color = QColor(248, 250, 252)
        font_anchor_height = rect.height() if anchor_height is None else anchor_height

        bubble_rect = rect.adjusted(0, 0, -1, -1)
        radius = max(min(rect.height() // 4, 8), 2)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(background)
        painter.drawRoundedRect(bubble_rect, radius, radius)

        painter.setPen(QPen(accent, 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(bubble_rect, radius, radius)

        horizontal_padding = max(min(rect.height() // 4, 12), 2)
        vertical_padding = max(min(rect.height() // 8, 6), 1)
        text_rect = bubble_rect.adjusted(
            horizontal_padding,
            vertical_padding,
            -horizontal_padding,
            -vertical_padding,
        )
        if text_rect.width() <= 0 or text_rect.height() <= 0:
            text_rect = bubble_rect

        font = self._font_for_text(text, text_rect, overlay_font_pixel_size(font_anchor_height))
        painter.setFont(font)
        painter.setPen(text_color)
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap,
            text,
        )

    @staticmethod
    def _expanded_bubble_rect(
        anchor_rect: QRect,
        text: str,
        *,
        bounds_width: int,
        bounds_height: int,
    ) -> QRect:
        normalized = " ".join(text.split())
        if not normalized or anchor_rect.width() <= 0 or anchor_rect.height() <= 0:
            return anchor_rect

        horizontal_padding = max(min(anchor_rect.height() // 4, 12), 2)
        vertical_padding = max(min(anchor_rect.height() // 8, 6), 1)
        base_pixel_size = overlay_font_pixel_size(anchor_rect.height())
        if len(normalized) >= 18:
            base_pixel_size = max(base_pixel_size, 10)

        font = QFont("Segoe UI")
        font.setWeight(QFont.Weight.DemiBold)
        font.setPixelSize(max(base_pixel_size, 1))
        metrics = QFontMetrics(font)
        flags = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap

        single_line_width = metrics.horizontalAdvance(normalized)
        max_bubble_width = min(max(bounds_width - 8, anchor_rect.width()), max(260, int(bounds_width * 0.45)))
        target_bubble_width = max(anchor_rect.width(), single_line_width + (horizontal_padding * 2) + 4)
        if target_bubble_width > max_bubble_width:
            target_bubble_width = max_bubble_width
        if single_line_width > anchor_rect.width():
            target_bubble_width = max(target_bubble_width, min(max_bubble_width, max(anchor_rect.width() * 2, 180)))

        target_text_width = max(target_bubble_width - (horizontal_padding * 2), 1)
        text_bounds = metrics.boundingRect(QRect(0, 0, target_text_width, 10_000), flags, normalized)
        target_bubble_height = max(
            anchor_rect.height(),
            text_bounds.height() + (vertical_padding * 2) + 4,
        )
        max_bubble_height = max(anchor_rect.height(), int(bounds_height * 0.28))
        target_bubble_height = min(target_bubble_height, max_bubble_height)

        left = anchor_rect.x()
        top = anchor_rect.y()
        if left + target_bubble_width > bounds_width:
            left = max(bounds_width - target_bubble_width - 4, 0)
        if top + target_bubble_height > bounds_height:
            top = max(bounds_height - target_bubble_height - 4, 0)

        return QRect(left, top, target_bubble_width, target_bubble_height)

    @staticmethod
    def _font_for_text(text: str, rect: QRect, max_pixel_size: int) -> QFont:
        flags = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap
        font = QFont("Segoe UI")
        font.setWeight(QFont.Weight.DemiBold)
        for pixel_size in range(max(max_pixel_size, 1), 0, -1):
            font.setPixelSize(pixel_size)
            bounds = QFontMetrics(font).boundingRect(rect, flags, text)
            if bounds.height() <= rect.height() and bounds.width() <= rect.width():
                return font
        font.setPixelSize(1)
        return font

    def _apply_monitor_geometry(self, monitor: MonitorSpec) -> None:
        app = QApplication.instance()
        if app is not None:
            screens = app.screens()
            screen_index = monitor.index - 1
            if 0 <= screen_index < len(screens):
                self.setGeometry(screens[screen_index].geometry())
                return

        self.setGeometry(monitor.left, monitor.top, monitor.width, monitor.height)

    def _ensure_capture_exclusion(self) -> None:
        if self._capture_exclusion_applied:
            return

        self._capture_exclusion_applied = set_window_capture_exclusion(int(self.winId()))
