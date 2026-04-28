from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from .models import DetectionBox, FrameAnalysis, MonitorSpec
from .windows_capture_exclusion import set_window_capture_exclusion


@dataclass(slots=True, frozen=True)
class OverlayBox:
    x: int
    y: int
    w: int
    h: int
    text: str
    missing_frames: int = 0
    translated: bool = False


@dataclass(slots=True, frozen=True)
class _LocalTrack:
    box: OverlayBox
    offset_x: int
    offset_y: int
    confidence: float


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
        self._realtime_tracking_active = False
        self._tracking_lost_frames = 0
        self._tracking_gray_frame: np.ndarray | None = None
        self._max_tracked_missing_frames = 10
        self._max_realtime_lost_frames = 6
        self._max_local_tracked_boxes = 12
        self._max_visible_overlay_boxes = 24
        self._local_tracking_min_confidence = 0.42
        self._scene_change_ratio_threshold = 0.42

    def show_for_monitor(self, monitor: MonitorSpec) -> None:
        self._monitor = monitor
        self._apply_monitor_geometry(monitor)
        self.show()
        self._ensure_capture_exclusion()
        self.raise_()

    def clear_analysis(self) -> None:
        self._overlay_boxes = []
        self.update()

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
        self._tracking_gray_frame = gray.copy()
        if not self._tracking_enabled or not self._realtime_tracking_active or not self._overlay_boxes:
            return
        if previous_gray is None or previous_gray.shape != gray.shape:
            return

        frame_scale = max(float(getattr(tracking_frame, "frame_scale", 1.0)), 1e-6)
        global_confidence = float(getattr(tracking_frame, "global_confidence", 0.0))
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

            self._overlay_boxes = self._limit_overlay_boxes(tracked_boxes)
            self.update()
            return

        if probable_scene_change:
            self.clear_analysis()
            return

        self._tracking_lost_frames += 1
        if self._tracking_lost_frames >= self._max_realtime_lost_frames:
            self.clear_analysis()

    def apply_tracking_offset(self, offset_x: float, offset_y: float, confidence: float) -> None:
        if not self._tracking_enabled or not self._realtime_tracking_active or not self._overlay_boxes:
            return

        if confidence < 0.10:
            self._tracking_lost_frames += 1
            if self._tracking_lost_frames >= self._max_realtime_lost_frames:
                self.clear_analysis()
            return

        self._tracking_lost_frames = 0
        self._overlay_boxes = self._limit_overlay_boxes(
            self._offset_overlay_boxes(
                self._overlay_boxes,
                int(round(offset_x)),
                int(round(offset_y)),
                increment_missing=False,
            )
        )
        self.update()

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

        return self._offset_single_box(box, 0, 0, missing_frames=missing_frames)

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
            self._overlay_boxes = self._limit_overlay_boxes(self._merge_tracked_boxes(current_boxes, analysis))
        elif current_boxes:
            self._overlay_boxes = self._limit_overlay_boxes(current_boxes)
        else:
            self._overlay_boxes = []
        self.update()

    def _merge_tracked_boxes(
        self,
        current_boxes: list[OverlayBox],
        analysis: FrameAnalysis,
    ) -> list[OverlayBox]:
        predicted_boxes = self._predict_tracked_boxes(analysis)
        if not predicted_boxes:
            return current_boxes
        if not current_boxes:
            return predicted_boxes

        merged: list[OverlayBox] = list(current_boxes)
        matched_predicted_indices: set[int] = set()
        for current_box in current_boxes:
            match_index = self._find_matching_overlay_box(current_box, predicted_boxes, matched_predicted_indices)
            if match_index is not None:
                matched_predicted_indices.add(match_index)

        for index, predicted_box in enumerate(predicted_boxes):
            if index in matched_predicted_indices:
                continue
            if self._overlaps_any(predicted_box, current_boxes):
                continue
            merged.append(predicted_box)

        merged.sort(key=lambda box: (box.y, box.x))
        return merged

    def _predict_tracked_boxes(self, analysis: FrameAnalysis) -> list[OverlayBox]:
        if not self._overlay_boxes:
            return []
        if self._realtime_tracking_active:
            return self._offset_overlay_boxes(self._overlay_boxes, 0, 0, increment_missing=True)
        if analysis.content_motion_confidence < 0.08:
            return []

        offset_x = int(round(analysis.content_offset_x))
        offset_y = int(round(analysis.content_offset_y))
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

    def _find_matching_overlay_box(
        self,
        current_box: OverlayBox,
        predicted_boxes: list[OverlayBox],
        used_indices: set[int],
    ) -> int | None:
        best_index: int | None = None
        best_score = 0.0
        current_rect = (current_box.x, current_box.y, current_box.w, current_box.h)

        for index, predicted_box in enumerate(predicted_boxes):
            if index in used_indices:
                continue

            predicted_rect = (predicted_box.x, predicted_box.y, predicted_box.w, predicted_box.h)
            iou = self._intersection_over_union(current_rect, predicted_rect)
            if iou < 0.25:
                continue

            text_score = self._text_similarity(current_box.text, predicted_box.text)
            score = (iou * 0.75) + (text_score * 0.25)
            if score > best_score:
                best_score = score
                best_index = index

        return best_index

    @classmethod
    def _overlaps_any(cls, box: OverlayBox, current_boxes: list[OverlayBox]) -> bool:
        rect = (box.x, box.y, box.w, box.h)
        return any(
            cls._intersection_over_union(rect, (current.x, current.y, current.w, current.h)) >= 0.35
            for current in current_boxes
        )

    @staticmethod
    def _intersection_over_union(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[0] + first[2], second[0] + second[2])
        bottom = min(first[1] + first[3], second[1] + second[3])
        intersection = max(right - left, 0) * max(bottom - top, 0)
        if intersection <= 0:
            return 0.0

        first_area = max(first[2] * first[3], 1)
        second_area = max(second[2] * second[3], 1)
        return intersection / max(first_area + second_area - intersection, 1)

    @staticmethod
    def _text_similarity(first: str, second: str) -> float:
        first_tokens = set(first.casefold().split())
        second_tokens = set(second.casefold().split())
        if not first_tokens or not second_tokens:
            return 0.0
        return len(first_tokens & second_tokens) / len(first_tokens | second_tokens)

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
            self._paint_box(painter, QRect(*scaled), box.text)

    def _paint_box(self, painter: QPainter, rect: QRect, text: str) -> None:
        accent = QColor(48, 231, 149, 220)
        background = QColor(15, 23, 42, 212)
        text_color = QColor(248, 250, 252)

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

        font = self._font_for_text(text, text_rect, overlay_font_pixel_size(rect.height()))
        painter.setFont(font)
        painter.setPen(text_color)
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap,
            text,
        )

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
