from __future__ import annotations

from dataclasses import dataclass
from math import hypot


@dataclass(slots=True, frozen=True)
class OverlayBox:
    x: int
    y: int
    w: int
    h: int
    text: str
    missing_frames: int = 0
    translated: bool = False


@dataclass(slots=True)
class OverlayTrack:
    track_id: int
    box: OverlayBox
    state: str = "active"
    hidden_frames: int = 0
    age_frames: int = 1

    @property
    def visible(self) -> bool:
        return self.state in {"active", "predicted"}


class OverlayTrackManager:
    def __init__(
        self,
        *,
        max_visible_tracks: int = 24,
        max_predicted_frames: int = 3,
        max_memory_frames: int = 90,
    ) -> None:
        self.max_visible_tracks = max_visible_tracks
        self.max_predicted_frames = max_predicted_frames
        self.max_memory_frames = max_memory_frames
        self._tracks: list[OverlayTrack] = []
        self._next_track_id = 1

    def clear(self) -> None:
        self._tracks = []
        self._next_track_id = 1

    def replace_with_observations(self, observations: list[OverlayBox]) -> list[OverlayBox]:
        self.clear()
        for box in observations:
            self._tracks.append(self._new_track(self._as_visible(box, missing_frames=0), state="active"))
        return self.visible_boxes()

    def update_from_pipeline(
        self,
        observations: list[OverlayBox],
        predicted_boxes: list[OverlayBox],
    ) -> list[OverlayBox]:
        used_track_ids: set[int] = set()

        predicted_matches: dict[int, OverlayBox] = {}
        predicted_used_ids: set[int] = set()
        for predicted in predicted_boxes:
            if predicted.missing_frames > self.max_predicted_frames:
                continue
            match = self._find_best_track(predicted, predicted_used_ids, allow_hidden=False)
            if match is not None:
                predicted_matches[match.track_id] = predicted
                predicted_used_ids.add(match.track_id)

        for observation in observations:
            match = self._find_best_track(observation, used_track_ids, allow_hidden=True)
            if match is None:
                visible_observation = self._as_visible(observation, missing_frames=0)
                self._tracks.append(self._new_track(visible_observation, state="active"))
                used_track_ids.add(self._tracks[-1].track_id)
                continue

            predicted = predicted_matches.get(match.track_id)
            if predicted is not None:
                merged_box = OverlayBox(
                    x=predicted.x,
                    y=predicted.y,
                    w=predicted.w,
                    h=predicted.h,
                    text=observation.text,
                    missing_frames=0,
                    translated=observation.translated,
                )
                self._refresh_track(match, merged_box, state="active")
            else:
                visible_observation = self._as_visible(observation, missing_frames=0)
                self._refresh_track(match, visible_observation, state="active")

            used_track_ids.add(match.track_id)

        for predicted in predicted_boxes:
            if predicted.missing_frames > self.max_predicted_frames:
                continue

            match = self._find_best_track(predicted, used_track_ids, allow_hidden=False)
            if match is None:
                continue

            self._refresh_track(match, predicted, state="predicted")
            used_track_ids.add(match.track_id)

        self._hide_unused_tracks(used_track_ids)
        self._prune_tracks()
        return self.visible_boxes()

    def update_from_visual_tracking(self, tracked_boxes: list[OverlayBox]) -> list[OverlayBox]:
        used_track_ids: set[int] = set()

        for box in tracked_boxes:
            if box.missing_frames > self.max_predicted_frames:
                continue

            match = self._find_best_track(box, used_track_ids, allow_hidden=False)
            state = "active" if box.missing_frames == 0 else "predicted"
            if match is None:
                self._tracks.append(self._new_track(box, state=state))
                used_track_ids.add(self._tracks[-1].track_id)
                continue

            self._refresh_track(match, box, state=state)
            used_track_ids.add(match.track_id)

        self._hide_unused_tracks(used_track_ids)
        self._prune_tracks()
        return self.visible_boxes()

    def mark_all_occluded(self) -> list[OverlayBox]:
        self._hide_unused_tracks(set())
        self._prune_tracks()
        return self.visible_boxes()

    def visible_boxes(self) -> list[OverlayBox]:
        visible = [track.box for track in self._tracks if track.visible]
        visible.sort(key=self._overlay_box_priority, reverse=True)
        kept = visible[: self.max_visible_tracks]
        kept.sort(key=lambda item: (item.y, item.x))
        return kept

    def _new_track(self, box: OverlayBox, *, state: str) -> OverlayTrack:
        track = OverlayTrack(track_id=self._next_track_id, box=box, state=state)
        self._next_track_id += 1
        return track

    @staticmethod
    def _refresh_track(track: OverlayTrack, box: OverlayBox, *, state: str) -> None:
        track.box = box
        track.state = state
        track.hidden_frames = 0
        track.age_frames += 1

    def _hide_unused_tracks(self, used_track_ids: set[int]) -> None:
        for track in self._tracks:
            if track.track_id in used_track_ids:
                continue

            track.hidden_frames += 1
            track.age_frames += 1
            track.state = "occluded"
            track.box = self._as_visible(
                track.box,
                missing_frames=min(track.box.missing_frames + 1, self.max_memory_frames),
            )

    def _prune_tracks(self) -> None:
        self._tracks = [
            track
            for track in self._tracks
            if track.visible or track.hidden_frames <= self.max_memory_frames
        ]

    def _find_best_track(
        self,
        box: OverlayBox,
        used_track_ids: set[int],
        *,
        allow_hidden: bool,
    ) -> OverlayTrack | None:
        best_track: OverlayTrack | None = None
        best_score = 0.0

        for track in self._tracks:
            if track.track_id in used_track_ids:
                continue
            if not allow_hidden and not track.visible:
                continue

            score = self._association_score(box, track)
            if score > best_score:
                best_score = score
                best_track = track

        if best_track is None or best_score < 0.46:
            return None
        return best_track

    @classmethod
    def _association_score(cls, box: OverlayBox, track: OverlayTrack) -> float:
        text_score = cls._text_similarity(box.text, track.box.text)
        iou = cls._intersection_over_union(cls._rect(box), cls._rect(track.box))
        proximity = cls._center_proximity(box, track.box)
        score = (text_score * 0.55) + (iou * 0.30) + (proximity * 0.15)

        same_text = text_score >= 0.90
        if same_text and track.state != "active":
            score += 0.20
        elif same_text and (iou > 0.0 or proximity >= 0.28):
            score += 0.12

        if text_score < 0.34 and iou < 0.40:
            score -= 0.35

        return score

    @staticmethod
    def _as_visible(box: OverlayBox, *, missing_frames: int) -> OverlayBox:
        return OverlayBox(
            x=box.x,
            y=box.y,
            w=box.w,
            h=box.h,
            text=box.text,
            missing_frames=missing_frames,
            translated=box.translated,
        )

    @staticmethod
    def _rect(box: OverlayBox) -> tuple[int, int, int, int]:
        return box.x, box.y, box.w, box.h

    @staticmethod
    def _center_proximity(first: OverlayBox, second: OverlayBox) -> float:
        first_center_x = first.x + (first.w / 2.0)
        first_center_y = first.y + (first.h / 2.0)
        second_center_x = second.x + (second.w / 2.0)
        second_center_y = second.y + (second.h / 2.0)
        distance = hypot(first_center_x - second_center_x, first_center_y - second_center_y)
        tolerance = max(first.w, first.h, second.w, second.h, 1) * 6.0
        return max(1.0 - (distance / max(tolerance, 1.0)), 0.0)

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
