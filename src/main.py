import argparse
from collections import deque
from dataclasses import dataclass

import cv2
from ultralytics import YOLO

LINE_OFFSET_X = 15

model_YOLO = "./best-yolo11n_1.pt" # путь к модели

@dataclass
class TouchEvent:
    frame_idx: int
    side: str
    track_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        
    )
    
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Путь к видео файлу",
    )
    parser.add_argument(
        "--window-frames",
        type=int,
        default=30,
        help="Окно в кадрах для поиска пересечений с противоположных сторон.",
    )
    parser.add_argument(
        "--red-duration",
        type=int,
        default=45,
        help="Сколько кадров линия остаётся красной после события.",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="ID целевого класса для отслеживания",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default="person",
        help="Имя целевого класса, если не задан --class-id.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Порог уверенности детекции.",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Выводить отладочную информацию о детекциях и пересечениях.",
    )
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=3,
        help="Толщина центральной линии в пикселях.",
    )
   
    parser.add_argument(
        "--stand-frames",
        type=int,
        default=45,
        help="Кадры подряд, когда люди по обе стороны линии, для срабатывания события.",
    )
    parser.add_argument(
        "--stand-distance-px",
        type=int,
        default=220,
        help="Макс. дистанция (в пикселях) до линии для условия 'стоят по сторонам'.",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default="",
        help="Путь для сохранения выходного видео (например, out.mp4).",
    )
    return parser.parse_args()


def open_source(source_arg: str):
    source = int(source_arg) if source_arg.isdigit() else source_arg
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть источник видео: {source_arg}")
    return cap


def resolve_target_class_id(model: YOLO, class_id: int | None, class_name: str) -> int:
    if class_id is not None:
        return class_id

    names = model.names
    names_map = names if isinstance(names, dict) else {i: n for i, n in enumerate(names)}
    class_name = class_name.lower().strip()
    for idx, name in names_map.items():
        if str(name).lower() == class_name:
            return int(idx)
    raise ValueError(
        f"Класс '{class_name}' не найден в model.names: {names_map}. "
        "Укажите --class-id явно."
    )


def main() -> None:
    args = parse_args()

    model = YOLO(model_YOLO)
    names = model.names
    names_map = names if isinstance(names, dict) else {i: n for i, n in enumerate(names)}

    target_class_id = resolve_target_class_id(model, args.class_id, args.class_name)
    print(f"Tracking class id={target_class_id}, name='{names_map.get(target_class_id, 'unknown')}'")

    cap = open_source(args.source)

    touch_events: deque[TouchEvent] = deque()
    track_in_strip: dict[int, bool] = {}
    both_sides_stand_count = 0
    video_writer: cv2.VideoWriter | None = None

    frame_idx = 0
    red_until = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        cx = max(0, min(w - 1, (w // 2) + LINE_OFFSET_X))
        half_strip = max(1, args.line_thickness // 2)
        strip_left = cx - half_strip
        strip_right = cx + half_strip

        if args.output_video and video_writer is None:
            src_fps = cap.get(cv2.CAP_PROP_FPS)
            fps = src_fps if src_fps and src_fps > 0 else 25.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(args.output_video, fourcc, fps, (w, h))
            if not video_writer.isOpened():
                raise RuntimeError(f"Не удалось открыть файл для записи видео: {args.output_video}")
            print(f"Сохранение выходное видео: {args.output_video} (fps={fps:.2f})")

        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=args.conf,
            verbose=False,
        )

        people_in_frame = 0
        current_left_ids: set[int] = set()
        current_right_ids: set[int] = set()
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            ids = boxes.id.int().cpu().tolist()
            classes = boxes.cls.int().cpu().tolist()
            xyxy = boxes.xyxy.cpu().tolist()

            for track_id, cls_id, box in zip(ids, classes, xyxy):
                x1, y1, x2, y2 = map(int, box)
                cls_name = str(names_map.get(cls_id, f"class_{cls_id}"))
                color = (0, 255, 0) if cls_id == target_class_id else (255, 180, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{cls_name} id={track_id}",
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

                if cls_id != target_class_id:
                    continue

                people_in_frame += 1
                person_cx = (x1 + x2) // 2
                intersects_strip = (x1 <= strip_right) and (x2 >= strip_left)
                side = "L" if person_cx < cx else "R"
                was_in_strip = track_in_strip.get(track_id, False)

                if intersects_strip and not was_in_strip:
                    touch_events.append(TouchEvent(frame_idx=frame_idx, side=side, track_id=track_id))
                    if args.debug:
                        print(f"[DEBUG] frame={frame_idx} touch_from={side} id={track_id}")
                near_line = abs(person_cx - cx) <= args.stand_distance_px
                if near_line:
                    if side == "L":
                        current_left_ids.add(track_id)
                    else:
                        current_right_ids.add(track_id)

                track_in_strip[track_id] = intersects_strip

        while touch_events and frame_idx - touch_events[0].frame_idx > args.window_frames:
            touch_events.popleft()

        left_ids = {e.track_id for e in touch_events if e.side == "L"}
        right_ids = {e.track_id for e in touch_events if e.side == "R"}
        has_pair = any(l_id != r_id for l_id in left_ids for r_id in right_ids)
        both_sides_now = any(l_id != r_id for l_id in current_left_ids for r_id in current_right_ids)
        if both_sides_now:
            both_sides_stand_count += 1
        else:
            both_sides_stand_count = 0

        if has_pair or both_sides_stand_count >= args.stand_frames:
            red_until = frame_idx + args.red_duration
            print("Люди передают предмет")
            touch_events.clear()
            both_sides_stand_count = 0

        if args.debug and frame_idx % 30 == 0:
            print(
                f"[DEBUG] frame={frame_idx} people={people_in_frame} "
                f"left_ids={len(left_ids)} right_ids={len(right_ids)} "
                f"stand_count={both_sides_stand_count} conf={args.conf}"
            )

        line_color = (0, 0, 255) if frame_idx <= red_until else (0, 255, 255)
        cv2.line(frame, (cx, 0), (cx, h), line_color, args.line_thickness)

        if video_writer is not None:
            video_writer.write(frame)

        cv2.imshow("YOLO Line", frame)
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
