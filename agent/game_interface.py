"""
game_interface.py — Надёжный интерфейс с C# через subprocess.

Протокол:
  → Python: строка действия (0-6) или "reset [seed]" + LF
  ← C#:     JSON-строка состояния + LF

Формат JSON (C# → Python):
  {
    "player_x": int, "player_y": int,
    "player_hp": int,
    "exit_x": int, "exit_y": int,
    "map": [[int, ...], ...],   ← матрица [H][W], сущности встроены в тайлы
    "reward": float, "done": bool, "step": int
  }

Кодировка ячеек матрицы map[y][x]:
  0=Empty  1=Wall  2=Floor  3=Exit  4=Pit
  5=WalkingEnemy  6=FlyingEnemy  7=CrawlingEnemy  8=Player

Наблюдение (357 признаков):
  [0..288]   — 17×17 локальный вид (289 пикселей), нормализован / ENTITY_MAX
  [289..352] — 8×8 мини-карта (64 пикселя), avg-pooling полной карты / ENTITY_MAX
  [353]      — (exit_x − px) / map_w     ∈ [−1, 1]
  [354]      — (exit_y − py) / map_h     ∈ [−1, 1]
  [355]      — player_hp / MAX_HP        ∈ [0, 1]
  [356]      — step / MAX_STEPS          ∈ [0, 1]

Защита от зависаний (Windows-специфика):
  • _send_raw()  — поток с таймаутом SEND_TIMEOUT сек.
  • _recv_json() — поток с таймаутом RECV_TIMEOUT сек.
  • _kill_tree() — taskkill /F /T убивает ДЕРЕВО процессов.
  • --no-build   — перезапуск без рекомпиляции.
"""

import json
import platform
import subprocess
import threading
import time
import numpy as np
from typing import Tuple
import os

# ── Путь к проекту ─────────────────────────────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_PATH = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "Game", "DungeonRL", "DungeonRL.csproj"))
PROJECT_DIR  = os.path.dirname(PROJECT_PATH)

# ── Параметры наблюдения ───────────────────────────────────────────────────
VIEW      = 33          # 33×33 локальный вид (16 тайлов в каждую сторону)
HALF      = VIEW // 2   # = 16
MINI_SIZE = 8           # размер глобальной мини-карты (8×8)

# OBS_SIZE = 33×33 + 8×8 + 4 = 1089 + 64 + 4 = 1157
# (должен совпадать с model.py OBS_SIZE)
OBS_SIZE = VIEW * VIEW + MINI_SIZE * MINI_SIZE + 4

# Кодировка ячеек entity-матрицы (совпадает с AiProtocol.cs / BuildEntityMap)
TILE_EMPTY    = 0
TILE_WALL     = 1
TILE_FLOOR    = 2
ENTITY_PLAYER = 3   # низкий приоритет — мы и так знаем где игрок
TILE_PIT      = 4
ENEMY_WALK    = 5
ENEMY_FLY     = 6
ENEMY_CRAWL   = 7
TILE_EXIT     = 8   # вершина иерархии — доминирует в max-pool мини-карты
ENTITY_MAX    = 8   # делитель нормализации (exit/8 = 1.0)

MAX_STEPS      = 500
MAX_HP         = 10
_CS_SEED_START = 75

# ── Таймауты и лимиты ──────────────────────────────────────────────────────
SEND_TIMEOUT  = 10.0
RECV_TIMEOUT  = 30.0
RESTART_DELAY = 2.0
MAX_RESTARTS  = 5
CRASH_REWARD  = -5.0


class GameInterface:
    def __init__(self, seed_start: int = 0, visual_mode: bool = False,
                 map_size: int = 16, no_enemies: bool = False, player_hp: int = 10):
        """
        map_size   — размер карты (передаётся C# через --map-size:N).
        no_enemies — если True, враги не спавнятся (Фаза 0: чистая навигация).
        player_hp  — HP игрока при старте эпизода (передаётся через --player-hp:N).
        """
        self._visual_mode  = visual_mode
        self._map_size     = map_size
        self._no_enemies   = no_enemies
        self._player_hp    = player_hp
        self._restarts     = 0
        self._built        = False
        self._state: dict  = {}

        self._cs_seed      = _CS_SEED_START
        self._episode_seed = _CS_SEED_START

        self._build_once()
        self._proc = self._launch()

    # ── Сборка проекта (один раз при старте) ──────────────────────────────
    def _build_once(self) -> None:
        if self._built:
            return
        print("[GameInterface] dotnet build … ", end="", flush=True)
        try:
            result = subprocess.run(
                ["dotnet", "build", PROJECT_PATH,
                 "-c", "Debug", "--nologo", "-v", "q"],
                capture_output=True, text=True, timeout=120,
                cwd=PROJECT_DIR,
            )
            if result.returncode == 0:
                print("OK")
                self._built = True
            else:
                print(f"ОШИБКА:\n{result.stderr[-600:]}")
        except Exception as exc:
            print(f"не удалось: {exc}")

    # ── Запуск процесса ────────────────────────────────────────────────────
    def _launch(self) -> subprocess.Popen:
        flag = "--ai-visual" if self._visual_mode else "--ai"
        cmd  = ["dotnet", "run", "--project", PROJECT_PATH]
        if self._built:
            cmd.append("--no-build")
        # Передаём флаг режима, размер карты, HP игрока и опционально --no-enemies
        cmd += ["--", flag, f"--map-size:{self._map_size}", f"--player-hp:{self._player_hp}"]
        if self._no_enemies:
            cmd.append("--no-enemies")

        kwargs: dict = dict(
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=PROJECT_DIR,
        )
        if platform.system() == "Windows":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        proc = subprocess.Popen(cmd, **kwargs)
        self._drain_stderr(proc)
        return proc

    # ── Слив stderr ───────────────────────────────────────────────────────
    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        lines: list[str] = []

        def _reader():
            try:
                for line in proc.stderr:
                    lines.append(line)
            except Exception:
                pass
            rc = proc.poll()
            if rc not in (None, 0) and lines:
                tail = "".join(lines[-20:]).strip()
                print(f"\n[GameInterface] C# stderr:\n{tail}\n", flush=True)

        t = threading.Thread(target=_reader, daemon=True, name="stderr-drain")
        t.start()

    # ── Убийство ДЕРЕВА процессов ──────────────────────────────────────────
    def _kill_tree(self) -> None:
        pid = self._proc.pid
        if platform.system() == "Windows":
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True, timeout=8
                )
            except Exception:
                pass
        else:
            import signal
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except Exception:
                pass
        try:
            self._proc.wait(timeout=5)
        except Exception:
            pass

    # ── Перезапуск ────────────────────────────────────────────────────────
    def _restart(self) -> None:
        self._restarts += 1
        if self._restarts > MAX_RESTARTS:
            raise RuntimeError(
                f"[GameInterface] C# упал {MAX_RESTARTS} раз подряд — "
                "обучение остановлено."
            )
        print(
            f"\n[GameInterface] Перезапуск C# ({self._restarts}/{MAX_RESTARTS})…",
            flush=True
        )
        self._kill_tree()
        time.sleep(RESTART_DELAY)
        self._cs_seed = _CS_SEED_START
        self._proc = self._launch()

    # ── Reset ─────────────────────────────────────────────────────────────
    def reset(self) -> np.ndarray:
        for attempt in range(3):
            try:
                self._episode_seed = self._cs_seed
                # Явно передаём сид в C# — иначе C# генерирует случайный сид
                self._send_raw(f"reset {self._cs_seed}")
                self._cs_seed += 1          # следующий эпизод = следующий сид
                self._state = self._recv_json()
                self._restarts = 0
                return self._make_obs(self._state)
            except Exception as exc:
                print(f"\n[GameInterface] reset сбой (попытка {attempt+1}): {exc}",
                      flush=True)
                self._restart()
        return np.zeros(OBS_SIZE, dtype=np.float32)

    def reset_to_seed(self, seed: int) -> np.ndarray:
        for attempt in range(3):
            try:
                self._episode_seed = seed
                self._cs_seed      = seed
                self._send_raw(f"reset {seed}")
                self._state = self._recv_json()
                self._restarts = 0
                return self._make_obs(self._state)
            except Exception as exc:
                print(f"\n[GameInterface] reset_to_seed({seed}) сбой "
                      f"(попытка {attempt+1}): {exc}", flush=True)
                self._restart()
        return np.zeros(OBS_SIZE, dtype=np.float32)

    @property
    def episode_seed(self) -> int:
        return self._episode_seed

    # ── Step ──────────────────────────────────────────────────────────────
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        try:
            self._send_raw(str(action))
            self._state = self._recv_json()
            self._restarts = 0
            obs    = self._make_obs(self._state)
            reward = float(self._state.get("reward", 0.0))
            done   = bool(self._state.get("done", True))
            return obs, reward, done
        except Exception as exc:
            print(f"\n[GameInterface] step сбой: {exc}", flush=True)
            self._restart()
            return np.zeros(OBS_SIZE, dtype=np.float32), CRASH_REWARD, True

    @property
    def last_state(self) -> dict:
        return self._state

    # ── Построение наблюдения ─────────────────────────────────────────────
    def _make_obs(self, s: dict) -> np.ndarray:
        """
        Строит вектор наблюдения из entity-матрицы C#.

        Итоговый вектор (1157 признаков):
          [0..1088]    — 33×33 local view (нормализован / ENTITY_MAX)
          [1089..1152] — 8×8 мини-карта (max-pooling, нормализован / ENTITY_MAX)
          [1153]       — (exit_x − px) / map_w
          [1154]       — (exit_y − py) / map_h
          [1155]       — player_hp / MAX_HP
          [1156]       — step / MAX_STEPS
        """
        px, py = s["player_x"], s["player_y"]
        map2d  = s["map"]             # list[list[int]] — [H][W]
        map_h  = len(map2d)
        map_w  = len(map2d[0]) if map_h > 0 else self._map_size

        # ── 17×17 локальный вид ─────────────────────────────────────────────
        view = np.zeros(VIEW * VIEW, dtype=np.float32)
        idx  = 0
        for dy in range(-HALF, HALF + 1):
            for dx in range(-HALF, HALF + 1):
                tx, ty = px + dx, py + dy
                if 0 <= ty < map_h and 0 <= tx < map_w:
                    val = map2d[ty][tx]
                else:
                    val = TILE_WALL   # за границей = стена
                view[idx] = val / ENTITY_MAX
                idx += 1

        # ── 8×8 глобальная мини-карта (max-pooling) ─────────────────────────
        # Вся карта → нормализованный массив (H×W) → max-pool до (8×8)
        # max-pooling гарантирует: выход (3), враги (5-7) видны в любой ячейке,
        # даже если занимают 1 тайл из 16. avg-pooling размывает их до шума.
        full_map = np.array(map2d, dtype=np.float32) / ENTITY_MAX  # (H, W)
        mini_map = _max_pool_2d(full_map, MINI_SIZE, MINI_SIZE)     # (8, 8)
        mini_flat = mini_map.flatten()                               # (64,)

        # ── Скалярные признаки ───────────────────────────────────────────────
        ex, ey    = s["exit_x"], s["exit_y"]
        max_hp    = s.get("max_hp", MAX_HP)          # реальный MaxHP из стейта
        hp_norm   = s["player_hp"] / max_hp if max_hp > 0 else 0.0
        step_norm = s.get("step", 0) / MAX_STEPS

        return np.concatenate([
            view,                                           # 289
            mini_flat,                                      # 64
            [(ex - px) / map_w, (ey - py) / map_h,        # 2
             hp_norm, step_norm],                           # 2  → total 357
        ])

    # ── Низкоуровневый протокол ────────────────────────────────────────────
    def _send_raw(self, text: str) -> None:
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"C# завершился с кодом {self._proc.poll()} до отправки '{text}'"
            )
        error: list = [None]

        def _writer():
            try:
                self._proc.stdin.write(text + "\n")
                self._proc.stdin.flush()
            except Exception as e:
                error[0] = e

        t = threading.Thread(target=_writer, daemon=True)
        t.start()
        t.join(SEND_TIMEOUT)

        if t.is_alive():
            raise TimeoutError(f"C# не читает stdin за {SEND_TIMEOUT} сек")
        if error[0] is not None:
            raise error[0]

    def _recv_json(self) -> dict:
        result: list = [None]
        error:  list = [None]

        def _reader():
            try:
                result[0] = self._proc.stdout.readline()
            except Exception as e:
                error[0] = e

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        t.join(RECV_TIMEOUT)

        if t.is_alive():
            raise TimeoutError(f"C# не ответил за {RECV_TIMEOUT} сек")
        if error[0] is not None:
            raise error[0]

        line = result[0]
        if not line:
            raise EOFError("C# закрыл stdout (EOF) — процесс упал")

        return json.loads(line)

    # ── Закрытие ──────────────────────────────────────────────────────────
    def close(self) -> None:
        self._kill_tree()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ── Вспомогательная функция: max-pool 2D массива до заданного размера ──────
def _max_pool_2d(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    Уменьшает 2D-массив до (out_h, out_w) взятием максимума по блоку.
    Работает для любого входного размера (не обязательно кратного out_h/out_w).

    В отличие от avg-pooling, max-pooling сохраняет сигнал редких объектов:
      - Выход (TILE_EXIT=3): 3/8=0.375  (vs floor 2/8=0.250) — чётко виден
      - Враги (5-7/8=0.625-0.875)       — доминируют в ячейке
      - Стены (1/8=0.125) не маскируют floor/exit даже в угловых блоках

    Пример 16x16 -> 8x8: блок 2x2, max из 4 тайлов.
             32x32 -> 8x8: блок 4x4, max из 16 тайлов — сигнал не теряется.
    """
    h, w = arr.shape
    result = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        y0 = int(i * h / out_h)
        y1 = int((i + 1) * h / out_h)
        y1 = max(y1, y0 + 1)
        for j in range(out_w):
            x0 = int(j * w / out_w)
            x1 = int((j + 1) * w / out_w)
            x1 = max(x1, x0 + 1)
            result[i, j] = arr[y0:y1, x0:x1].max()
    return result
