"""
game_interface.py — Интерфейс с C# игрой через subprocess (stdin/stdout JSON).

Запускает  dotnet run --project ../Game/DungeonRL/DungeonRL.csproj -- --ai
или        dotnet run --project ../Game/DungeonRL/DungeonRL.csproj -- --ai-visual
и общается по протоколу:
  → отправляем: целое число 0-7 (индекс ActionType) или "reset" + '\\n'
  ← получаем:   JSON-строка состояния

Наблюдение (85 признаков):
  [0:81]  — локальный вид 9×9 вокруг игрока (нормализованные 0..1)
  [81:83] — (dx_exit/W, dy_exit/H) до выхода
  [83]    — hp_norm
  [84]    — step_norm

Устойчивость:
  • Перед каждым write проверяем proc.poll() — процесс жив?
  • readline() выполняется в daemon-потоке с таймаутом RECV_TIMEOUT сек.
  • При BrokenPipeError / таймауте / EOF — процесс перезапускается.
  • step() при сбое возвращает (zeros, CRASH_REWARD, True) — эпизод завершается.
"""

import json
import subprocess
import threading
import time
import numpy as np
from typing import Tuple

# ── Путь к проекту ────────────────────────────────────────────────────────────
PROJECT_PATH = "../Game/DungeonRL/DungeonRL.csproj"

# ── Размеры наблюдения ────────────────────────────────────────────────────────
MAP_W = MAP_H = 32
VIEW  = 9
HALF  = VIEW // 2

# Коды тайлов/существ в наблюдении
TILE_WALL     = 0
TILE_FLOOR    = 1
TILE_EXIT     = 2
TILE_PIT      = 3
ENEMY_WALK    = 4
ENEMY_FLY     = 5
ENEMY_CRAWL   = 6
ENTITY_PLAYER = 7

MAX_STEPS    = 500        # совпадает с DungeonEnv.MaxSteps
MAX_HP       = 10

# ── Надёжность ────────────────────────────────────────────────────────────────
RECV_TIMEOUT  = 30.0      # секунд ожидания ответа от C# до рестарта
CRASH_REWARD  = -5.0      # штраф за крэш C# (эпизод завершается)
MAX_RESTARTS  = 10        # максимум авторестартов подряд (потом исключение)


class GameInterface:
    def __init__(self, seed_start: int = 0, visual_mode: bool = False):
        """
        Parameters
        ----------
        seed_start : int
            Начальный сид генерации карты.
        visual_mode : bool
            True  → запускает C# с флагом --ai-visual (открывает SFML-окно).
            False → запускает с --ai (без окна, только JSON).
        """
        self._seed          = seed_start
        self._visual_mode   = visual_mode
        self._restarts      = 0
        self._proc          = self._launch()
        self._state: dict   = {}

    # ── Запуск C# процесса ────────────────────────────────────────────────────
    def _launch(self) -> subprocess.Popen:
        flag = "--ai-visual" if self._visual_mode else "--ai"
        return subprocess.Popen(
            ["dotnet", "run", "--project", PROJECT_PATH, "--", flag],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    # ── Перезапуск упавшего процесса ─────────────────────────────────────────
    def _restart_proc(self) -> None:
        self._restarts += 1
        if self._restarts > MAX_RESTARTS:
            raise RuntimeError(
                f"C# процесс упал {MAX_RESTARTS} раз подряд — остановка."
            )
        print(f"\n[GameInterface] C# упал, перезапуск #{self._restarts}…", flush=True)
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            pass
        time.sleep(1.0)   # небольшая пауза перед рестартом
        self._proc = self._launch()

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset(self) -> np.ndarray:
        """
        Отправляет 'reset', ждёт начального состояния.
        При сбое перезапускает процесс и повторяет.
        """
        for attempt in range(3):
            try:
                self._send_raw("reset")
                self._state = self._recv_json()
                self._restarts = 0          # успешный обмен — сбрасываем счётчик
                return self._make_obs(self._state)
            except (TimeoutError, EOFError, RuntimeError, json.JSONDecodeError,
                    BrokenPipeError, OSError) as exc:
                print(f"\n[GameInterface] reset сбой (попытка {attempt+1}): {exc}", flush=True)
                self._restart_proc()
        # Если все попытки провалились — возвращаем нулевое наблюдение
        return np.zeros(VIEW * VIEW + 4, dtype=np.float32)

    # ── Step ──────────────────────────────────────────────────────────────────
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Отправляет действие (0-7), возвращает (obs, reward, done).
        При сбое C# возвращает (zeros, CRASH_REWARD, True).
        """
        try:
            self._send_raw(str(action))
            self._state = self._recv_json()
            self._restarts = 0
            obs    = self._make_obs(self._state)
            reward = float(self._state.get("reward", 0.0))
            done   = bool(self._state.get("done", True))
            return obs, reward, done
        except (TimeoutError, EOFError, RuntimeError, json.JSONDecodeError,
                BrokenPipeError, OSError) as exc:
            print(f"\n[GameInterface] step сбой: {exc}", flush=True)
            self._restart_proc()
            zeros = np.zeros(VIEW * VIEW + 4, dtype=np.float32)
            return zeros, CRASH_REWARD, True   # завершаем эпизод

    # ── Последнее известное состояние (для визуализатора) ─────────────────────
    @property
    def last_state(self) -> dict:
        return self._state

    # ── Построение вектора наблюдения ────────────────────────────────────────
    def _make_obs(self, s: dict) -> np.ndarray:
        px, py   = s["player_x"], s["player_y"]
        flat_map = s["map"]

        # --- Карта врагов ---
        enemy_at: dict[tuple, int] = {}
        for e in s.get("enemies", []):
            if e["alive"]:
                code = ENEMY_WALK + e["type"]
                enemy_at[(e["x"], e["y"])] = code

        # --- Локальный вид 9×9 ---
        view = np.zeros(VIEW * VIEW, dtype=np.float32)
        idx  = 0
        for dy in range(-HALF, HALF + 1):
            for dx in range(-HALF, HALF + 1):
                tx, ty = px + dx, py + dy

                if not (0 <= tx < MAP_W and 0 <= ty < MAP_H):
                    val = TILE_WALL
                elif (tx, ty) == (px, py):
                    val = ENTITY_PLAYER
                elif (tx, ty) in enemy_at:
                    val = enemy_at[(tx, ty)]
                else:
                    raw = flat_map[ty * MAP_W + tx]
                    val = max(0, raw - 1)

                view[idx] = val / 7.0
                idx += 1

        # --- Дополнительные признаки ---
        ex, ey    = s["exit_x"], s["exit_y"]
        dx_exit   = (ex - px) / MAP_W
        dy_exit   = (ey - py) / MAP_H
        hp_norm   = s["player_hp"] / MAX_HP
        step_norm = s.get("step", 0) / MAX_STEPS

        return np.concatenate([view, [dx_exit, dy_exit, hp_norm, step_norm]])

    # ── Протокол ─────────────────────────────────────────────────────────────
    def _send_raw(self, text: str) -> None:
        """Пишет строку в stdin C#. Поднимает OSError/BrokenPipeError если процесс мёртв."""
        if self._proc.poll() is not None:
            raise RuntimeError(f"C# процесс завершился с кодом {self._proc.poll()}")
        self._proc.stdin.write(text + "\n")
        self._proc.stdin.flush()

    def _recv_json(self) -> dict:
        """
        Читает одну JSON-строку из stdout C# с таймаутом RECV_TIMEOUT.
        Запускает readline() в daemon-потоке, чтобы не блокировать навсегда.
        """
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
            raise TimeoutError(
                f"C# не ответил за {RECV_TIMEOUT} сек — процесс завис"
            )
        if error[0] is not None:
            raise error[0]

        line = result[0]
        if not line:
            raise EOFError("C# закрыл stdout (EOF)")

        return json.loads(line)

    # ── Закрытие ──────────────────────────────────────────────────────────────
    def close(self) -> None:
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
