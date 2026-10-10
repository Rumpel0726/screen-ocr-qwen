# -*- coding: utf-8 -*-
r"""
OCR выделенной области экрана через Qwen3-VL-8B по API RouterAI.

Работа:
    1. Скрипт висит в фоне и слушает глобальный хоткей (по умолчанию Ctrl+Shift+F9).
    2. По хоткею экран затемняется, курсор — перекрестие, мышью выделяется область.
       Esc (или правая кнопка мыши) отменяет выделение.
    3. Область скриншотится, ужимается по ширине и уходит в RouterAI.
    4. Результат кладётся в буфер обмена и показывается в маленьком окне.
    5. В окне есть строка вопроса и кнопка "Спросить": можно спросить модель
       про этот же фрагмент (перевести, объяснить ошибку, посчитать).
       Картинка и распознанный текст остаются в контексте, вопросы можно
       задавать подряд. Enter отправляет, Esc чистит поле, Esc на пустом
       поле закрывает окно. Скопировать ответ можно четырьмя способами:
       выделить мышью и Ctrl+C, правый клик → меню, кнопка "Копировать
       ответ", Ctrl+A → Ctrl+C. Всё это работает и на русской раскладке.
    6. Скрипт возвращается в фоновое ожидание следующего хоткея.

------------------------------------------------------------------------------
УСТАНОВКА ЗАВИСИМОСТЕЙ (Windows 10, Python 3.11+)
------------------------------------------------------------------------------
    py -m pip install -r requirements.txt

    tkinter входит в стандартную поставку Python с python.org.
    Если tkinter отсутствует — переустановите Python, отметив
    "tcl/tk and IDLE" в инсталляторе.

API-КЛЮЧ ROUTERAI (файл .env)
    Ключ выдаётся в личном кабинете https://routerai.ru и НЕ хранится в коде.
    Рядом со скриптом лежит файл .env с одной строкой:

        ROUTERAI_API_KEY=sk-...

    Шаблон — .env.example. Файл .env внесён в .gitignore: не коммитьте его
    и не пересылайте. Если ключ утёк — отзовите его в личном кабинете
    и впишите сюда новый.

    Читается при старте скрипта, поэтому после правки .env нужен перезапуск.
    Переменная окружения с тем же именем, если она задана, имеет приоритет
    над файлом — удобно для временной подмены ключа.

    Список моделей и их цены: https://routerai.ru/models
    Нужна vision-модель, у неё в описании modality "text+image->text".

ЗАПУСК
    py ocr_qwen.py            — с консолью (видны логи и ошибки)
    pyw ocr_qwen.py           — без консоли, чисто фоном

------------------------------------------------------------------------------
АВТОЗАГРУЗКА WINDOWS (папка shell:startup)
------------------------------------------------------------------------------
    Положите в папку автозагрузки ярлык, например "OCR Qwen.lnk"
        %APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\OCR Qwen.lnk
    с объектом
        "<...>\Python313\pythonw.exe" "<путь к проекту>\ocr_qwen.py"
    и рабочей папкой <путь к проекту>

    Открыть эту папку: Win+R → shell:startup → Enter.
    Отключить автозапуск — удалить оттуда ярлык (или снять галку
    в Диспетчере задач → вкладка "Автозагрузка").

    Почему ярлык, а не .cmd: если путь проекта содержит кириллицу, а .cmd-файлы
    интерпретатор читает в кодировке OEM (cp866) — путь в нём может побиться.
    Ярлык хранит путь в Unicode и вдобавок не мигает окном консоли.
    Если проект переедет в путь без кириллицы, годится и .cmd:

        @echo off
        start "" "<путь>\pythonw.exe" "<путь>\ocr_qwen.py"

    pythonw.exe (не python.exe) запускает скрипт без окна консоли.
    Точный путь: where pythonw

    Скрипт не показывает иконку в трее — завершать через Диспетчер задач
    (процесс pythonw.exe) или по Ctrl+C в консоли, если запущен через py.

    Ключ берётся из .env рядом со скриптом, поэтому автозагрузке не нужны
    никакие переменные окружения — путь к .env вычисляется от файла скрипта,
    а не от текущей папки.
------------------------------------------------------------------------------
"""

import base64
import ctypes
import io
import os
import queue
import sys
import threading
import tkinter as tk

import mss
import pyperclip
import requests
from PIL import Image
from pynput import keyboard

# =============================================================================
#                                  КОНФИГ
# =============================================================================

# Глобальный хоткей в формате pynput: <ctrl>, <alt>, <shift>, <cmd> + клавиша.
# F-клавиша с двумя модификаторами почти никогда не занята чужими программами.
# Другие свободные варианты, если этот тоже конфликтует:
#   "<ctrl>+<shift>+<f10>"   "<ctrl>+<alt>+<f9>"
#   "<ctrl>+<shift>+<alt>+q" "<cmd>+<shift>+q"   (<cmd> — клавиша Win)
HOTKEY = "<ctrl>+<shift>+<f9>"

# OpenAI-совместимый эндпоинт RouterAI.
API_URL = "https://routerai.ru/api/v1/chat/completions"

# Ключ не хранится в исходнике. Он берётся из файла .env рядом со скриптом
# (строка ROUTERAI_API_KEY=sk-...), а если такой строки нет — из одноимённой
# переменной окружения Windows. Здесь задаётся только ИМЯ переменной.
API_KEY_ENV = "ROUTERAI_API_KEY"

# Путь к .env. По умолчанию — рядом со скриптом, независимо от того, из какой
# папки его запустили (важно для автозагрузки: там текущая папка — System32).
ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

# Аварийный вариант: вписать ключ прямо сюда. Оставьте пустым, если есть .env.
API_KEY_FALLBACK = ""

# Модель. Список: https://routerai.ru/models или GET /api/v1/models.
# Нужна vision-модель (modality "text+image->text").
MODEL = "qwen/qwen3-vl-8b-instruct"

# Промпт составлен так, чтобы модель вернула только текст без комментариев.
PROMPT = (
    "Extract all text from this image. Output only the extracted text, exactly "
    "as it appears, preserving line breaks and original language. Do not add any "
    "commentary, explanation, headers, or markdown formatting. If there is no "
    "text, output nothing."
)

# Таймаут запроса, секунды: (на соединение, на ответ).
REQUEST_TIMEOUT = (10, 120)

# Потолок токенов в ответе на распознавание.
MAX_TOKENS = 2048

# Режим вопросов (кнопка «Спросить» в окне результата). Картинка остаётся
# в истории диалога, поэтому модель видит и её, и распознанный текст.
CHAT_SYSTEM = (
    "You are a concise assistant. The user has captured a screen region; the "
    "image and its extracted text are in the conversation. Answer questions "
    "about that content. Always reply in the same language as the question. "
    "Be brief and practical, no filler."
)
CHAT_TEMPERATURE = 0.3
CHAT_MAX_TOKENS = 2048

# Сколько сообщений диалога держать в истории (не считая первого с картинкой).
# Ограничение бережёт токены: картинка отправляется заново с каждым вопросом.
CHAT_HISTORY_LIMIT = 12

# Максимальная ширина картинки перед отправкой, px. Более широкие области
# ужимаются с сохранением пропорций, чтобы не раздувать число визуальных
# токенов. Значение намеренно щедрое — мелкий текст должен остаться читаемым.
# Уменьшать ниже ~1000 не стоит.
MAX_IMAGE_WIDTH = 1600

# Минимальная сторона выделения, px. Всё меньше считается случайным кликом
# и отменяется молча.
MIN_SELECTION_PX = 8

# Внешний вид затемняющего оверлея.
OVERLAY_ALPHA = 0.35
OVERLAY_COLOR = "#000000"
SELECTION_OUTLINE = "#00c8ff"

# Окно результата.
RESULT_MAX_COLS = 90       # ширина Text в символах
RESULT_MAX_ROWS = 28       # высота Text в строках до появления прокрутки
RESULT_FONT = ("Consolas", 10)

# Минимальный размер окна после первого вопроса: короткий OCR-результат даёт
# узкое окно, в котором ответы модели читать неудобно.
CHAT_MIN_COLS = 64
CHAT_MIN_ROWS = 18

# =============================================================================


# DPI awareness обязателен ДО создания любых окон, иначе на масштабе != 100%
# координаты выделения разъезжаются с реальными пикселями экрана.
def _enable_dpi_awareness() -> None:
    try:
        # 2 == PROCESS_PER_MONITOR_DPI_AWARE
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


_enable_dpi_awareness()


def load_env_file(path: str) -> None:
    """Минимальный загрузчик .env: KEY=VALUE, строки с # и пустые пропускаются.

    Значения из настоящего окружения имеют приоритет и не перезаписываются —
    так временный `set ROUTERAI_API_KEY=...` перекрывает файл.
    Отдельная зависимость (python-dotenv) ради десятка строк не нужна.
    """
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            lines = fh.readlines()
    except OSError:
        return

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file(ENV_FILE)


def get_virtual_desktop() -> tuple:
    """Границы виртуального рабочего стола (все мониторы) в абсолютных px."""
    user32 = ctypes.windll.user32
    SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN = 76, 77
    SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 78, 79
    return (
        user32.GetSystemMetrics(SM_XVIRTUALSCREEN),
        user32.GetSystemMetrics(SM_YVIRTUALSCREEN),
        user32.GetSystemMetrics(SM_CXVIRTUALSCREEN),
        user32.GetSystemMetrics(SM_CYVIRTUALSCREEN),
    )


# -----------------------------------------------------------------------------
# Захват экрана
# -----------------------------------------------------------------------------

def grab_region(left: int, top: int, width: int, height: int) -> Image.Image:
    """Скриншот прямоугольника по абсолютным координатам виртуального десктопа."""
    # В mss 10 класс переименован в MSS, mss.mss() оставлен как устаревший алиас.
    factory = getattr(mss, "MSS", None) or mss.mss
    with factory() as sct:
        raw = sct.grab({"left": left, "top": top, "width": width, "height": height})
    return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")


def image_to_base64_png(img: Image.Image) -> str:
    """Даунскейл по ширине с сохранением пропорций → PNG → base64."""
    if img.width > MAX_IMAGE_WIDTH:
        ratio = MAX_IMAGE_WIDTH / float(img.width)
        new_size = (MAX_IMAGE_WIDTH, max(1, int(round(img.height * ratio))))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# -----------------------------------------------------------------------------
# Запрос к RouterAI
# -----------------------------------------------------------------------------

class OcrError(Exception):
    """Ошибка распознавания с готовым для показа пользователю текстом."""


def strip_wrappers(text: str) -> str:
    """Снимает markdown-обёртку ```...```, если модель всё же её добавила."""
    t = text.strip()
    if not t.startswith("```"):
        return t

    lines = t.split("\n")
    # Первая строка — ``` или ```lang, её выбрасываем целиком.
    lines = lines[1:]
    # Ищем последнюю строку, состоящую только из закрывающих бэктиков.
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip().startswith("```"):
            lines = lines[:i]
            break
    return "\n".join(lines).strip("\n")


def get_api_key() -> str:
    key = (os.environ.get(API_KEY_ENV) or API_KEY_FALLBACK).strip()
    if not key:
        raise OcrError(
            "Не задан API-ключ RouterAI.\n\n"
            f"Ожидался файл:\n{ENV_FILE}\n\n"
            "со строкой:\n\n"
            f"    {API_KEY_ENV}=sk-...\n\n"
            "Файл читается при старте, так что после правки перезапустите скрипт.\n"
            "Ключ выдаётся в личном кабинете: https://routerai.ru"
        )
    return key


def build_ocr_messages(b64_png: str) -> list:
    """Первое сообщение диалога: картинка + строгий OCR-промпт."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + b64_png},
                },
                {"type": "text", "text": PROMPT},
            ],
        }
    ]


def call_api(messages: list, temperature: float, max_tokens: int) -> str:
    """Один запрос к RouterAI. Возвращает content ответа как есть."""
    headers = {
        "Authorization": "Bearer " + get_api_key(),
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    try:
        resp = requests.post(
            API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT
        )
    except requests.exceptions.SSLError as exc:
        raise OcrError(f"Ошибка TLS при обращении к RouterAI:\n\n{exc}")
    except requests.exceptions.ConnectionError:
        raise OcrError(
            "Не удалось подключиться к RouterAI.\n\n"
            f"Адрес: {API_URL}\n\n"
            "Проверьте интернет, прокси/VPN и доступность routerai.ru\n"
            "(в браузере или командой: curl https://routerai.ru/api/v1/models)."
        )
    except requests.exceptions.ReadTimeout:
        raise OcrError(
            f"Таймаут {REQUEST_TIMEOUT[1]} с — модель не успела ответить.\n\n"
            "Попробуйте выделить область поменьше, уменьшить MAX_IMAGE_WIDTH\n"
            "или увеличить REQUEST_TIMEOUT в конфиге скрипта."
        )
    except requests.exceptions.RequestException as exc:
        raise OcrError(f"Ошибка запроса к RouterAI:\n\n{exc}")

    body = resp.text
    try:
        data = resp.json()
    except ValueError:
        raise OcrError(
            f"RouterAI вернул не JSON (HTTP {resp.status_code}):\n\n{body[:2000]}"
        )

    def detail_of(payload_obj):
        detail = payload_obj.get("error", payload_obj) if isinstance(payload_obj, dict) else payload_obj
        if isinstance(detail, dict):
            detail = detail.get("message") or detail.get("detail") or detail
        return detail

    if resp.status_code != 200:
        hints = {
            400: f"Проверьте параметры запроса и имя модели (MODEL = {MODEL!r}).",
            401: "Неверный или просроченный API-ключ. Проверьте "
                 f"переменную {API_KEY_ENV} / API_KEY_FALLBACK.",
            402: "Недостаточно средств на балансе RouterAI — пополните счёт.",
            404: f"Модель {MODEL!r} не найдена. Список: https://routerai.ru/models",
            429: "Превышен лимит запросов — подождите немного и повторите.",
        }
        hint = hints.get(
            resp.status_code,
            "Ошибка на стороне сервиса, попробуйте повторить запрос."
            if resp.status_code >= 500
            else f"Текущая модель в конфиге: {MODEL!r}",
        )
        raise OcrError(
            f"RouterAI ответил ошибкой HTTP {resp.status_code}:\n\n"
            f"{detail_of(data)}\n\n{hint}"
        )

    if "error" in data:
        raise OcrError(
            f"RouterAI вернул ошибку:\n\n{detail_of(data)}\n\n"
            f"Текущая модель в конфиге: {MODEL!r}"
        )

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise OcrError(
            "Неожиданный формат ответа RouterAI:\n\n" + body[:2000]
        )

    # Некоторые провайдеры возвращают content списком блоков.
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )

    return content or ""


def ocr_via_api(b64_png: str) -> str:
    """Распознавание: temperature 0 и снятие возможной markdown-обёртки."""
    content = call_api(build_ocr_messages(b64_png), 0, MAX_TOKENS)
    return strip_wrappers(content)


def ask_via_api(history: list) -> str:
    """Вопрос по уже распознанной области. Обёртки не снимаем — в ответе
    могут быть законные блоки кода."""
    messages = [{"role": "system", "content": CHAT_SYSTEM}] + history
    return call_api(messages, CHAT_TEMPERATURE, CHAT_MAX_TOKENS).strip()


# -----------------------------------------------------------------------------
# Оверлей выделения
# -----------------------------------------------------------------------------

class SelectionOverlay:
    """Полупрозрачное окно на весь виртуальный десктоп для выбора области."""

    def __init__(self, root: tk.Tk, on_done):
        self.on_done = on_done          # callback(rect | None), rect = (l, t, w, h)
        self.vx, self.vy, self.vw, self.vh = get_virtual_desktop()
        self.start = None
        self.rect_id = None
        self.finished = False

        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.geometry(f"{self.vw}x{self.vh}+{self.vx}+{self.vy}")
        self.win.attributes("-topmost", True)
        self.win.attributes("-alpha", OVERLAY_ALPHA)
        self.win.configure(bg=OVERLAY_COLOR, cursor="crosshair")

        self.canvas = tk.Canvas(
            self.win,
            bg=OVERLAY_COLOR,
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.win.bind("<Escape>", lambda e: self._finish(None))
        self.win.bind("<ButtonPress-3>", lambda e: self._finish(None))

        self.win.focus_force()
        self.canvas.focus_set()
        try:
            self.win.grab_set_global()
        except tk.TclError:
            self.win.grab_set()

    # -- события мыши ---------------------------------------------------------

    def _on_press(self, event):
        self.start = (event.x, event.y)
        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline=SELECTION_OUTLINE, width=2,
        )

    def _on_drag(self, event):
        if self.start is None:
            return
        self.canvas.coords(self.rect_id, self.start[0], self.start[1], event.x, event.y)

    def _on_release(self, event):
        if self.start is None:
            self._finish(None)
            return
        x0, y0 = self.start
        x1, y1 = event.x, event.y
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        width, height = right - left, bottom - top

        # Клик без перетаскивания — тихая отмена.
        if width < MIN_SELECTION_PX or height < MIN_SELECTION_PX:
            self._finish(None)
            return

        # Координаты канвы → абсолютные координаты виртуального десктопа.
        self._finish((left + self.vx, top + self.vy, width, height))

    # -- завершение -----------------------------------------------------------

    def _finish(self, rect):
        if self.finished:
            return
        self.finished = True
        try:
            self.win.grab_release()
        except tk.TclError:
            pass
        self.win.destroy()
        # Даём Windows перерисовать экран, прежде чем делать скриншот,
        # иначе в кадр попадёт затемнение оверлея.
        self.on_done(rect)


# -----------------------------------------------------------------------------
# Окна: индикатор и результат
# -----------------------------------------------------------------------------

class BusyWindow:
    """Маленькое окно-индикатор без рамки поверх всех окон."""

    def __init__(self, root: tk.Tk, text="Распознаю..."):
        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.configure(bg="#1e1e1e")

        frame = tk.Frame(self.win, bg="#1e1e1e", padx=18, pady=10,
                         highlightbackground="#3c3c3c", highlightthickness=1)
        frame.pack()
        self.label = tk.Label(
            frame, text=text, bg="#1e1e1e", fg="#e8e8e8", font=("Segoe UI", 10)
        )
        self.label.pack()

        self.win.update_idletasks()
        w, h = self.win.winfo_width(), self.win.winfo_height()
        sw = self.win.winfo_screenwidth()
        sh = self.win.winfo_screenheight()
        # Правый нижний угол основного монитора — ненавязчиво.
        self.win.geometry(f"+{sw - w - 40}+{sh - h - 80}")

        self._dots = 0
        self._base = text
        self._tick()

    def _tick(self):
        if not self.win.winfo_exists():
            return
        self._dots = (self._dots + 1) % 4
        self.label.config(text=self._base.rstrip(".") + "." * (self._dots or 1))
        self.win.after(400, self._tick)

    def close(self):
        try:
            self.win.destroy()
        except tk.TclError:
            pass


class ResultWindow:
    """Окно с распознанным текстом и, если есть контекст, строкой вопроса.

    Esc и клик мимо закрывают окно, Ctrl+C копирует выделенное. После первого
    вопроса автозакрытие по потере фокуса отключается — иначе набранный вопрос
    терялся бы при переключении на другое окно.
    """

    def __init__(self, root: tk.Tk, text: str, title: str = "OCR",
                 history: list = None, on_ask=None):
        self.history = history          # None → окно без строки вопроса
        self.on_ask = on_ask            # on_ask(window, question) — запуск запроса
        self.pending = False            # ждём ответ модели
        self.autoclose = True
        self.menu_open = False
        self.last_answer = ""
        self.copy_btn = None

        self.win = tk.Toplevel(root)
        self.win.title(title)
        self.win.attributes("-topmost", True)
        self.win.configure(bg="#1e1e1e")

        cols, rows, need_scroll = self._measure(text)

        frame = tk.Frame(self.win, bg="#1e1e1e")
        frame.pack(fill="both", expand=True, padx=8, pady=8)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.txt = tk.Text(
            frame,
            width=cols,
            height=rows,
            wrap="word",
            font=RESULT_FONT,
            bg="#252526",
            fg="#e8e8e8",
            insertbackground="#e8e8e8",
            selectbackground="#264f78",
            # Без этого выделение пропадает, как только фокус уходит в поле
            # ввода: у Text по умолчанию inactiveselectbackground пустой.
            inactiveselectbackground="#264f78",
            relief="flat",
            padx=8,
            pady=6,
        )
        self.txt.grid(row=0, column=0, sticky="nsew")

        self.scroll = tk.Scrollbar(frame, command=self.txt.yview)
        self.txt.configure(yscrollcommand=self.scroll.set)
        self.scroll_shown = False
        if need_scroll:
            self._show_scrollbar()

        self.txt.tag_configure("question", foreground="#4ec9b0",
                               spacing1=8, spacing3=2)
        self.txt.tag_configure("answer", foreground="#e8e8e8", spacing3=6)
        self.txt.tag_configure("status", foreground="#808080")

        self.txt.insert("1.0", text)
        self.txt.bind("<Key>", self._block_edit)
        self.txt.bind("<<Paste>>", lambda e: "break")
        # Привязка по keycode, а не по keysym: см. _on_ctrl_key.
        self.txt.bind("<Control-KeyPress>", self._on_ctrl_key)
        self._build_context_menu()

        self.entry = None
        if self.history is not None and self.on_ask is not None:
            self._build_ask_bar(frame)

        self.win.bind("<Escape>", self._on_escape)
        self.win.bind("<FocusOut>", self._on_focus_out)

        self.win.update_idletasks()
        w, h = self.win.winfo_width(), self.win.winfo_height()
        sw, sh = self.win.winfo_screenwidth(), self.win.winfo_screenheight()
        self.win.geometry(f"+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 3)}")

        self.win.deiconify()
        self.win.focus_force()
        self.txt.focus_set()

    # -- построение -----------------------------------------------------------

    def _measure(self, text):
        lines = text.split("\n")
        longest = max((len(l) for l in lines), default=0)
        cols = max(30, min(RESULT_MAX_COLS, longest + 2))
        # Учитываем перенос длинных строк при подсчёте высоты.
        wrapped = sum(max(1, -(-len(l) // cols)) for l in lines)
        rows = max(3, min(RESULT_MAX_ROWS, wrapped + 1))
        return cols, rows, wrapped + 1 > RESULT_MAX_ROWS

    def _build_ask_bar(self, parent):
        bar = tk.Frame(parent, bg="#1e1e1e")
        bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        bar.columnconfigure(0, weight=1)

        self.entry = tk.Entry(
            bar,
            font=("Segoe UI", 10),
            bg="#3c3c3c",
            fg="#e8e8e8",
            insertbackground="#e8e8e8",
            relief="flat",
        )
        self.entry.grid(row=0, column=0, sticky="ew", ipady=5, padx=(0, 6))
        self.entry.insert(0, "Вопрос по этому тексту...")
        self.entry.configure(fg="#8a8a8a")
        self.entry.bind("<FocusIn>", self._clear_placeholder)
        self.entry.bind("<Return>", lambda e: self._submit())
        # В поле ввода Ctrl+V/Ctrl+A/Ctrl+C ломаются на русской раскладке
        # по той же причине, что и в тексте, — тоже вешаем по keycode.
        self.entry.bind("<Control-KeyPress>", self._on_ctrl_key_entry)

        self.ask_btn = tk.Button(
            bar,
            text="Спросить",
            command=self._submit,
            font=("Segoe UI", 9),
            bg="#0e639c",
            fg="#ffffff",
            activebackground="#1177bb",
            activeforeground="#ffffff",
            relief="flat",
            padx=14,
            pady=4,
            cursor="hand2",
        )
        self.ask_btn.grid(row=0, column=1)

        # Кнопка на случай, когда выделять мышью лень: копирует последний ответ.
        self.copy_btn = tk.Button(
            bar,
            text="Копировать ответ",
            command=self._copy_last_answer,
            font=("Segoe UI", 9),
            bg="#3c3c3c",
            fg="#e8e8e8",
            activebackground="#4a4a4a",
            activeforeground="#ffffff",
            relief="flat",
            padx=10,
            pady=4,
            cursor="hand2",
            state="disabled",
        )
        self.copy_btn.grid(row=0, column=2, padx=(6, 0))

    def _on_ctrl_key_entry(self, event):
        """Ctrl-сочетания в поле ввода — тоже по keycode (см. _on_ctrl_key)."""
        kc = event.keycode
        if kc == 86:                       # VK_V — вставить
            try:
                self.entry.insert("insert", pyperclip.paste())
            except Exception:
                pass
            return "break"
        if kc == 67:                       # VK_C — копировать выделенное
            if self.entry.selection_present():
                pyperclip.copy(self.entry.selection_get())
            else:
                self._copy_selection()     # в поле пусто — копируем из текста
            return "break"
        if kc == 88:                       # VK_X — вырезать
            if self.entry.selection_present():
                pyperclip.copy(self.entry.selection_get())
                self.entry.delete("sel.first", "sel.last")
            return "break"
        if kc == 65:                       # VK_A — выделить всё в поле
            self.entry.select_range(0, "end")
            self.entry.icursor("end")
            return "break"
        return None

    def _clear_placeholder(self, event=None):
        if self.entry.cget("fg") == "#8a8a8a":
            self.entry.delete(0, "end")
            self.entry.configure(fg="#e8e8e8")

    def _show_scrollbar(self):
        if not self.scroll_shown:
            self.scroll.grid(row=0, column=1, sticky="ns")
            self.scroll_shown = True

    # -- поведение текста -----------------------------------------------------

    def _block_edit(self, event):
        # Текст читаемый и выделяемый, но не редактируемый: гасим обычные
        # клавиши, пропуская сочетания с Control (Ctrl+C, Ctrl+A).
        if event.state & 0x4:
            return None
        if event.keysym in ("Left", "Right", "Up", "Down", "Home", "End",
                            "Prior", "Next", "Shift_L", "Shift_R"):
            return None
        return "break"

    def _on_ctrl_key(self, event):
        """Ctrl+C / Ctrl+A по keycode, а не по keysym.

        В Tk привязка <Control-c> сравнивает keysym, а на кириллической
        раскладке та же физическая клавиша даёт keysym Cyrillic_es. Из-за
        этого на русской раскладке не работают ни <Control-c>, ни встроенное
        <<Copy>> — а вопрос модели набирают как раз по-русски. keycode —
        это виртуальный код клавиши Windows, он от раскладки не зависит.
        """
        kc = event.keycode
        if kc == 67:                       # VK_C
            return self._copy_selection()
        if kc == 65:                       # VK_A
            return self._select_all()
        if kc == 45:                       # VK_INSERT — Ctrl+Ins тоже копирует
            return self._copy_selection()
        return None

    def _copy_selection(self, event=None):
        try:
            sel = self.txt.get("sel.first", "sel.last")
        except tk.TclError:
            sel = ""
        if sel:
            pyperclip.copy(sel)
        return "break"

    def _copy_all(self, event=None):
        pyperclip.copy(self.txt.get("1.0", "end-1c"))
        return "break"

    def _copy_last_answer(self, event=None):
        if self.last_answer:
            pyperclip.copy(self.last_answer)
            self._flash(self.copy_btn, "Скопировано")
        return "break"

    def _flash(self, widget, text, restore="Копировать ответ", ms=1200):
        """Короткая подсказка на кнопке вместо всплывающего окна."""
        widget.configure(text=text)
        widget.after(ms, lambda: widget.winfo_exists() and
                     widget.configure(text=restore))

    def _select_all(self, event=None):
        self.txt.tag_add("sel", "1.0", "end-1c")
        return "break"

    # -- контекстное меню -----------------------------------------------------

    def _build_context_menu(self):
        """Копирование мышью — путь, не зависящий ни от раскладки, ни от
        того, какой виджет держит фокус."""
        self.menu_open = False
        self.menu = tk.Menu(self.win, tearoff=0, bg="#252526", fg="#e8e8e8",
                            activebackground="#0e639c", activeforeground="#ffffff",
                            relief="flat")
        self.menu.add_command(label="Копировать", command=self._copy_selection)
        self.menu.add_command(label="Выделить всё", command=self._select_all)
        self.menu.add_separator()
        self.menu.add_command(label="Копировать всё", command=self._copy_all)
        self.txt.bind("<Button-3>", self._popup_menu)
        self.menu.bind("<Unmap>", lambda e: setattr(self, "menu_open", False))

    def _popup_menu(self, event):
        self.menu_open = True
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()
            self.win.after(400, lambda: setattr(self, "menu_open", False))
        return "break"

    # -- закрытие -------------------------------------------------------------

    def _on_escape(self, event=None):
        # Если в поле набран вопрос — Esc сначала чистит поле, а не закрывает окно.
        if self.entry is not None and self.entry.get().strip() and \
                self.entry.cget("fg") != "#8a8a8a":
            self.entry.delete(0, "end")
            return "break"
        self.close()

    def _on_focus_out(self, event=None):
        """Закрыть окно, только если фокус ушёл из приложения совсем.

        Привязка <FocusOut> на окне ловит и события дочерних виджетов: в Tk
        в bindtags виджета входит его toplevel. Клик по полю ввода — это
        FocusOut у Text, и по нему окно закрывалось. Фильтровать по
        event.widget нельзя: при уходе в другое приложение событие тоже
        приходит от дочернего виджета. Поэтому смотрим на фактический фокус
        через focus_get(): внутри приложения он вернёт виджет, снаружи — None.
        Проверка отложена, чтобы фокус успел установиться на новом виджете.
        """
        if not self.autoclose or self.pending or self.menu_open:
            return
        self.win.after(120, self._close_if_focus_lost)

    def _close_if_focus_lost(self):
        if not self.alive() or not self.autoclose or self.pending or self.menu_open:
            return
        try:
            if self.win.focus_get() is None:
                self.close()
        except (tk.TclError, KeyError):
            # focus_get() спотыкается, если фокус на чужом окне — это как раз
            # означает, что мы фокус потеряли.
            self.close()

    def close(self):
        try:
            self.win.destroy()
        except tk.TclError:
            pass

    def alive(self) -> bool:
        try:
            return bool(self.win.winfo_exists())
        except tk.TclError:
            return False

    # -- диалог ---------------------------------------------------------------

    def _append(self, text, tag):
        self.txt.insert("end", text, tag)
        self.txt.see("end")
        self._show_scrollbar()

    def _submit(self):
        if self.pending or self.entry is None:
            return
        question = self.entry.get().strip()
        if not question or self.entry.cget("fg") == "#8a8a8a":
            return

        # Первый вопрос переводит окно в режим диалога: оно больше не
        # закрывается само и получает нормальную высоту под ответы.
        self.autoclose = False
        if int(self.txt.cget("height")) < CHAT_MIN_ROWS:
            self.txt.configure(height=CHAT_MIN_ROWS)
        if int(self.txt.cget("width")) < CHAT_MIN_COLS:
            self.txt.configure(width=CHAT_MIN_COLS)

        self.entry.delete(0, "end")
        self.pending = True
        self.ask_btn.configure(state="disabled", text="Жду...")
        self._append(f"\n\n> {question}\n", "question")
        self._append("думаю...", "status")
        self.on_ask(self, question)

    def show_answer(self, answer: str):
        """Ответ модели. Вызывается только из главного потока."""
        self.pending = False
        self._drop_status()
        self._append(answer + "\n", "answer")
        self.last_answer = answer
        if self.copy_btn:
            self.copy_btn.configure(state="normal")
        self.history.append({"role": "assistant", "content": answer})
        # История режется, чтобы картинка + переписка не разрастались:
        # первое сообщение (с картинкой) сохраняем всегда.
        if len(self.history) > CHAT_HISTORY_LIMIT + 1:
            self.history[1:] = self.history[-CHAT_HISTORY_LIMIT:]
        if self.ask_btn:
            self.ask_btn.configure(state="normal", text="Спросить")
        self.entry.focus_set()

    def show_ask_error(self, message: str):
        self.pending = False
        self._drop_status()
        self._append(message + "\n", "status")
        # Неудачный вопрос из истории убираем, чтобы он не уехал в следующий
        # запрос без ответа.
        if self.history and self.history[-1].get("role") == "user":
            self.history.pop()
        if self.ask_btn:
            self.ask_btn.configure(state="normal", text="Спросить")

    def _drop_status(self):
        ranges = self.txt.tag_ranges("status")
        if ranges:
            self.txt.delete(ranges[-2], ranges[-1])


# -----------------------------------------------------------------------------
# Приложение
# -----------------------------------------------------------------------------

class OcrApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.events = queue.Queue()
        self.busy = None          # активен ли цикл захвата/распознавания
        self.busy_win = None
        self.last_b64 = None      # картинка последнего захвата — контекст вопросов
        self.root.after(50, self._pump)

    # -- очередь событий из фоновых потоков -----------------------------------

    def post(self, kind, payload=None):
        self.events.put((kind, payload))

    def _pump(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "hotkey":
                    self._start_capture()
                elif kind == "result":
                    self._on_result(payload)
                elif kind == "error":
                    self._on_error(payload)
                elif kind == "answer":
                    window, answer = payload
                    if window.alive():
                        window.show_answer(answer)
                elif kind == "answer_error":
                    window, message = payload
                    if window.alive():
                        window.show_ask_error(message)
        except queue.Empty:
            pass
        self.root.after(50, self._pump)

    # -- шаги сценария --------------------------------------------------------

    def _start_capture(self):
        if self.busy:
            return
        self.busy = True
        SelectionOverlay(self.root, self._on_selection)

    def _on_selection(self, rect):
        if rect is None:
            self.busy = False
            return
        # Пауза, чтобы оверлей успел исчезнуть с экрана до скриншота.
        self.root.after(120, lambda: self._capture_and_send(rect))

    def _capture_and_send(self, rect):
        left, top, width, height = rect
        try:
            img = grab_region(left, top, width, height)
            b64 = image_to_base64_png(img)
        except Exception as exc:
            self.busy = False
            self._on_error(f"Не удалось снять область экрана:\n\n{exc}")
            return

        self.last_b64 = b64
        self.busy_win = BusyWindow(self.root)
        threading.Thread(target=self._worker, args=(b64,), daemon=True).start()

    def _worker(self, b64):
        try:
            text = ocr_via_api(b64)
        except OcrError as exc:
            self.post("error", str(exc))
        except Exception as exc:                       # ничто не должно ронять скрипт
            self.post("error", f"Непредвиденная ошибка:\n\n{exc!r}")
        else:
            self.post("result", text)

    def _close_busy(self):
        if self.busy_win is not None:
            self.busy_win.close()
            self.busy_win = None

    def _on_result(self, text):
        self._close_busy()
        self.busy = False

        # История диалога: сообщение с картинкой + ответ модели. Дальше в неё
        # дописываются вопросы пользователя, картинка остаётся в контексте.
        history = build_ocr_messages(self.last_b64)
        history.append({"role": "assistant", "content": text})

        if not text.strip():
            ResultWindow(
                self.root,
                "Модель не нашла текста в выделенной области.\n"
                "Можно всё равно спросить что-нибудь по картинке.",
                title="OCR — пусто",
                history=history,
                on_ask=self._start_ask,
            )
            return

        try:
            pyperclip.copy(text)
        except Exception as exc:
            text = f"[Не удалось записать в буфер обмена: {exc}]\n\n{text}"

        ResultWindow(
            self.root,
            text,
            title="OCR — скопировано в буфер",
            history=history,
            on_ask=self._start_ask,
        )

    def _on_error(self, message):
        self._close_busy()
        self.busy = False
        # У окна ошибки контекста нет — строка вопроса не показывается.
        ResultWindow(self.root, message, title="OCR — ошибка")

    # -- вопросы по распознанной области --------------------------------------

    def _start_ask(self, window, question):
        window.history.append({"role": "user", "content": question})
        threading.Thread(
            target=self._ask_worker,
            args=(window, list(window.history)),
            daemon=True,
        ).start()

    def _ask_worker(self, window, history):
        try:
            answer = ask_via_api(history)
        except OcrError as exc:
            self.post("answer_error", (window, str(exc)))
        except Exception as exc:
            self.post("answer_error", (window, f"Непредвиденная ошибка:\n{exc!r}"))
        else:
            self.post("answer", (window, answer or "(пустой ответ)"))

    # -- запуск ---------------------------------------------------------------

    def run(self):
        listener = keyboard.GlobalHotKeys({HOTKEY: lambda: self.post("hotkey")})
        listener.daemon = True
        listener.start()

        print(f"OCR через RouterAI запущен. Хоткей: {HOTKEY}")
        print(f"Эндпоинт: {API_URL}   модель: {MODEL}")
        if not (os.environ.get(API_KEY_ENV) or API_KEY_FALLBACK).strip():
            print(f"ВНИМАНИЕ: ключ не найден — добавьте {API_KEY_ENV} в {ENV_FILE}")
        print("Ctrl+C в этом окне — выход.")

        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            pass
        finally:
            listener.stop()


if __name__ == "__main__":
    if sys.platform != "win32":
        print("Скрипт рассчитан на Windows.", file=sys.stderr)
    OcrApp().run()
