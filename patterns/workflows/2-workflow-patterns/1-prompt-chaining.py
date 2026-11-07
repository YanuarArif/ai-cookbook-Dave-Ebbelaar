"""
================================================================================
POLA WORKFLOW LLM: PROMPT CHAINING DENGAN GATE CHECK
================================================================================
File ini mendemonstrasikan pola "Prompt Chaining" (merangkai prompt secara berurutan)
menggunakan Python murni (pure Python) dan Pydantic.

Konsep Utama:
1. Memecah tugas kompleks menjadi beberapa langkah LLM yang lebih kecil dan fokus.
2. Menggunakan "Gate Check" (filter kondisi) di antara langkah untuk memvalidasi
   apakah data layak diproses ke langkah berikutnya atau langsung dihentikan.
3. Mendukung multi-provider: otomatis mendeteksi DeepSeek API atau OpenAI API.
================================================================================
"""

# ==============================================================================
# BAGIAN 1: IMPORT LIBRARY (MODUL BAWAAN DAN PIHAK KETIGA)
# ==============================================================================

# Modul bawaan Python (Standard Library):
import json  # Untuk memanipulasi format data JSON (JavaScript Object Notation)
import logging  # Untuk mencatat riwayat eksekusi (log) ke terminal/layar
import os  # Untuk berinteraksi dengan sistem operasi (misal: membaca environment variable)
import sys  # Untuk konfigurasi sistem dan encoding output terminal
from datetime import datetime  # Untuk mengambil tanggal dan waktu saat ini
from pathlib import Path  # Untuk manipulasi path (lokasi folder/file) secara modern & lintas OS
from typing import (
    Optional,
    TypeVar,
)  # Type hint: Optional (bisa bernilai None), TypeVar (tipe data generik)

# Menyetel encoding terminal agar aman menampilkan karakter UTF-8 di Windows:
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Library pihak ketiga (diinstall via pip / uv):
from dotenv import (
    find_dotenv,
    load_dotenv,
)  # Untuk membaca file rahasia/konfigurasi (.env)
from openai import OpenAI  # Client resmi untuk memanggil API LLM (OpenAI / DeepSeek)
from pydantic import (
    BaseModel,
    Field,
)  # Untuk validasi tipe data dan struktur output LLM

# ==============================================================================
# BAGIAN 2: MEMUAT ENVIRONMENT VARIABLES (.env)
# ==============================================================================

# Mencari lokasi file .env:
# Path(__file__) adalah lokasi file ini saat ini.
# .resolve() mengubahnya menjadi path absolut lengkap.
# .parents[2] naik 2 tingkat folder ke folder utama (root project: ai-cookbook).
project_root = Path(__file__).resolve().parents[2]

# Memuat variabel rahasia seperti API KEY dari file .env ke memory sistem:
load_dotenv(project_root / ".env")
load_dotenv(find_dotenv())  # Cadangan jika file dijalankan dari folder lain

# ==============================================================================
# BAGIAN 3: KONFIGURASI LOGGING
# ==============================================================================

# Logging membantu kita melihat proses yang sedang berjalan di terminal dengan format yang rapi:
logging.basicConfig(
    level=logging.INFO,  # Menampilkan pesan level INFO, WARNING, dan ERROR
    format="%(asctime)s - %(levelname)s - %(message)s",  # Format: Waktu - Tipe - Pesan
    datefmt="%Y-%m-%d %H:%M:%S",  # Format tanggal: Tahun-Bulan-Hari Jam:Menit:Detik
)
# Membuat objek logger khusus untuk file ini:
logger = logging.getLogger(__name__)

# ==============================================================================
# BAGIAN 4: DETEKSI PENYEDIA API (DEEPSEEK ATAU OPENAI)
# ==============================================================================

# Mengambil nilai API key yang tersimpan di file .env:
deepseek_key = os.getenv("DEEPSEEK_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

# Logika pemilihan provider secara otomatis:
if deepseek_key:
    # Jika DEEPSEEK_API_KEY ditemukan, gunakan endpoint DeepSeek:
    logger.info("Menggunakan provider: DeepSeek API")
    client = OpenAI(
        api_key=deepseek_key,
        base_url="https://api.deepseek.com",  # Alamat server DeepSeek
    )
    model = "deepseek-chat"  # Model utama DeepSeek (DeepSeek-V3)
elif openai_key:
    # Jika OPENAI_API_KEY ditemukan, gunakan endpoint resmi OpenAI:
    logger.info("Menggunakan provider: OpenAI API")
    client = OpenAI(api_key=openai_key)
    model = "gpt-4o"  # Model utama OpenAI
else:
    # Jika kedua API key tidak ada, hentikan program dengan pesan peringatan:
    raise ValueError(
        "Tidak ditemukan API key! Harap isi DEEPSEEK_API_KEY atau OPENAI_API_KEY di file .env Anda."
    )

# ==============================================================================
# BAGIAN 5: DEFINISI STRUKTUR DATA (PYDANTIC MODELS)
# ==============================================================================
# Pydantic digunakan untuk memaksa LLM mengembalikan data dalam format JSON
# yang tipe datanya terjamin (misal: string, integer, boolean, atau list).


class EventExtraction(BaseModel):
    """Hasil LLM Tahap 1: Ekstraksi informasi dasar untuk memeriksa apakah input relevan."""

    description: str = Field(
        description="Deskripsi mentah dari kegiatan/event yang ingin dijadwalkan"
    )
    is_calendar_event: bool = Field(
        description="Apakah teks ini memang merupakan permintaan jadwal kalender (True/False)"
    )
    confidence_score: float = Field(
        description="Skor keyakinan model antara 0.0 (tidak yakin) sampai 1.0 (sangat yakin)"
    )


class EventDetails(BaseModel):
    """Hasil LLM Tahap 2: Parsing rincian detail jadwal kalender."""

    name: str = Field(description="Nama acara atau judul rapat")
    date: str = Field(
        description="Waktu dan tanggal acara dalam format standar ISO 8601 (contoh: 2026-11-10T14:00:00)"
    )
    duration_minutes: int = Field(
        description="Estimasi durasi pertemuan dalam satuan menit (contoh: 60)"
    )
    participants: list[str] = Field(
        description="Daftar nama orang-orang yang berpartisipasi/hadir"
    )


class EventConfirmation(BaseModel):
    """Hasil LLM Tahap 3: Menghasilkan pesan teks konfirmasi akhir yang ramah kepada pengguna."""

    confirmation_message: str = Field(
        description="Pesan konfirmasi ramah dalam bahasa manusia"
    )
    calendar_link: Optional[str] = Field(
        default=None,  # default=None penting agar tidak error jika link tidak digenerate
        description="Tautan (URL) Google Calendar jika tersedia/dibuat",
    )


# ==============================================================================
# BAGIAN 6: FUNGSI BANTUAN UNTUK STRUCTURED OUTPUT (UNIVERSAL HELPER)
# ==============================================================================

# TypeVar memungkinkan fungsi ini menerima model Pydantic apa saja (EventExtraction, EventDetails, dll.)
T = TypeVar("T", bound=BaseModel)


def get_structured_completion(
    messages: list[dict],
    response_model: type[T],
) -> T:
    """Fungsi pembantu yang menjembatani perbedaan antara DeepSeek dan OpenAI.

    - OpenAI: Mendukung mode bawaan `client.beta.chat.completions.parse`.
    - DeepSeek: Membutuhkan `response_format={'type': 'json_object'}`,
      schema JSON disuntikkan ke prompt, lalu diparsing dengan Pydantic.
    """
    # Mengecek apakah kita sedang terhubung ke DeepSeek:
    is_deepseek = "deepseek.com" in str(client.base_url)

    if is_deepseek:
        # 1. Ambil schema JSON dari class Pydantic:
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)

        # 2. Buat instruksi tambahan agar LLM mengembalikan format JSON yang sesuai:
        format_instruction = (
            f"\n\nYou must respond strictly with a valid JSON object matching this schema:\n"
            f"{schema_json}\n"
            f"Ensure the output contains only the valid json string without additional commentary."
        )

        # 3. Masukkan instruksi ke dalam system prompt:
        formatted_messages = [dict(m) for m in messages]
        if formatted_messages and formatted_messages[0]["role"] == "system":
            formatted_messages[0]["content"] += format_instruction
        else:
            formatted_messages.insert(
                0, {"role": "system", "content": format_instruction}
            )

        # 4. Panggil model menggunakan mode json_object:
        completion = client.chat.completions.create(
            model=model,
            messages=formatted_messages,
            response_format={"type": "json_object"},
        )

        # 5. Bersihkan teks respon dari pembungkus markdown (seperti ```json ... ```):
        content = (completion.choices[0].message.content or "").strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        # 6. Validasi teks JSON ke objek Pydantic dan kembalikan hasilnya:
        return response_model.model_validate_json(content.strip())
    else:
        # Jika menggunakan OpenAI asli, gunakan fitur resmi .parse():
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_model,
        )
        return completion.choices[0].message.parsed


# ==============================================================================
# BAGIAN 7: FUNGSI-FUNGSI LANGKAH PROMPT CHAIN (STEP FUNCTIONS)
# ==============================================================================


def extract_event_info(user_input: str) -> EventExtraction:
    """Langkah 1: Memeriksa apakah input pengguna benar-benar mengenai jadwal kalender."""
    logger.info("Memulai Langkah 1: Analisis ekstraksi event awal...")
    logger.debug(f"Input pengguna: {user_input}")

    # Mengetahui konteks hari dan tanggal sekarang (membantu LLM mengerti kata 'besok', 'selasa depan', dll.)
    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    # Panggil LLM dengan struktur EventExtraction:
    result = get_structured_completion(
        messages=[
            {
                "role": "system",
                "content": f"{date_context} Analyze if the text describes a calendar event.",
            },
            {"role": "user", "content": user_input},
        ],
        response_model=EventExtraction,
    )
    logger.info(
        f"Langkah 1 selesai -> Apakah event kalender: {result.is_calendar_event}, Tingkat Keyakinan: {result.confidence_score:.2f}"
    )
    return result


def parse_event_details(description: str) -> EventDetails:
    """Langkah 2: Mengekstrak detail spesifik (nama acara, tanggal ISO, durasi, partisipan)."""
    logger.info("Memulai Langkah 2: Parsing rincian detail event...")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    # Panggil LLM dengan struktur EventDetails:
    result = get_structured_completion(
        messages=[
            {
                "role": "system",
                "content": f"{date_context} Extract detailed event information. When dates reference 'next Tuesday' or similar relative dates, use this current date as reference.",
            },
            {"role": "user", "content": description},
        ],
        response_model=EventDetails,
    )
    logger.info(
        f"Langkah 2 selesai -> Judul: {result.name}, Waktu: {result.date}, Durasi: {result.duration_minutes} menit"
    )
    logger.debug(f"Partisipan: {', '.join(result.participants)}")
    return result


def generate_confirmation(event_details: EventDetails) -> EventConfirmation:
    """Langkah 3: Membuat pesan teks konfirmasi akhir yang sopan dari asisten virtual (Susie)."""
    logger.info("Memulai Langkah 3: Membuat pesan konfirmasi ramah...")

    # Panggil LLM dengan struktur EventConfirmation:
    result = get_structured_completion(
        messages=[
            {
                "role": "system",
                "content": "Generate a natural confirmation message for the event. Sign off with your name; Susie",
            },
            # event_details.model_dump() mengubah objek Pydantic menjadi dictionary Python:
            {"role": "user", "content": str(event_details.model_dump())},
        ],
        response_model=EventConfirmation,
    )
    logger.info("Langkah 3 selesai -> Pesan konfirmasi berhasil dibuat.")
    return result


# ==============================================================================
# BAGIAN 8: ORKESTRASI RANTAI PROMPT + GATE CHECK (FUNGSI UTAMA)
# ==============================================================================


def process_calendar_request(user_input: str) -> Optional[EventConfirmation]:
    """Menggabungkan ketiga fungsi di atas dalam satu alur (Chain).

    Juga menerapkan "Gate Check" (pintu penyaring) setelah langkah 1.
    """
    logger.info("--- [MULAI WORKFLOW KALENDER] ---")
    logger.debug(f"Teks input mentah: {user_input}")

    # 1. Jalankan LLM Pertama: Ekstraksi awal
    initial_extraction = extract_event_info(user_input)

    # --------------------------------------------------------------------------
    # GATE CHECK (Kondisi Penyaring Keamanan & Relevansi)
    # --------------------------------------------------------------------------
    # Jika teks BUKAN tentang kalender ATAU keyakinan model di bawah 0.7 (70%),
    # langsung tolak permintaan ini dan jangan buang token untuk Langkah 2 & 3.
    if (
        not initial_extraction.is_calendar_event
        or initial_extraction.confidence_score < 0.7
    ):
        logger.warning(
            f"Gate Check GAGAL: is_calendar_event={initial_extraction.is_calendar_event}, confidence={initial_extraction.confidence_score:.2f}"
        )
        # Kembalikan None menandakan input tidak valid sebagai event kalender
        return None

    logger.info(
        "Gate Check LOLOS! Input terbukti merupakan jadwal kalender yang valid."
    )

    # 2. Jalankan LLM Kedua: Ambil detail lengkap
    event_details = parse_event_details(initial_extraction.description)

    # 3. Jalankan LLM Ketiga: Buat pesan konfirmasi
    confirmation = generate_confirmation(event_details)

    logger.info("--- [WORKFLOW SELESAI DENGAN SUKSES] ---")
    return confirmation


# ==============================================================================
# BAGIAN 9: PENGUJIAN KODE (TEST RUNNER)
# ==============================================================================

# Blok 'if __name__ == "__main__":' adalah standar Python:
# Kode di dalam blok ini HANYA akan dijalankan jika file ini dieksekusi secara langsung
# (misal: python 1-prompt-chaining.py), dan TIDAK akan otomatis jalan jika di-import oleh file lain.
if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("SKENARIO PENGUJIAN 1: INPUT JADWAL RAPAT VALID (HARUS SUKSES)")
    print("=" * 70)

    # Contoh kalimat pengguna yang meminta penjadwalan meeting:
    input_valid = "Let's schedule a 1h team meeting next Tuesday at 2pm with Alice and Bob to discuss the project roadmap."

    hasil_1 = process_calendar_request(input_valid)

    # Mengecek apakah hasil ada (tidak None):
    if hasil_1:
        print("\n[BERHASIL] HASIL SUKSES:")
        print(f"Pesan Konfirmasi: {hasil_1.confirmation_message}")
        if hasil_1.calendar_link:
            print(f"Tautan Kalender: {hasil_1.calendar_link}")
    else:
        print("\n[DITOLAK] Permintaan ditolak (Bukan event kalender).")

    print("\n" + "=" * 70)
    print("SKENARIO PENGUJIAN 2: INPUT PERMINTAAN EMAIL (HARUS DITOLAK GATE CHECK)")
    print("=" * 70)

    # Contoh kalimat pengguna yang meminta kirim email (bukan jadwal kalender):
    input_invalid = (
        "Can you send an email to Alice and Bob to discuss the project roadmap?"
    )

    hasil_2 = process_calendar_request(input_invalid)

    # Mengecek apakah hasil ada (tidak None):
    if hasil_2:
        print("\n[BERHASIL] HASIL SUKSES:")
        print(f"Pesan Konfirmasi: {hasil_2.confirmation_message}")
        if hasil_2.calendar_link:
            print(f"Tautan Kalender: {hasil_2.calendar_link}")
    else:
        print("\n[DITOLAK] Permintaan ini ditolak oleh sistem.")
        print(
            "Penjelasan: Sistem mendeteksi bahwa ini adalah permintaan kirim email, bukan jadwal kalender."
        )
