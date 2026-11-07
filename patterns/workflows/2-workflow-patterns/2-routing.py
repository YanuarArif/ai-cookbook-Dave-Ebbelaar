"""
================================================================================
POLA WORKFLOW LLM: ROUTING (PERCABANGAN TUGAS CERDAS)
================================================================================
File ini mendemonstrasikan pola "Routing" (perutean/pengalihan tugas) dalam sistem LLM.

Konsep Utama:
1. LLM Router: Model pertama bertindak sebagai "resepsionis / dispatcher" yang
   menganalisis niat pengguna dan mengklasifikasikan jenis permintaan (new_event,
   modify_event, atau other).
2. Specialized Handlers: Setelah kategori diketahui, permintaan diarahkan ke
   fungsi penanganan khusus (handle_new_event atau handle_modify_event).
3. Efisiensi & Fokus: Setiap handler memiliki instruksi (prompt) dan model data
   yang spesifik, membuat sistem lebih akurat dan terisolasi dengan rapi.
4. Multi-Provider: Otomatis mendeteksi DeepSeek API (deepseek-flash) atau OpenAI API (gpt-4o).
================================================================================
"""

# ==============================================================================
# BAGIAN 1: IMPORT LIBRARY (MODUL BAWAAN DAN PIHAK KETIGA)
# ==============================================================================

import json  # Untuk memanipulasi format data JSON
import logging  # Untuk mencatat riwayat eksekusi (log) ke terminal
import os  # Untuk membaca environment variables dari OS
import sys  # Untuk konfigurasi sistem dan encoding output terminal
from pathlib import Path  # Untuk manipulasi path folder/file secara lintas OS
from typing import (
    Literal,
    Optional,
    TypeVar,
)  # Literal (pilihan string tetap), Optional (bisa None), TypeVar (generik)

from dotenv import (
    find_dotenv,
    load_dotenv,
)  # Untuk membaca file rahasia/konfigurasi (.env)
from openai import OpenAI  # Client resmi untuk memanggil API LLM (DeepSeek / OpenAI)
from pydantic import (
    BaseModel,
    Field,
)  # Untuk validasi tipe data dan skema output terstruktur

# Menyetel encoding terminal agar aman menampilkan karakter UTF-8 di Windows:
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ==============================================================================
# BAGIAN 2: MEMUAT ENVIRONMENT VARIABLES (.env)
# ==============================================================================

# Mencari lokasi file .env di root project (naik 2 tingkat folder dari posisi file ini):
project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / ".env")
load_dotenv(find_dotenv())  # Cadangan jika file dieksekusi dari lokasi lain

# ==============================================================================
# BAGIAN 3: KONFIGURASI LOGGING
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,  # Level INFO menampilkan status proses yang sedang berjalan
    format="%(asctime)s - %(levelname)s - %(message)s",  # Format: Waktu - Level - Pesan
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ==============================================================================
# BAGIAN 4: DETEKSI PENYEDIA API (DEEPSEEK ATAU OPENAI)
# ==============================================================================

deepseek_key = os.getenv("DEEPSEEK_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

if deepseek_key:
    logger.info("Menggunakan provider: DeepSeek API (model: deepseek-flash)")
    client = OpenAI(
        api_key=deepseek_key,
        base_url="https://api.deepseek.com",  # Endpoint resmi DeepSeek
    )
    model = "deepseek-flash"  # Menggunakan model deepseek-flash sesuai permintaan
elif openai_key:
    logger.info("Menggunakan provider: OpenAI API (model: gpt-4o)")
    client = OpenAI(api_key=openai_key)
    model = "gpt-4o"
else:
    raise ValueError(
        "Tidak ditemukan API key! Harap isi DEEPSEEK_API_KEY atau OPENAI_API_KEY di file .env Anda."
    )

# ==============================================================================
# BAGIAN 5: DEFINISI MODEL DATA (PYDANTIC SCHEMAS)
# ==============================================================================


class CalendarRequestType(BaseModel):
    """Hasil LLM Router: Mengklasifikasikan jenis permintaan kalender."""

    # Literal membatasi nilai request_type HANYA boleh salah satu dari 3 pilihan berikut:
    request_type: Literal["new_event", "modify_event", "other"] = Field(
        description="Jenis permintaan: 'new_event' (buat baru), 'modify_event' (ubah jadwal lama), atau 'other' (bukan jadwal)"
    )
    confidence_score: float = Field(
        description="Skor keyakinan model antara 0.0 (tidak yakin) sampai 1.0 (sangat yakin)"
    )
    description: str = Field(
        description="Deskripsi bersih dan ringkas dari permintaan pengguna"
    )


class NewEventDetails(BaseModel):
    """Hasil Handler: Rincian untuk membuat acara kalender baru."""

    name: str = Field(description="Nama acara atau judul pertemuan")
    date: str = Field(
        description="Waktu dan tanggal acara dalam format ISO 8601 (contoh: 2026-11-10T14:00:00)"
    )
    duration_minutes: int = Field(
        description="Estimasi durasi pertemuan dalam satuan menit"
    )
    participants: list[str] = Field(
        description="Daftar nama orang-orang yang berpartisipasi"
    )


class Change(BaseModel):
    """Rincian perubahan spesifik untuk suatu bidang (field)."""

    field: str = Field(
        description="Nama bidang yang diubah (misal: 'date', 'time', 'location')"
    )
    new_value: str = Field(description="Nilai baru untuk bidang tersebut")


class ModifyEventDetails(BaseModel):
    """Hasil Handler: Rincian untuk mengubah acara kalender yang sudah ada."""

    event_identifier: str = Field(
        description="Nama atau deskripsi untuk mengidentifikasi acara lama yang ingin diubah"
    )
    changes: list[Change] = Field(
        default_factory=list, description="Daftar rincian perubahan yang diminta"
    )
    participants_to_add: list[str] = Field(
        default_factory=list, description="Daftar peserta baru yang ditambahkan"
    )
    participants_to_remove: list[str] = Field(
        default_factory=list, description="Daftar peserta yang dihapus dari acara"
    )


class CalendarResponse(BaseModel):
    """Format respon akhir yang dikembalikan ke pengguna."""

    success: bool = Field(
        description="Apakah operasi kalender berhasil diproses (True/False)"
    )
    message: str = Field(description="Pesan ramah untuk ditampilkan ke pengguna")
    calendar_link: Optional[str] = Field(
        default=None,
        description="Tautan (URL) ke aplikasi kalender jika tersedia",
    )


# ==============================================================================
# BAGIAN 6: FUNGSI BANTUAN UNTUK STRUCTURED OUTPUT (UNIVERSAL HELPER)
# ==============================================================================

T = TypeVar("T", bound=BaseModel)


def get_structured_completion(
    messages: list[dict],
    response_model: type[T],
) -> T:
    """Menjembatani pemanggilan output terstruktur antara DeepSeek dan OpenAI.

    - DeepSeek: Memerlukan mode `json_object`, schema diinjeksi ke system prompt,
      dan diparse menggunakan `response_model.model_validate_json()`.
    - OpenAI: Menggunakan endpoint bawaan `beta.chat.completions.parse()`.
    """
    is_deepseek = "deepseek.com" in str(client.base_url)

    if is_deepseek:
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        format_instruction = (
            f"\n\nYou must respond strictly with a valid JSON object matching this schema:\n"
            f"{schema_json}\n"
            f"Ensure the output contains only the valid json string without additional commentary."
        )

        formatted_messages = [dict(m) for m in messages]
        if formatted_messages and formatted_messages[0]["role"] == "system":
            formatted_messages[0]["content"] += format_instruction
        else:
            formatted_messages.insert(
                0, {"role": "system", "content": format_instruction}
            )

        completion = client.chat.completions.create(
            model=model,
            messages=formatted_messages,
            response_format={"type": "json_object"},
        )
        content = (completion.choices[0].message.content or "").strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return response_model.model_validate_json(content.strip())
    else:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_model,
        )
        return completion.choices[0].message.parsed


# ==============================================================================
# BAGIAN 7: FUNGSI ROUTER DAN PENANGAN KHUSUS (SPECIALIZED HANDLERS)
# ==============================================================================


def route_calendar_request(user_input: str) -> CalendarRequestType:
    """Fungsi Router: Menganalisis niat pengguna dan menentukan rute penanganan."""
    logger.info("Memulai router kalender untuk mengklasifikasikan niat pengguna...")

    result = get_structured_completion(
        messages=[
            {
                "role": "system",
                "content": "Determine if this is a request to create a new calendar event, modify an existing one, or other.",
            },
            {"role": "user", "content": user_input},
        ],
        response_model=CalendarRequestType,
    )
    logger.info(
        f"Rute ditentukan: '{result.request_type}' (Tingkat Keyakinan: {result.confidence_score:.2f})"
    )
    return result


def handle_new_event(description: str) -> CalendarResponse:
    """Handler Khusus 1: Memproses pembuatan acara kalender baru."""
    logger.info("Mengarahkan ke: Handler Acara Baru (New Event Handler)...")

    details = get_structured_completion(
        messages=[
            {
                "role": "system",
                "content": "Extract details for creating a new calendar event.",
            },
            {"role": "user", "content": description},
        ],
        response_model=NewEventDetails,
    )

    logger.info(f"Detail acara baru berhasil diparsing: {details.name}")

    # Membuat respon sukses pembuatan jadwal:
    peserta = (
        ", ".join(details.participants) if details.participants else "tanpa peserta"
    )
    return CalendarResponse(
        success=True,
        message=f"Berhasil membuat acara baru '{details.name}' pada {details.date} bersama {peserta}.",
        calendar_link=f"calendar://new?event={details.name}",
    )


def handle_modify_event(description: str) -> CalendarResponse:
    """Handler Khusus 2: Memproses perubahan jadwal acara yang sudah ada."""
    logger.info("Mengarahkan ke: Handler Perubahan Acara (Modify Event Handler)...")

    details = get_structured_completion(
        messages=[
            {
                "role": "system",
                "content": "Extract details for modifying an existing calendar event.",
            },
            {"role": "user", "content": description},
        ],
        response_model=ModifyEventDetails,
    )

    logger.info(
        f"Detail perubahan acara berhasil diparsing untuk acara: {details.event_identifier}"
    )

    # Membuat respon sukses perubahan jadwal:
    return CalendarResponse(
        success=True,
        message=f"Berhasil mengubah acara '{details.event_identifier}' sesuai perubahan yang diminta.",
        calendar_link=f"calendar://modify?event={details.event_identifier}",
    )


# ==============================================================================
# BAGIAN 8: ORKESTRASI WORKFLOW ROUTING
# ==============================================================================


def process_calendar_request(user_input: str) -> Optional[CalendarResponse]:
    """Fungsi Utama: Mengatur alur routing dari klasifikasi hingga eksekusi handler."""
    logger.info("--- [MEMULAI PROSES ROUTING KALENDER] ---")

    # 1. Jalankan LLM Router untuk mengidentifikasi jenis request:
    route_result = route_calendar_request(user_input)

    # 2. Gate Check: Jika model tidak yakin (< 0.7), jangan lanjutkan:
    if route_result.confidence_score < 0.7:
        logger.warning(
            f"Skor keyakinan router terlalu rendah: {route_result.confidence_score:.2f}"
        )
        return None

    # 3. Percabangan logika (Routing Logic) berdasarkan hasil klasifikasi:
    if route_result.request_type == "new_event":
        return handle_new_event(route_result.description)
    elif route_result.request_type == "modify_event":
        return handle_modify_event(route_result.description)
    else:
        logger.warning(
            f"Jenis permintaan '{route_result.request_type}' bukan operasi kalender yang didukung."
        )
        return None


# ==============================================================================
# BAGIAN 9: PENGUJIAN KODE (TEST RUNNER)
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SKENARIO PENGUJIAN 1: PERMINTAAN BUAT ACARA BARU (NEW EVENT)")
    print("=" * 70)

    input_baru = (
        "Let's schedule a team meeting next Tuesday at 2pm with Alice and Bob"
    )
    hasil_1 = process_calendar_request(input_baru)
    if hasil_1:
        print(f"[BERHASIL] Respon: {hasil_1.message}")
        if hasil_1.calendar_link:
            print(f"[INFO] Link: {hasil_1.calendar_link}")
    else:
        print("[GAGAL] Permintaan tidak dapat diproses.")

    print("\n" + "=" * 70)
    print("SKENARIO PENGUJIAN 2: PERMINTAAN UBAH JADWAL LAMA (MODIFY EVENT)")
    print("=" * 70)

    input_ubah = (
        "Can you move the team meeting with Alice and Bob to Wednesday at 3pm instead?"
    )
    hasil_2 = process_calendar_request(input_ubah)
    if hasil_2:
        print(f"[BERHASIL] Respon: {hasil_2.message}")
        if hasil_2.calendar_link:
            print(f"[INFO] Link: {hasil_2.calendar_link}")
    else:
        print("[GAGAL] Permintaan tidak dapat diproses.")

    print("\n" + "=" * 70)
    print("SKENARIO PENGUJIAN 3: PERMINTAAN DI LUAR KALENDER (OTHER / INVALID)")
    print("=" * 70)

    input_salah = "What's the weather like today?"
    hasil_3 = process_calendar_request(input_salah)
    if hasil_3:
        print(f"[BERHASIL] Respon: {hasil_3.message}")
    else:
        print(
            "[DITOLAK] Permintaan dikenali bukan sebagai operasi kalender (Rute 'other')."
        )
