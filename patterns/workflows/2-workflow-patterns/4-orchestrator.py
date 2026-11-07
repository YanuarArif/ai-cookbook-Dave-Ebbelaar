"""
================================================================================
POLA WORKFLOW LLM: ORCHESTRATOR-WORKERS (ORKESTRATOR & PEKERJA SPESIALIS)
================================================================================
File ini mendemonstrasikan pola "Orchestrator-Workers" (Arsitektur Orkestrator dan Pekerja).

Konsep Utama:
1. Orchestrator (Pemimpin Proyek):
   Menganalisis topik secara menyeluruh, menentukan target pembaca, dan memecah
   tugas besar (menulis artikel blog) menjadi sub-tugas kecil (sub-tasks/sections) yang terstruktur.
2. Workers (Pekerja Spesialis):
   Setiap section ditulis oleh worker LLM secara bergantian. Setiap worker diberikan
   konteks dari section yang telah ditulis sebelumnya agar narasi tetap nyambung (kohesif).
3. Reviewer (Penyunting Akhir):
   LLM ketiga bertindak sebagai editor/penyunting yang menilai skor kohesi (alur logika),
   memberikan saran revisi, dan menyusun artikel akhir yang matang dan terpoles.
4. Multi-Provider: Otomatis mendeteksi DeepSeek API (deepseek-flash) atau OpenAI API (gpt-4o-mini).
================================================================================
"""

# ==============================================================================
# BAGIAN 1: IMPORT LIBRARY
# ==============================================================================

import json  # Untuk manipulasi format JSON
import logging  # Untuk mencatat riwayat eksekusi (log) ke terminal
import os  # Untuk membaca environment variables dari OS
import sys  # Untuk konfigurasi sistem dan encoding output terminal
from pathlib import Path  # Untuk manipulasi path folder/file secara lintas OS
from typing import (
    Dict,
    List,
    TypeVar,  # Type hint: Dict (kamus data), List (daftar data), TypeVar (generik)
)

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

project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / ".env")
load_dotenv(find_dotenv())

# ==============================================================================
# BAGIAN 3: KONFIGURASI LOGGING
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
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
        base_url="https://api.deepseek.com",
    )
    model = "deepseek-flash"
elif openai_key:
    logger.info("Menggunakan provider: OpenAI API (model: gpt-4o-mini)")
    client = OpenAI(api_key=openai_key)
    model = "gpt-4o-mini"
else:
    raise ValueError(
        "Tidak ditemukan API key! Harap isi DEEPSEEK_API_KEY atau OPENAI_API_KEY di file .env Anda."
    )

# ==============================================================================
# BAGIAN 5: DEFINISI MODEL DATA (PYDANTIC SCHEMAS)
# ==============================================================================


class SubTask(BaseModel):
    """Sub-tugas penulisan bagian (section) blog yang dibuat oleh Orchestrator."""

    section_type: str = Field(
        description="Jenis bagian blog (misal: 'Introduction', 'Key Trends', 'Challenges', 'Conclusion')"
    )
    description: str = Field(
        description="Panduan pokok bahasan yang harus ditulis di bagian ini"
    )
    style_guide: str = Field(
        description="Panduan gaya penulisan untuk bagian ini"
    )
    target_length: int = Field(
        description="Target jumlah kata untuk bagian ini"
    )


class OrchestratorPlan(BaseModel):
    """Master Plan (rencana induk) struktur blog yang dirancang oleh Orchestrator."""

    topic_analysis: str = Field(
        description="Analisis mendalam mengenai topik artikel"
    )
    target_audience: str = Field(
        description="Sasaran pembaca dan tingkat pemahaman mereka"
    )
    sections: List[SubTask] = Field(
        description="Daftar rencana bagian-bagian tulisan yang akan dikerjakan oleh worker"
    )


class SectionContent(BaseModel):
    """Hasil karya tulisan yang dihasilkan oleh Worker LLM."""

    content: str = Field(
        description="Teks isi tulisan untuk bagian yang ditugaskan"
    )
    key_points: List[str] = Field(
        description="Poin-poin kunci atau ringkasan gagasan utama"
    )


class SuggestedEdits(BaseModel):
    """Saran penyuntingan untuk suatu bagian tulisan."""

    section_name: str = Field(description="Nama bagian yang diberikan masukan")
    suggested_edit: str = Field(
        description="Saran perbaikan atau penyesuaian kalimat"
    )


class ReviewFeedback(BaseModel):
    """Evaluasi akhir dan hasil tulisan yang sudah disempurnakan oleh Reviewer."""

    cohesion_score: float = Field(
        description="Skor kohesi/keterpaduan antar bagian (0.0 sampai 1.0)"
    )
    suggested_edits: List[SuggestedEdits] = Field(
        default_factory=list, description="Daftar saran perbaikan tiap bagian"
    )
    final_version: str = Field(
        description="Versi lengkap artikel blog yang telah disatukan dan dipoles dengan rapi"
    )


# ==============================================================================
# BAGIAN 6: FUNGSI BANTUAN UNTUK STRUCTURED OUTPUT (UNIVERSAL HELPER)
# ==============================================================================

T = TypeVar("T", bound=BaseModel)


def repair_and_parse_json(content: str, model_cls: type[T]) -> T:
    """Helper untuk membersihkan, memperbaiki JSON yang terpotong/memiliki newline kontrol,

    dan memvalidasinya ke model Pydantic.
    """
    s = content.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    s = s.strip()

    # 1. Coba parsing langsung dengan json.loads (strict=False mengizinkan unescaped control chars):
    try:
        data = json.loads(s, strict=False)
        return model_cls.model_validate(data)
    except Exception:
        pass

    # 2. Jika gagal karena terpotong token (EOF / unclosed string), perbaiki kutip & kurung kurawal:
    quotes = 0
    i = 0
    while i < len(s):
        if s[i] == '"' and (i == 0 or s[i - 1] != '\\'):
            quotes += 1
        i += 1
    if quotes % 2 != 0:
        s += '"'

    open_brackets = s.count("[") - s.count("]")
    if open_brackets > 0:
        s += "]" * open_brackets

    open_braces = s.count("{") - s.count("}")
    if open_braces > 0:
        s += "}" * open_braces

    try:
        data = json.loads(s, strict=False)
        return model_cls.model_validate(data)
    except Exception:
        return model_cls.model_validate_json(s)


def get_structured_completion(
    messages: list[dict],
    response_model: type[T],
) -> T:
    """Menjembatani structured output antara DeepSeek (json_object) dan OpenAI (parse)."""
    is_deepseek = "deepseek.com" in str(client.base_url)

    if is_deepseek:
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        format_instruction = (
            f"\n\nYou must respond strictly with a valid JSON object matching this schema:\n"
            f"{schema_json}\n"
            f"Ensure the output contains only the valid json string without additional commentary. "
            f"Keep string values properly escaped."
        )

        formatted_messages = [dict(m) for m in messages]
        if formatted_messages and formatted_messages[0]["role"] == "system":
            formatted_messages[0]["content"] += format_instruction
        else:
            formatted_messages.insert(
                0, {"role": "system", "content": format_instruction}
            )

        # DeepSeek memerlukan pesan 'user' agar tidak mengembalikan respon kosong:
        has_user = any(m.get("role") == "user" for m in formatted_messages)
        if not has_user:
            formatted_messages.append(
                {
                    "role": "user",
                    "content": "Please provide the requested output in valid json format.",
                }
            )

        completion = client.chat.completions.create(
            model=model,
            messages=formatted_messages,
            response_format={"type": "json_object"},
            max_tokens=4096,
        )
        content = (completion.choices[0].message.content or "").strip()
        return repair_and_parse_json(content, response_model)
    else:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_model,
        )
        return completion.choices[0].message.parsed


# ==============================================================================
# BAGIAN 7: TEMPLATE PROMPT UNTUK SETIAP PERAN
# ==============================================================================

ORCHESTRATOR_PROMPT = """
You are a senior editorial director and content strategist.
Analyze this blog topic and break it down into 3-4 logical, cohesive sections.

Topic: {topic}
Target Total Length: {target_length} words
Target Writing Style: {style}

Plan the article structure so that each section flows smoothly into the next.
"""

WORKER_PROMPT = """
You are a specialized content writer.
Write a high-quality blog section based on the following instructions:

Topic: {topic}
Section Type: {section_type}
Section Goal: {description}
Style Guide: {style_guide}
Target Length: {target_length} words

Previous Context:
{previous_sections}

Write engaging, clear content for this section while maintaining consistency with previous sections.
"""

REVIEWER_PROMPT = """
You are a chief editor. Review this multi-section blog draft for overall cohesion, narrative flow, and tone consistency:

Topic: {topic}
Target Audience: {audience}

Full Draft:
{sections}

Evaluate how well the sections connect with each other. Provide:
1. A cohesion score between 0.0 and 1.0 (1.0 = flawless flow).
2. Actionable suggested edits by section.
3. A polished, publication-ready final version that harmonizes all sections into one complete article.
Keep the final version concise and focused (around 400-500 words total) to fit within token limits.
"""

# ==============================================================================
# BAGIAN 8: IMPLEMENTASI KELAS BLOG ORCHESTRATOR
# ==============================================================================


class BlogOrchestrator:
    """Mengelola alur kolaborasi antara peran Orchestrator, Workers, dan Reviewer."""

    def __init__(self):
        # Dictionary untuk menyimpan hasil tulisan tiap section: {section_type: SectionContent}
        self.sections_content: Dict[str, SectionContent] = {}

    def get_plan(
        self, topic: str, target_length: int, style: str
    ) -> OrchestratorPlan:
        """Langkah 1 (Orchestrator): Menyusun rencana struktur dan pembagian bab/section."""
        logger.info(
            "[Orchestrator] Merancang master plan dan memecah topik tulisan..."
        )

        return get_structured_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior editorial director and content strategist. Plan the blog post sections in valid json format.",
                },
                {
                    "role": "user",
                    "content": ORCHESTRATOR_PROMPT.format(
                        topic=topic,
                        target_length=target_length,
                        style=style,
                    ),
                },
            ],
            response_model=OrchestratorPlan,
        )

    def write_section(self, topic: str, section: SubTask) -> SectionContent:
        """Langkah 2 (Worker): Menulis bab tertentu dengan menyertakan konteks bab sebelumnya."""
        # Menggabungkan teks bab-bab yang telah selesai ditulis sebelumnya sebagai konteks:
        previous_sections = "\n\n".join(
            [
                f"=== {sec_type} ===\n{sec_data.content}"
                for sec_type, sec_data in self.sections_content.items()
            ]
        )

        prompt_content = WORKER_PROMPT.format(
            topic=topic,
            section_type=section.section_type,
            description=section.description,
            style_guide=section.style_guide,
            target_length=section.target_length,
            previous_sections=previous_sections
            if previous_sections
            else "This is the first section of the blog post.",
        )

        return get_structured_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a specialized content writer. Write the section content in valid json format.",
                },
                {"role": "user", "content": prompt_content},
            ],
            response_model=SectionContent,
        )

    def review_post(
        self, topic: str, plan: OrchestratorPlan
    ) -> ReviewFeedback:
        """Langkah 3 (Reviewer): Menyunting keseluruhan naskah dan menghasilkan versi final."""
        # Mengumpulkan semua section menjadi satu naskah lengkap:
        full_sections_text = "\n\n".join(
            [
                f"=== {sec_type} ===\n{sec_data.content}"
                for sec_type, sec_data in self.sections_content.items()
            ]
        )

        prompt_content = REVIEWER_PROMPT.format(
            topic=topic,
            audience=plan.target_audience,
            sections=full_sections_text,
        )

        return get_structured_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a chief editor reviewing a multi-section draft. Provide feedback and the final version in valid json format.",
                },
                {"role": "user", "content": prompt_content},
            ],
            response_model=ReviewFeedback,
        )

    def write_blog(
        self, topic: str, target_length: int = 1000, style: str = "informative"
    ) -> Dict:
        """Mengatur orkestrasi keseluruhan dari perancangan hingga penyuntingan akhir."""
        logger.info(f"--- [MEMULAI ORKESTRASI PENULISAN BLOG: '{topic}'] ---")

        # 1. Orchestrator membuat perencanaan:
        plan = self.get_plan(topic, target_length, style)
        logger.info(
            f"[Plan Dibuat] Jumlah section: {len(plan.sections)}, Sasaran: {plan.target_audience}"
        )

        # 2. Worker menulis setiap section satu per satu:
        for idx, section in enumerate(plan.sections, start=1):
            logger.info(
                f"[Worker #{idx}] Menulis section: '{section.section_type}'..."
            )
            content = self.write_section(topic, section)
            self.sections_content[section.section_type] = content
            logger.info(
                f"[Worker #{idx} Selesai] Poin utama: {', '.join(content.key_points)}"
            )

        # 3. Reviewer mengevaluasi dan merapikan tulisan:
        logger.info(
            "[Reviewer] Memulai penyuntingan akhir dan evaluasi kohesi..."
        )
        review = self.review_post(topic, plan)
        logger.info(
            f"[Review Selesai] Skor Kohesi Tulisan: {review.cohesion_score:.2f}/1.00"
        )

        return {"structure": plan, "sections": self.sections_content, "review": review}


# ==============================================================================
# BAGIAN 9: PENGUJIAN KODE (RUNNER)
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SKENARIO PENGUJIAN: ORKESTRASI ARTIKEL BLOG DENGAN DEEPSEEK-FLASH")
    print("=" * 70)

    # Inisialisasi sistem orkestrator:
    orchestrator = BlogOrchestrator()

    # Contoh topik artikel teknis:
    topic_blog = "The impact of AI on software development"

    # Jalankan proses pembuatan artikel (target_length 500 kata agar ringkas, padat, dan aman dari batas token):
    hasil = orchestrator.write_blog(
        topic=topic_blog,
        target_length=500,
        style="technical but accessible and engaging",
    )

    print("\n" + "=" * 70)
    print("HASIL AKHIR ARTIKEL BLOG (FINAL POLISHED VERSION):")
    print("=" * 70)
    print(hasil["review"].final_version)

    print("\n" + "-" * 70)
    print(
        f"[INFO] Skor Keterpaduan Tulisan (Cohesion Score): {hasil['review'].cohesion_score:.2f} / 1.00"
    )

    if hasil["review"].suggested_edits:
        print("\n[INFO] Catatan Saran Penyuntingan (Suggested Edits):")
        for edit in hasil["review"].suggested_edits:
            print(f"- Bagian '{edit.section_name}': {edit.suggested_edit}")
