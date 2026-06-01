"""Rule-based subject classifier для school questions.

Категории:
- math:     арифметика, алгебра, геометрия (содержит ops/числа/уравнения)
- russian:  разбор слов, падежи, склонения, части речи
- chemistry: формулы реакций, оксиды, кислоты
- physics:   плотность, сила, скорость, энергия
- biology:   клетка, фотосинтез, ген, организм
- history:   года, века, войны, императоры
- literature: автор, стих, роман, произведение
- english:    латиница доминирует в основной части
- geography:  страна, столица, континент
- other:      fallback
"""
import re

# Compile patterns once
_MATH_PAT = re.compile(r"[+\-*/=^]|\d.*[+\-*/=].*\d|\bурав|\bвычисл|\bнайти.*[xy]|\bx\s*=", re.IGNORECASE)
_RUS_PAT = re.compile(r"\b(разбор|падеж|склон|морфолог|морфем|фонет|части речи|корен[ьья]|окончан|приставк|суффикс|существительн|прилагательн|деепричаст|причаст|глагол.*вид|правописан)", re.IGNORECASE)
_CHEM_PAT = re.compile(r"NaOH|HCl|H2O|CO2|H2SO4|CaCO3|NaCl|NH3|[A-Z][a-z]?\d+|\b(реакц|кислот|оксид|щёлоч|основан\s+типа|соляной|серной|углерод)", re.IGNORECASE)
_PHYS_PAT = re.compile(r"\b(плотност|вместимост|скорост|ускорен|масс[ауы]|сил[ауыё]|энерги|давлен|температур|кпд|сопротивлен|напряжен\s*ток|трение|инерц|импульс|механич|тепловая)", re.IGNORECASE)
_BIO_PAT = re.compile(r"\b(клетк|организм|фотосинтез|митоз|мейоз|днк|рнк|хромосом|эволюци|популяци|размножен|питан|животн|растен|орган[ое]|нервн|кровь)", re.IGNORECASE)
_HIST_PAT = re.compile(r"\b(\d{3,4}\s*год|века|веке|войн|революци|царь|император|князь|правлен|реформ|восстан|сражен|битв|нашествие|крепост)", re.IGNORECASE)
_LIT_PAT = re.compile(r"\b(автор|поэт|писател|стих|строф|роман|повест|рассказ|произведен|герой|пушкин|толстой|чехов|есенин|тургенев|лермонтов|достоевск|булгаков|маяковск)", re.IGNORECASE)
_GEO_PAT = re.compile(r"\b(страна|столиц|материк|континент|океан|карт[еау]|географ|климат)", re.IGNORECASE)
_ENG_TOPIC_PAT = re.compile(r"\b(Present|Past|Future)\s+(Simple|Continuous|Perfect|Tense)|\bto\s+be\b|\barticle\b|артикл[ьея]|неправильн.*глагол", re.IGNORECASE)


def _is_english_dominant(q: str) -> bool:
    """Question is predominantly English (latin chars)."""
    letters = [c for c in q if c.isalpha()]
    if not letters:
        return False
    latin = sum(1 for c in letters if 0x41 <= ord(c) <= 0x7A)
    return latin / len(letters) > 0.6


def classify(question: str) -> str:
    q = question
    # English topic check (Russian question about English) — before math (no operators match)
    if _ENG_TOPIC_PAT.search(q):
        return "english"
    if _is_english_dominant(q):
        return "english"
    # Chemistry before math (формула содержит '+' но это химия)
    if _CHEM_PAT.search(q):
        return "chemistry"
    if _MATH_PAT.search(q):
        return "math"
    if _RUS_PAT.search(q):
        return "russian"
    if _PHYS_PAT.search(q):
        return "physics"
    if _BIO_PAT.search(q):
        return "biology"
    if _LIT_PAT.search(q):
        return "literature"
    if _HIST_PAT.search(q):
        return "history"
    if _GEO_PAT.search(q):
        return "geography"
    return "other"


if __name__ == "__main__":
    samples = [
        "3 8 2 + 5 7 4 =",
        "разбор слова весёлые по составу",
        "Какова вместимость канистры, если в неё можно налить 4,2 кг бензина?",
        "расставьте коэффициенты в уравнении химической реакции AlCl3 + NaOH",
        "Приведи 5 примеров использования глагола to be в Present Simple.",
        "Кто автор романа Война и мир?",
        "В каком году была Куликовская битва?",
        "Назови столицу Австралии",
        "Опиши процесс фотосинтеза кратко",
    ]
    for s in samples:
        print(f"{classify(s):>12}  |  {s[:80]}")
