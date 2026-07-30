# Silent Guardian — Ministral 3 lokalno

Gostovanje in fino uglaševanje modela **Ministral 3** (Apache 2.0) za uporabo kot
krmilnik/orkestrator pametnega doma, ki kliče orodja.

Projekt pokriva dvoje:

1. **Gostovanje** — model teče lokalno, dostopen prek OpenAI-združljivega vmesnika.
2. **Učenje** — cevovod LoRA/QLoRA za klicanje orodij, z gradnikom in **validatorjem**
   učne množice.

> **Kaj ta projekt ni.** Ne vsebuje povezav do dejanskih naprav. Funkcije za GPIO,
> SSH, MQTT in robotskega psa so prazne predloge (`tools/registry.json`) — sheme so
> definirane, izvedba je tvoja. Prav tako projekt ne vsebuje izdelane učne množice:
> priloženih je 10 primerov, ki prikazujejo *obliko*, ne pa vsebine za resno učenje.

---

## Preden začneš: kaj fino uglaševanje sploh naredi

To je najpomembnejši odstavek v dokumentu.

Fino uglaševanje modela **ne nauči novih zmožnosti in ne novih dejstev.** Ne bo se
naučilo, katere naprave imaš doma. Naprave se model nauči *ob vsakem klicu*, iz
opisov orodij, ki mu jih pošlješ v kontekstu (`tools/registry.json`).

To je tudi razlog, zakaj registra ne smeš »zapeči« v uteži: ko dodaš nov Raspberry Pi,
ga v register vpišeš in model ga takoj zna uporabljati. Če bi bile naprave naučene v
utežeh, bi moral po vsaki novi napravi model ponovno učiti.

**Kaj fino uglaševanje torej prinese:** disciplino pri obliki. Majhen model pri
4-bitni kvantizaciji in dolgem seznamu orodij zna zaiti — pokvarjen JSON, izmišljeni
parametri, klic orodja tam, kjer bi moral vprašati za manjkajoč podatek. Učenje na
tvojih lastnih zapisih to stabilizira.

Skratka: *modela ne učiš, da bi poznal tvojo hišo — učiš ga, da brez napak govori
tvoj protokol.*

---

## Strojne zahteve

| | Priporočeno |
|---|---|
| Gostovanje 8B | 8 GB VRAM (4-bit), kontekst 8–16k |
| Gostovanje 3B | 6 GB VRAM ali Apple Silicon |
| Učenje 3B (QLoRA) | 8 GB VRAM — udobno |
| Učenje 8B (QLoRA) | 8 GB VRAM — tesno: `max_seq_len` ≤ 2048, `batch_size: 1` |

Model podpira kontekst 256k, kar pa na 8 GB kartici ni uporabno — pomnilnik poje
predpomnilnik KV. Računaj z 8–16k.

**Priporočena kombinacija:** uči **3B**, gosti **8B**. Če ti je vseeno za zadnji
odstotek kakovosti, ostani pri 3B za oboje — cela zanka je hitrejša.

---

## Namestitev

```bash
git clone https://github.com/nouveau-rubynho/host_and_train.git
cd host_and_train

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

**Torch namesti glede na strojno opremo** (ne iz `requirements.txt`):

```bash
# NVIDIA RTX 50xx (Blackwell) — OBVEZNO cu128 ali novejši
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install "bitsandbytes>=0.45.0"

# Apple Silicon
pip install torch
```

Nato ostalo:

```bash
pip install -r requirements.txt
python check_env.py
```

`check_env.py` zaženi **prvi**. Večina začetnih težav se pokaže tam kot jasno
sporočilo namesto kasneje kot nerazumljivo sesutje.

> **RTX 5050 in ostale 50xx:** to so kartice arhitekture Blackwell (`sm_120`).
> Starejši torch zanje nima jeder in pade z `no kernel image is available for
> execution on the device`. Napaka izgleda, kot da je narobe ta projekt — ni.
> `check_env.py` to preveri in ti pove točno, kaj namestiti.

---

## Nastavitve

Vse je v `config.yaml`. Kode ni treba spreminjati.

```yaml
model:
  id: mistralai/Ministral-3-3B-Instruct-2512
  device: auto          # auto | cuda | mps | cpu
  precision: bf16       # bf16 na Apple Silicon, 4bit na NVIDIA (QLoRA)
  adapter: null         # pot do naučenega adapterja
```

`precision: 4bit` zahteva bitsandbytes in torej NVIDIA. Na Apple Silicon uporabi
`bf16` — skripte so iste, spremeni se samo ta vrstica.

---

## 1. del — gostovanje

### Pogovor v terminalu

```bash
python -m src.chat --groups rpi
python -m src.chat --groups rpi --prompt "Prizgi luc v garazi."
```

Naprave niso priključene, zato rezultat orodja vtipkaš sam (ali pritisneš Enter za
`{"ok": true}`). Tako vidiš celo zanko: klic → rezultat → povzetek.

### Strežnik

```bash
python -m src.serve
```

Odpre OpenAI-združljiv `POST /v1/chat/completions` na `http://127.0.0.1:8000`
(dokumentacija na `/docs`).

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "tool_groups": ["rpi"],
        "messages": [
          {"role": "system", "content": "Si Silent Guardian, krmilnik pametnega doma."},
          {"role": "user", "content": "Prizgi luc v garazi."}
        ]
      }'
```

Poleg standardnega `tools` sprejme tudi `tool_groups`, kar je bližnjica: strežnik
sam vzame ustrezna orodja iz registra.

### Ollama

Za igranje z **osnovnim** modelom je Ollama najhitrejša pot:

```bash
ollama run ministral-3:8b
```

Za **naučen** model pa Ollame ne uporabljaj. Zahtevala bi zlitje adapterja in
pretvorbo v GGUF ob vsaki spremembi. `src/serve.py` naloži adapter neposredno
zraven osnovnega modela — brez pretvorbe, zamenjaš ga z eno vrstico v `config.yaml`.

### vLLM (za NVIDIA, ko gre v resno uporabo)

vLLM zna adapterje menjati med tekom, brez ponovnega nalaganja osnovnega modela:

```bash
pip install vllm
vllm serve mistralai/Ministral-3-8B-Instruct-2512 \
  --enable-lora \
  --lora-modules guardian=adapters/silent-guardian-v1 \
  --max-model-len 16384
```

`src/serve.py` je namenjen razvoju in preizkušanju; vLLM je hitrejši in zdrži več
hkratnih zahtevkov.

---

## 2. del — učenje

### Register orodij

`tools/registry.json` je edino mesto, kjer so naprave opisane. Novo napravo dodaš
tam — v Pythonu ni treba spremeniti ničesar.

Imena so oblike `<skupina>.<naprava>.<akcija>`, npr. `rpi.gpio_set`. Polje `group`
omogoča, da modelu pošlješ samo relevantno podmnožico orodij.

> **Proračun orodij.** Natančnost klicanja orodij pri majhnem 4-bitnem modelu pada,
> ko seznam raste. Računaj z **10–15 orodji naenkrat**. Če jih boš imel več,
> usmerjaj po skupinah (`tool_groups`) namesto da vsakič pošlješ vsa.

### Oblika učnih podatkov

Ena vrstica JSON na primer (`data/seed.jsonl`), v OpenAI obliki sporočil:

```json
{
  "tool_groups": ["rpi"],
  "messages": [
    {"role": "system", "content": "Si Silent Guardian ..."},
    {"role": "user", "content": "Prizgi luc v garazi."},
    {"role": "assistant", "tool_calls": [
      {"id": "a1b2c3d4e", "type": "function",
       "function": {"name": "rpi.gpio_set",
                    "arguments": {"host": "garaza", "pin": 17, "state": "high"}}}]},
    {"role": "tool", "name": "rpi.gpio_set", "tool_call_id": "a1b2c3d4e",
     "content": "{\"ok\": true}"},
    {"role": "assistant", "content": "Luc v garazi je prizgana."}
  ]
}
```

Pravila:

- zadnje sporočilo mora biti **assistant** (sicer primer nima česa učiti);
- vsak `tool_call` mora imeti pripadajoč odgovor z istim `tool_call_id`;
- `tool_call_id` naj bo **9 alfanumeričnih znakov** (nekatere Mistralove predloge
  to zahtevajo);
- orodje mora obstajati v registru **in** biti med ponujenimi v tem primeru;
- namesto `tool_groups` lahko podaš `tools` neposredno.

### Preveri, preden učiš

```bash
python -m src.validate
```

Ne potrebuje ne modela ne interneta. Preveri sheme, obvezne parametre, tipe,
`enum` vrednosti, ujemanje `tool_call_id` in nepovezane klice orodij.

**To zaženi vsakič, ko dodaš primere.** Napačna učna množica se ne pritoži — model
se tiho nauči narobe in to opaziš šele čez pol ure učenja.

### Poglej masko z očmi

```bash
python -m src.dataset stats
python -m src.dataset inspect --index 0
```

`inspect` izpiše primer tako, da je maska vidna: **zeleno** so tokeni, ki jih model
uči generirati, **sivo** je samo kontekst. Zeleno mora biti *samo* to, kar bi model
napisal sam — klici orodij in njegovi odgovori. Nikoli sistemski poziv, uporabnikov
vnos ali odgovor orodja.

To je edina zanesljiva kontrola maskiranja. Če je zeleno kaj drugega, se bo model
učil oponašati uporabnika ali izmišljati rezultate senzorjev.

Za priložene primere je v izgubi ~11 % tokenov — velika večina konteksta so opisi
orodij, kar je pričakovano.

### Učenje

```bash
python -m src.train
```

Adapter se shrani v `adapters/silent-guardian-v1` skupaj z `training_meta.json`
(kateri osnovni model, koliko primerov, koliko epoh — čez mesec dni tega ne boš
vedel na pamet).

Uporabiš ga tako, da v `config.yaml` nastaviš:

```yaml
model:
  adapter: adapters/silent-guardian-v1
```

Nato spet `python -m src.chat --groups rpi` in primerjaj z osnovnim modelom.

Nastavitve so v `config.yaml` pod `train:`. Pri majhni množici je `epochs: 8`
smiselno; pri nekaj sto primerih pojdi na 2–3, sicer se model nauči primerov na
pamet.

---

## Podatki iz uporabe (najpomembnejši del za dolgi rok)

Strežnik vsak pogovor zapiše v `logs/traces.jsonl` — **v isti shemi, kot jo bere
trenažer.**

To pomeni, da sistem sam proizvaja svoje učne podatke. Deset ročno napisanih
primerov ne bo naredilo dobrega modela; nekaj sto resničnih pogovorov iz tvoje hiše
pa lahko. Postopek:

1. nekaj tednov uporabljaj sistem (`log_traces: true`, privzeto vklopljeno);
2. preglej `logs/traces.jsonl` in **popravi** zapise, kjer je model zgrešil —
   popravljena napaka je vredna veliko več kot pravilen primer;
3. popravljene vrstice prilepi v `data/seed.jsonl`;
4. `python -m src.validate`, nato `python -m src.train`.

Tako se nadzor in razvoj sistema zapreta v zanko: uporaba → zapisi → boljši model.

---

## Slike

Ministral 3 zna brati slike, **ta cevovod pa slik še ne uči** (`vision.enabled` je
`false` in nima učinka).

Kar bi bilo treba dodati:

- `AutoProcessor` namesto samega tokenizatorja (ustvari `pixel_values`);
- sporočila kot seznam delov (`{"type": "image"}`, `{"type": "text"}`) namesto
  navadnega niza;
- **maskiranje slikovnih nadomestnih tokenov** iz izgube — če se to spregleda, se
  model uči na smeteh in to ni vidno v nobeni metriki;
- zbiralnik za spremenljivo število slik na primer.

Nasvet: najprej spravi v pogon besedilno zanko od konca do konca. Fino uglaševanje
vizualno-jezikovnega modela je bistveno težji projekt in je običajno mesto, kjer se
takšni poskusi ustavijo.

---

## Ko kaj ne dela

| Napaka | Vzrok |
|---|---|
| `no kernel image is available` | prestar torch za tvojo kartico → `check_env.py` |
| `precision: 4bit zahteva CUDA` | bitsandbytes na Apple Silicon ne dela → `bf16` |
| `predloga za klepet ni stabilna po predponah` | neujemanje različice `transformers` in predloge modela — posodobi `transformers` |
| `zavrzenih N primerov` | primeri daljši od `max_seq_len` → povečaj ali skrajšaj |
| `maska je prazna` | primer se ne konča z zavojem asistenta |
| model kliče neobstoječa orodja | premalo primerov, kjer *ne* pokliče orodja (glej primera 2 in 7) |
| model izmišlja parametre | dodaj primere z manjkajočimi podatki, kjer asistent vpraša |
| zmanjkalo pomnilnika med učenjem | `batch_size: 1`, `grad_accum` višje, `max_seq_len` nižje, manjši model |

---

## Struktura

```
config.yaml            vse nastavitve
check_env.py           preverjanje okolja — zaženi prvega
tools/registry.json    naprave in orodja (tu dodajaš)
data/seed.jsonl        10 primerov za prikaz oblike
src/common.py          konfiguracija, register, nalaganje modela
src/dataset.py         predloga za klepet, maskiranje izgube, zbiralnik
src/validate.py        preverjanje podatkov brez modela
src/train.py           učenje LoRA / QLoRA
src/serve.py           OpenAI-združljiv strežnik + zapisovanje sledi
src/chat.py            pogovor v terminalu
logs/traces.jsonl      zapisi uporabe (vir bodočih učnih podatkov)
```

## Preverjeno na

Celotna zanka (učenje → nalaganje adapterja → klic orodja → strežnik) je bila
pognana od konca do konca na:

- MacBook M1, 16 GB, `mps`, `precision: bf16`
- Ministral-3-3B-Instruct-2512, torch 2.13, transformers 5.14, peft 0.20
- 10 primerov, 8 epoh, LoRA r=16 → 24,7 M učljivih parametrov (0,64 %)
- **~35 minut**, izguba 1,07 → 0,003

Toliko epoh na desetih primerih pomeni, da se jih model nauči na pamet. To je za
prikaz mehanike v redu, za uporabo pa ne — glej razdelek o podatkih iz uporabe.

Na RTX 5050 s `precision: 4bit` bo učenje bistveno hitrejše.

## Licenca

Ministral 3 je pod **Apache 2.0** — dovoljena je tudi komercialna uporaba.

Pozor na zamenjavo: starejši **Ministral-8B-Instruct-2410** (2024) je pod *Mistral
Research License* in za komercialno uporabo zahteva ločeno pogodbo z Mistralom. Ta
projekt uporablja družino **Ministral 3** (`-2512`).
