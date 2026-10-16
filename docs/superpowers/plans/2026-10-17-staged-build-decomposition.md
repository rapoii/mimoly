# Staged Decomposition for Ambiguous Build Prompts — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Permintaan pembangunan yang ambigu/besar ("bikinin hp", "bikinin aplikasi pantau harga...") tidak lagi menggantung upstream sampai deadline — gateway menulis ulang query menjadi tahap rencana-dulu agar turn pertama selalu kembali dengan jawaban.

**Architecture:** Tambah fungsi heuristik murni `_ambiguous_build_request(text)` yang mendeteksi kata kerja pembuatan + objek samar/lingkup multi-komponen tanpa sinyal konkret (kode/nama file/URL/path). Saat terdeteksi DAN sesi agentik (tools present), `query` upstream ditulis ulang menjadi prompt dekomposisi bertahap sebelum `upstream_payload` dibangun. Non-ambiguous tidak tersentuh.

**Tech Stack:** Python, `mimoly.py`, pytest offline (`test_agent_responsiveness.py` harness `wire_client`).

**Spec:** Eksperimen live 2026-10-16 — "bikinin hp" hang 313 dtk baru selesai; prompt PantauHarga full tidak pernah first-emission dalam 600 dtk; prompt yang sama dipecah kecil ("Sebutkan 3 komponen...") lolos 8 dtk; prompt non-build (rendang, lorem 22k, "jawab OK") selalu lolos. Pemicu = reasoning upstream berputar pada permintaan besar/ambigu, bukan filter kata dan bukan ukuran.

## Global Constraints

- Offline only, tanpa live call; replay memakai `wire_client` persis bentuk live.
- Commit scoped hanya `mimoly.py` + `test_agent_responsiveness.py`.
- Frasa busy sejati (服务器繁忙 dsb.) tetap retry seperti sekarang.
- Tanpa ubah logika cancel/timeout/retry yang ada.

## Review Focus

- Prompt ambigu TANPA tools (plain chat) harus tetap diteruskan apa adanya — rewrite hanya untuk sesi agentik.
- Prompt non-build dengan kata kerja pembuatan ("buatkan puisi") tidak boleh kena rewrite.
- Rewrite tidak boleh menelan permintaan user asli — user goal tetap di query.
- Retry/busy path tidak berubah untuk prompt yang sudah di-rewrite.
- Satu attempt untuk replay deterministik (budget `_request_budget_seconds` mock).

---

### Task 1: Heuristik `_ambiguous_build_request(text) -> bool`

**Files:**
- Modify: `mimoly.py` (dekat `is_upstream_busy`, ~baris 242-246)
- Test: `test_agent_responsiveness.py` (unit test langsung, tanpa wire)

**Interfaces:**
- Consumes: tidak ada (fungsi murni, input `str`, output `bool`).
- Produces: `_ambiguous_build_request(text: str) -> bool` dipakai Task 2.

Aturan (lock):
- True jika ada kata kerja pembuatan (ID: bikin/bikinin/buatkan/buat...; EN: build/create/make...) DAN (tidak ada sinyal konkret: blok kode, nama file berekstensi, URL, path) DAN (objek samar seperti "hp"/"aplikasi" saja ATAU lingkup multi-komponen: "dashboard" + "disimpen"/"database"/"naik turun").
- False untuk: "buatkan puisi tentang hujan", "tulis kode python tambah(a, b)", "Sebutkan 3 bahan rendang", "jawab persis: OK".

- [ ] **Step 1: Write the failing test**

```python
def test_ambiguous_build_request_heuristic(self):
    self.assertTrue(m._ambiguous_build_request("bikinin hp"))
    self.assertTrue(m._ambiguous_build_request(
        "bang bikinin aku aplikasi buat mantau harga hp bekas iphone 13 dong. "
        "yang ada dashboardnya bisa diliat di browser, datanya disimpen biar "
        "bisa liat naik turunnya."))
    self.assertFalse(m._ambiguous_build_request("Tolong buatkan saya puisi singkat tentang hujan."))
    self.assertFalse(m._ambiguous_build_request("tulis kode python tambah(a, b)"))
    self.assertFalse(m._ambiguous_build_request("Sebutkan 3 bahan utama rendang."))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/c/Users/Rafi/AppData/Local/hermes/hermes-agent/venv/Scripts/python -m pytest test_agent_responsiveness.py -q -k ambiguous_build`
Expected: FAIL dengan `AttributeError` (fungsi belum ada)

- [ ] **Step 3: Implement `_ambiguous_build_request(text: str) -> bool` di `mimoly.py` dekat `is_upstream_busy`**

Regex kata kerja ID+EN, cek sinyal konkret (kode/file/URL/path), cek objek samar/multi-komponen. Satu fungsi, tanpa dependensi baru.

- [ ] **Step 4: Run test to verify it passes**

Run: sama seperti Step 2
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mimoly.py test_agent_responsiveness.py
git commit -m "feat: add _ambiguous_build_request heuristic for stall-prone build prompts"
```

### Task 2: Staged-query rewrite sebelum `upstream_payload`

**Files:**
- Modify: `mimoly.py` (~baris 3890-3942, antara `user_prompt` jadi dan `upstream_payload` dibangun)
- Test: `test_agent_responsiveness.py` (`NonstreamFrameProbeTests`, pakai `wire_client`)

**Interfaces:**
- Consumes: `_ambiguous_build_request` (Task 1), `user_prompt`, `tools`, `agent_framework`.
- Produces: `upstream_payload["query"]` berisi scaffold tahap-rencana saat ambiguous + tools present; apa adanya bila tidak.

Scaffold rewrite (lock, verbatim yang diminta):
- Awali: "Permintaan pembangunan di bawah ini besar/ambigu untuk satu turn. "
- Lalu: "Jabarkan dulu RENCANA langkah konkret (maks 5 langkah, 1 kalimat per langkah), JANGAN tulis kode dulu. "
- Lalu: "Permintaan asli user: " + user_prompt asli utuh.

- [ ] **Step 1: Write the failing test**

```python
async def test_ambiguous_build_prompt_is_staged_not_stalled(self):
    # Replay live 2026-10-16: "bikinin hp" stalls upstream (>300s) with tools
    # present. Gateway must rewrite the query into a plan-first scaffold so
    # the first turn returns, single attempt, user goal preserved.
    self.body['tools'] = TOOLS
    self.body['stream'] = False
    self.body['messages'] = [{'role': 'user', 'content': 'bikinin hp'}]
    plan = 'RENCANA: 1 LANGKAH_SENTINEL'
    client = self.wire_client([{'type': 'text', 'content': plan}, 'data: [DONE]'])
    result = await m.handle_chat_completion(self.body, request())
    output = await self.collect(result)
    self.assertEqual(result.status_code, 200)
    self.assertIn('LANGKAH_SENTINEL', output)
    sent_query = client.payloads[0]['query']
    self.assertIn('RENCANA', sent_query)
    self.assertIn('bikinin hp', sent_query)
    self.assertEqual(len(client.queries), 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/c/Users/Rafi/AppData/Local/hermes/hermes-agent/venv/Scripts/python -m pytest test_agent_responsiveness.py::NonstreamFrameProbeTests::test_ambiguous_build_prompt_is_staged_not_stalled -q`
Expected: FAIL — `sent_query` tidak mengandung `RENCANA` (rewrite belum ada)

- [ ] **Step 3: Implement rewrite di `mimoly.py` antara `user_prompt` jadi dan `upstream_payload`**

Hanya bila `_ambiguous_build_request(user_prompt)` True DAN `tools` present. Tambah `print("[mimoly] Staged decomposition rewrite ...")` untuk observabilitas. Jangan sentuh jalur plain-chat.

- [ ] **Step 4: Run test to verify it passes**

Run: sama seperti Step 2
Expected: PASS

- [ ] **Step 5: Tambah test non-regression untuk plain chat dan non-build**

Assert: prompt ambigu TANPA tools diteruskan apa adanya (query == pesan user, tanpa `RENCANA`); "buatkan puisi" dengan tools TIDAK di-rewrite.

- [ ] **Step 6: Commit**

```bash
git add mimoly.py test_agent_responsiveness.py
git commit -m "feat: stage ambiguous build prompts into plan-first queries"
```

### Task 3: Verifikasi penuh + merge

**Files:** tidak ada perubahan kode; hanya verifikasi.

- [ ] **Step 1: Run full suite**

Run: `/c/Users/Rafi/AppData/Local/hermes/hermes-agent/venv/Scripts/python -m pytest test_agent_responsiveness.py -q`
Expected: RED test Task 2 lulus; hanya fail flaky timing yang sudah dikenal (`test_cancel_...`, `test_global_deadline_...`) yang lolos saat run isolasi

- [ ] **Step 2: Run 5 runner lain + verify**

Run: `test_upstream_frames.py`, `test_reliability.py` (`all checks passed`), `pytest test_action_interceptor.py test_tool_rag.py test_rag_agent_e2e.py -q` (42 passed), `hermes verify --json --skip-start` (`ok: true`)

- [ ] **Step 3: Merge ke main**

```bash
git checkout main && git merge --no-ff fix/staged-build-decomposition
```

- [ ] **Step 4: Restart server 8080 dan live-test prompt PantauHarga via curl**

Expected: respons rencana ≤180 dtk (bukan 504), lalu live-test subagent PantauHarga di Hermia
